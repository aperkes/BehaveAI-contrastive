#!/usr/bin/env python3
"""Train and apply a track-supervised contrastive identity model.

The crop manifest produced by ``simple_track.py --export-crop-manifest`` is
used as weak supervision:

* detections in the same track are positive examples;
* tracks that coexist in at least one video frame are negative examples;
* non-overlapping tracks are not assumed to be different animals.

For training, source videos are decoded sequentially once and resized crops are
materialized as uint8 data. They are kept in RAM when space permits and in a
disk-backed array otherwise. On-demand video access remains available as an
explicit low-storage mode.

Optional median-background masking estimates one background per source video
from samples distributed across its full duration. The background pixels in
each detection crop are replaced with black before resizing or augmentation.

Typical commands::

    # Train and then cluster the tracks into four identities.
    python contrastive_identity.py --manifest crops.csv --video clip.mp4 \
        --num-individuals 4

    # Apply an existing network without further training.
    python contrastive_identity.py --mode predict --manifest crops.csv \
        --checkpoint model.pt --num-individuals 4

    # Fine-tune an existing network, or export its latent representation.
    python contrastive_identity.py --mode train --manifest new_crops.csv \
        --init-checkpoint model.pt
    python contrastive_identity.py --mode embed --manifest crops.csv \
        --checkpoint model.pt
"""

import argparse
import csv
import json
import os
import random
import shutil
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import BatchSampler, DataLoader, Dataset
except ImportError as exc:  # Keep --help and syntax inspection useful without PyTorch.
    torch = None
    nn = None
    F = None
    BatchSampler = object
    DataLoader = None
    Dataset = object
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


DEFAULT_EMBEDDING_DIM = 128
DEFAULT_INPUT_SIZE = 96


@dataclass(frozen=True)
class Detection:
    video: str
    frame: int
    track_id: str
    bbox: tuple
    manifest_index: int

    @property
    def track_key(self):
        # Track IDs are local to a video, so the video must be part of the key.
        return self.video, self.track_id


def require_torch():
    if torch is None:
        raise SystemExit(
            "PyTorch is required for contrastive training and inference. "
            "Install a PyTorch build suitable for this machine before running "
            f"this script. Original import error: {TORCH_IMPORT_ERROR}"
        )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def report_progress(label, completed, total, started_at):
    elapsed = perf_counter() - started_at
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else 0.0
    print(
        f"  {label}: {completed}/{total} ({completed / total:.1%}), "
        f"elapsed {format_duration(elapsed)}, ETA {format_duration(remaining)}",
        flush=True,
    )


def read_manifests(manifest_paths, video_overrides=None):
    """Read one or more crop manifests and return detections plus raw rows."""
    if video_overrides and len(video_overrides) != len(manifest_paths):
        raise ValueError("Provide one --video for each --manifest, or omit --video.")

    detections = []
    raw_rows = []
    for manifest_number, manifest_path in enumerate(manifest_paths):
        override = None
        if video_overrides:
            override = os.path.abspath(video_overrides[manifest_number])

        with open(manifest_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"source_video", "frame", "track_id", "x1", "y1", "x2", "y2"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{manifest_path} is missing manifest columns: {', '.join(sorted(missing))}"
                )

            for row in reader:
                video = override or os.path.abspath(row["source_video"])
                bbox = tuple(int(float(row[name])) for name in ("x1", "y1", "x2", "y2"))
                detection = Detection(
                    video=video,
                    frame=int(row["frame"]),
                    track_id=str(row["track_id"]),
                    bbox=bbox,
                    manifest_index=len(raw_rows),
                )
                detections.append(detection)
                copied_row = dict(row)
                copied_row["source_video"] = video
                raw_rows.append(copied_row)

    if not detections:
        raise ValueError("No detections were found in the supplied manifest(s).")

    missing_videos = sorted({item.video for item in detections if not os.path.isfile(item.video)})
    if missing_videos:
        preview = "\n  ".join(missing_videos[:5])
        raise FileNotFoundError(f"Source video file(s) not found:\n  {preview}")
    return detections, raw_rows


def build_track_metadata(detections):
    """Assign numeric labels and find track pairs known to be different."""
    track_keys = sorted({item.track_key for item in detections})
    track_to_label = {key: index for index, key in enumerate(track_keys)}
    indices_by_track = defaultdict(list)
    tracks_by_frame = defaultdict(set)
    seen_track_frames = set()

    for index, item in enumerate(detections):
        track_frame = (item.video, item.frame, item.track_id)
        if track_frame in seen_track_frames:
            raise ValueError(
                "Invalid crop manifest: track "
                f"{item.track_id} occurs more than once in frame {item.frame} of "
                f"{item.video}. Regenerate it with one detection per track per frame."
            )
        seen_track_frames.add(track_frame)
        label = track_to_label[item.track_key]
        indices_by_track[label].append(index)
        tracks_by_frame[(item.video, item.frame)].add(label)

    known_different = np.zeros((len(track_keys), len(track_keys)), dtype=bool)
    for labels in tracks_by_frame.values():
        labels = list(labels)
        for position, first in enumerate(labels):
            for second in labels[position + 1:]:
                known_different[first, second] = True
                known_different[second, first] = True

    return track_keys, track_to_label, indices_by_track, known_different


def letterbox_crop(image, size):
    """Resize without changing aspect ratio and pad to a square RGB image."""
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Cannot resize an empty crop.")
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x_offset = (size - resized_width) // 2
    y_offset = (size - resized_height) // 2
    canvas[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def lower_median_uint8(images):
    """Return a memory-efficient lower median for a list of uint8 images."""
    stack = np.stack(images, axis=0)
    middle = (len(stack) - 1) // 2
    stack.partition(middle, axis=0)
    return stack[middle].copy()


class MedianBackgroundStore:
    """Lazily construct and retain one robust median background per video."""

    def __init__(self, sample_count=1000, seed=42):
        self.sample_count = sample_count
        self.seed = seed
        self.backgrounds = {}

    def get(self, video):
        background = self.backgrounds.get(video)
        if background is None:
            background = self._build(video)
            self.backgrounds[video] = background
        return background

    def _build(self, video):
        capture = cv2.VideoCapture(video)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open source video for background estimation: {video}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            capture.release()
            raise RuntimeError(f"Video reports no frames: {video}")

        number_to_sample = min(self.sample_count, frame_count)
        positions = np.unique(
            np.linspace(0, frame_count - 1, number_to_sample, dtype=np.int64)
        )
        # An exact 1000-frame RGB median can require several GB. Splitting the
        # samples into roughly sqrt(N) groups bounds working RAM to O(sqrt(N))
        # full frames, while retaining a robust median-of-medians estimate.
        group_count = max(1, int(np.ceil(np.sqrt(len(positions)))))
        shuffled = positions.copy()
        np.random.default_rng(self.seed).shuffle(shuffled)
        groups = np.array_split(shuffled, group_count)
        group_medians = []

        print(
            f"Estimating median background from {len(positions)} frames: {video}",
            flush=True,
        )
        for group_number, group in enumerate(groups, start=1):
            frames = []
            for position in sorted(group.tolist()):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                ok, frame = capture.read()
                if ok:
                    frames.append(frame)
            if frames:
                group_medians.append(lower_median_uint8(frames))
            print(
                f"  background groups: {group_number}/{len(groups)}",
                end="\r" if group_number < len(groups) else "\n",
                flush=True,
            )
        capture.release()
        if not group_medians:
            raise RuntimeError(f"Could not decode samples for background estimation: {video}")
        return lower_median_uint8(group_medians)


def mask_to_moving_foreground(image, background, threshold, morph_kernel):
    """Keep the largest changed component and set the remaining pixels to black."""
    difference = cv2.absdiff(image, background)
    # Maximum channel difference avoids missing strongly colored foreground
    # whose luminance happens to be close to the background.
    difference = difference.max(axis=2)
    _, mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)

    if morph_kernel > 1:
        kernel_size = morph_kernel if morph_kernel % 2 else morph_kernel + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if component_count <= 1:
        return np.zeros_like(image)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels[..., None] == largest, image, 0).astype(np.uint8)


def prepare_detection_crop(
    frame,
    item,
    input_size,
    background=None,
    background_threshold=25,
    background_morph_kernel=3,
):
    """Extract, optionally mask, and resize one detection crop."""
    x1, y1, x2, y2 = item.bbox
    height, width = frame.shape[:2]
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid crop at frame {item.frame}, track {item.track_id}: {item.bbox}"
        )

    crop = frame[y1:y2, x1:x2]
    if background is not None:
        if background.shape != frame.shape:
            raise ValueError(
                f"Background shape {background.shape} does not match frame shape "
                f"{frame.shape} for {item.video}"
            )
        crop = mask_to_moving_foreground(
            crop,
            background[y1:y2, x1:x2],
            background_threshold,
            background_morph_kernel,
        )
    return letterbox_crop(crop, input_size)


class PreloadedCropStore:
    """Decode videos once into a RAM array or disk-backed memory map."""

    def __init__(
        self,
        detections,
        input_size,
        background_store=None,
        background_threshold=25,
        background_morph_kernel=3,
        storage="auto",
        cache_path=None,
    ):
        self.detections = detections
        self.input_size = input_size
        self.background_store = background_store
        self.background_threshold = background_threshold
        self.background_morph_kernel = background_morph_kernel
        self.shape = (len(self.detections), self.input_size, self.input_size, 3)
        self.required_bytes = int(np.prod(self.shape, dtype=np.int64))
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.storage = self._resolve_storage(storage)
        self.crops = None

    def _resolve_storage(self, requested):
        if requested != "auto":
            return requested
        try:
            import psutil

            available = psutil.virtual_memory().available
        except ImportError:
            available = 0
        reserve = max(2 * 1024 ** 3, self.required_bytes // 4)
        selected = "ram" if available >= self.required_bytes + reserve else "disk"
        print(
            f"Automatic crop storage selected {selected.upper()}: crops require "
            f"{self.required_bytes / 1024 ** 3:.2f} GiB and "
            f"{available / 1024 ** 3:.2f} GiB RAM is currently available.",
            flush=True,
        )
        return selected

    def get(self, index):
        if self.crops is None:
            self._build()
        return self.crops[index]

    def _build(self):
        gib = self.required_bytes / 1024 ** 3
        if self.storage == "ram":
            print(f"Preloading {len(self.detections)} crops into {gib:.2f} GiB of RAM...")
            self.crops = np.empty(self.shape, dtype=np.uint8)
        elif self.storage == "disk":
            if self.cache_path is None:
                raise ValueError("A cache path is required for disk crop storage.")
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            free_bytes = shutil.disk_usage(self.cache_path.parent).free
            if free_bytes < self.required_bytes:
                raise RuntimeError(
                    f"Disk crop cache requires {gib:.2f} GiB but only "
                    f"{free_bytes / 1024 ** 3:.2f} GiB is free at "
                    f"{self.cache_path.parent}. Use --crop-storage on-demand or free space."
                )
            print(
                f"Saving {len(self.detections)} crops to a {gib:.2f} GiB "
                f"memory-mapped cache: {self.cache_path}"
            )
            self.crops = np.lib.format.open_memmap(
                self.cache_path,
                mode="w+",
                dtype=np.uint8,
                shape=self.shape,
            )
        else:
            raise ValueError(f"Unsupported crop storage mode: {self.storage}")

        detections_by_video_and_frame = defaultdict(lambda: defaultdict(list))
        for index, item in enumerate(self.detections):
            detections_by_video_and_frame[item.video][item.frame].append((index, item))

        completed = 0
        next_report = 5
        for video, detections_by_frame in detections_by_video_and_frame.items():
            capture = cv2.VideoCapture(video)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open source video while preloading: {video}")
            background = (
                self.background_store.get(video)
                if self.background_store is not None
                else None
            )
            last_needed_frame = max(detections_by_frame)
            for frame_number in range(1, last_needed_frame + 1):
                ok, frame = capture.read()
                if not ok:
                    capture.release()
                    raise RuntimeError(f"Could not decode frame {frame_number} from {video}")
                for index, item in detections_by_frame.get(frame_number, ()):
                    self.crops[index] = prepare_detection_crop(
                        frame,
                        item,
                        self.input_size,
                        background,
                        self.background_threshold,
                        self.background_morph_kernel,
                    )
                    completed += 1
                percent = 100 * completed / len(self.detections)
                if percent >= next_report:
                    print(f"  crop preload: {percent:.1f}%", flush=True)
                    next_report += 5
            capture.release()
        if isinstance(self.crops, np.memmap):
            self.crops.flush()
        print("Crop preload complete.")


def make_preloaded_crop_store(args, detections, input_size, background_store):
    return PreloadedCropStore(
        detections,
        input_size,
        background_store,
        args.background_threshold,
        args.background_morph_kernel,
        storage=args.crop_storage,
        cache_path=Path(args.output_dir) / "crop_cache" / "crops.npy",
    )


class VideoCropDataset(Dataset):
    """Read video crops lazily and retain a bounded cache of resized uint8 crops."""

    def __init__(
        self,
        detections,
        track_to_label,
        input_size,
        cache_size,
        augment=False,
        background_store=None,
        background_threshold=25,
        background_morph_kernel=3,
        preloaded_crop_store=None,
    ):
        self.detections = detections
        self.labels = [track_to_label[item.track_key] for item in detections]
        self.input_size = input_size
        self.cache_size = max(0, cache_size)
        self.augment = augment
        self.background_store = background_store
        self.background_threshold = background_threshold
        self.background_morph_kernel = background_morph_kernel
        self.preloaded_crop_store = preloaded_crop_store
        self.crop_cache = OrderedDict()
        self.captures = {}
        self.last_frame_key = None
        self.last_frame = None

    def __len__(self):
        return len(self.detections)

    def _capture(self, video):
        capture = self.captures.get(video)
        if capture is None:
            capture = cv2.VideoCapture(video)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open source video: {video}")
            self.captures[video] = capture
        return capture

    def _read_crop(self, index):
        if self.preloaded_crop_store is not None:
            return self.preloaded_crop_store.get(index)

        cached = self.crop_cache.get(index)
        if cached is not None:
            self.crop_cache.move_to_end(index)
            return cached

        item = self.detections[index]
        frame_key = (item.video, item.frame)
        if frame_key == self.last_frame_key:
            frame = self.last_frame
        else:
            capture = self._capture(item.video)
            # OpenCV frame positions are zero-based; manifest frames are one-based.
            capture.set(cv2.CAP_PROP_POS_FRAMES, item.frame - 1)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {item.frame} from {item.video}")
            self.last_frame_key = frame_key
            self.last_frame = frame

        background = (
            self.background_store.get(item.video)
            if self.background_store is not None
            else None
        )
        crop = prepare_detection_crop(
            frame,
            item,
            self.input_size,
            background,
            self.background_threshold,
            self.background_morph_kernel,
        )

        if self.cache_size:
            self.crop_cache[index] = crop
            self.crop_cache.move_to_end(index)
            while len(self.crop_cache) > self.cache_size:
                self.crop_cache.popitem(last=False)
        return crop

    def _augment(self, tensor):
        if not self.augment:
            return tensor
        if torch.rand(()) < 0.5:
            tensor = torch.flip(tensor, dims=(2,))
        # Mild photometric variation preserves markings better than aggressive jitter.
        brightness = 1.0 + float(torch.empty(1).uniform_(-0.12, 0.12))
        contrast = 1.0 + float(torch.empty(1).uniform_(-0.12, 0.12))
        channel_mean = tensor.mean(dim=(1, 2), keepdim=True)
        tensor = (tensor - channel_mean) * contrast + channel_mean
        return (tensor * brightness).clamp(0.0, 1.0)

    def __getitem__(self, index):
        crop = self._read_crop(index)
        tensor = torch.from_numpy(crop.copy()).permute(2, 0, 1).float().div_(255.0)
        tensor = self._augment(tensor)
        return tensor, self.labels[index], index

    def close(self):
        for capture in self.captures.values():
            capture.release()
        self.captures.clear()
        self.last_frame_key = None
        self.last_frame = None

    def __del__(self):
        self.close()


class TrackBatchSampler(BatchSampler):
    """Construct batches with multiple crops per track and useful negatives."""

    def __init__(
        self,
        indices_by_track,
        known_different,
        tracks_per_batch,
        samples_per_track,
        batches_per_epoch,
    ):
        self.indices_by_track = dict(indices_by_track)
        self.known_different = known_different
        self.track_labels = list(self.indices_by_track)
        self.tracks_per_batch = min(tracks_per_batch, len(self.track_labels))
        self.samples_per_track = samples_per_track
        self.batches_per_epoch = batches_per_epoch

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            anchor = random.choice(self.track_labels)
            negative_candidates = np.flatnonzero(self.known_different[anchor]).tolist()
            random.shuffle(negative_candidates)
            selected = [anchor]
            selected.extend(negative_candidates[: self.tracks_per_batch - 1])

            if len(selected) < self.tracks_per_batch:
                remaining = [label for label in self.track_labels if label not in selected]
                selected.extend(random.sample(remaining, self.tracks_per_batch - len(selected)))

            batch = []
            for label in selected:
                candidates = self.indices_by_track[label]
                if len(candidates) >= self.samples_per_track:
                    batch.extend(random.sample(candidates, self.samples_per_track))
                else:
                    batch.extend(random.choices(candidates, k=self.samples_per_track))
            random.shuffle(batch)
            yield batch


_ModuleBase = nn.Module if nn is not None else object


class IdentityEncoder(_ModuleBase):
    """Small CNN with separate embedding and training projection outputs."""

    def __init__(self, embedding_dim=DEFAULT_EMBEDDING_DIM, projection_dim=64):
        if nn is None:
            require_torch()
        super().__init__()
        channels = (32, 64, 128, 256)
        blocks = []
        in_channels = 3
        for out_channels in channels:
            blocks.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_channels = out_channels
        self.backbone = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))
        self.embedding = nn.Linear(channels[-1], embedding_dim)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, images, project=True):
        features = self.backbone(images).flatten(1)
        embedding = F.normalize(self.embedding(features), dim=1)
        if not project:
            return embedding
        return F.normalize(self.projector(embedding), dim=1)


def track_contrastive_loss(projected, labels, known_different, margin):
    """Pairwise cosine loss with unknown track relationships masked out."""
    similarity = projected @ projected.T
    distance = 1.0 - similarity
    same_track = labels[:, None].eq(labels[None, :])
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = same_track & ~diagonal
    negative_lookup = known_different.to(labels.device)
    negative_mask = negative_lookup[labels[:, None], labels[None, :]] & ~diagonal

    positive_loss = distance[positive_mask].mean() if positive_mask.any() else distance.sum() * 0
    negative_loss = (
        F.relu(margin - distance[negative_mask]).square().mean()
        if negative_mask.any()
        else distance.sum() * 0
    )
    return positive_loss + negative_loss, positive_loss, negative_loss


def checkpoint_configuration(checkpoint):
    return checkpoint.get("model_config", {})


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions predating the weights_only argument.
        return torch.load(path, map_location=device)


def resolve_preprocessing(args, checkpoint_path, device):
    """Use explicit CLI values, then checkpoint values, then safe defaults."""
    stored = {}
    if checkpoint_path:
        stored = load_checkpoint(checkpoint_path, device).get("preprocessing", {})

    values = {
        "background_subtraction": "none",
        "background_samples": 1000,
        "background_threshold": 25,
        "background_morph_kernel": 3,
    }
    values.update(stored)
    for name, value in values.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def make_model(args, device, checkpoint_path=None):
    checkpoint = load_checkpoint(checkpoint_path, device) if checkpoint_path else None
    stored = checkpoint_configuration(checkpoint) if checkpoint else {}
    embedding_dim = stored.get("embedding_dim", args.embedding_dim or DEFAULT_EMBEDDING_DIM)
    projection_dim = stored.get("projection_dim", args.projection_dim)
    input_size = stored.get("input_size", args.input_size or DEFAULT_INPUT_SIZE)
    model = IdentityEncoder(embedding_dim, projection_dim).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    return model, embedding_dim, projection_dim, input_size


def train_model(
    args,
    detections,
    metadata,
    device,
    background_store,
    preloaded_crop_store=None,
):
    track_keys, track_to_label, indices_by_track, known_different = metadata
    model, embedding_dim, projection_dim, input_size = make_model(
        args, device, args.init_checkpoint
    )
    if args.crop_storage != "on-demand" and preloaded_crop_store is None:
        preloaded_crop_store = make_preloaded_crop_store(
            args, detections, input_size, background_store
        )
    dataset = VideoCropDataset(
        detections,
        track_to_label,
        input_size=input_size,
        cache_size=args.cache_crops,
        augment=True,
        background_store=background_store,
        background_threshold=args.background_threshold,
        background_morph_kernel=args.background_morph_kernel,
        preloaded_crop_store=preloaded_crop_store,
    )
    effective_tracks_per_batch = min(args.tracks_per_batch, len(indices_by_track))
    effective_batch_size = effective_tracks_per_batch * args.samples_per_track
    batches_per_epoch = args.batches_per_epoch or max(
        1, int(np.ceil(len(detections) / effective_batch_size))
    )
    sampler = TrackBatchSampler(
        indices_by_track,
        known_different,
        effective_tracks_per_batch,
        args.samples_per_track,
        batches_per_epoch,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    known_different_tensor = torch.from_numpy(known_different)

    if preloaded_crop_store is not None:
        print(
            "Crop loading: sequentially materialize every resized crop in "
            f"{preloaded_crop_store.storage.upper()} before the first training batch.",
            flush=True,
        )
    else:
        print(
            f"Crop loading: on demand from video with a {args.cache_crops}-crop RAM "
            "cache. Random video seeks can be slow.",
            flush=True,
        )

    if not known_different.any():
        print(
            "Warning: no temporally overlapping tracks were found. Training has positives "
            "but no reliable negative pairs, so identity separation will be weak."
        )

    print(
        f"Training on {len(detections)} detections from {len(track_keys)} tracks "
        f"using {device}. Effective batch size: "
        f"{effective_batch_size} images "
        f"({effective_tracks_per_batch} tracks x {args.samples_per_track} samples)."
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = np.zeros(3, dtype=float)
        epoch_started = perf_counter()
        last_report = epoch_started
        for batch_number, (images, labels, _) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            projected = model(images, project=True)
            loss, positive_loss, negative_loss = track_contrastive_loss(
                projected, labels, known_different_tensor, args.margin
            )
            loss.backward()
            optimizer.step()
            totals += (loss.item(), positive_loss.item(), negative_loss.item())
            now = perf_counter()
            if batch_number == 1 or batch_number == len(loader) or now - last_report >= 10:
                report_progress(
                    f"epoch {epoch:03d} batches",
                    batch_number,
                    len(loader),
                    epoch_started,
                )
                last_report = now

        means = totals / len(loader)
        print(
            f"Epoch {epoch:03d}/{args.epochs}: loss={means[0]:.5f} "
            f"positive_loss={means[1]:.5f} negative_loss={means[2]:.5f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "contrastive_identity.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "embedding_dim": embedding_dim,
                "projection_dim": projection_dim,
                "input_size": input_size,
            },
            "preprocessing": {
                "background_subtraction": args.background_subtraction,
                "background_samples": args.background_samples,
                "background_threshold": args.background_threshold,
                "background_morph_kernel": args.background_morph_kernel,
            },
            "epochs": args.epochs,
        },
        checkpoint_path,
    )
    dataset.close()
    print(f"Saved network: {checkpoint_path}")
    return model, input_size, checkpoint_path, preloaded_crop_store


def extract_embeddings(
    args,
    model,
    input_size,
    detections,
    metadata,
    device,
    background_store,
    preloaded_crop_store,
):
    _, track_to_label, _, _ = metadata
    dataset = VideoCropDataset(
        detections,
        track_to_label,
        input_size=input_size,
        cache_size=args.cache_crops,
        augment=False,
        background_store=background_store,
        background_threshold=args.background_threshold,
        background_morph_kernel=args.background_morph_kernel,
        preloaded_crop_store=preloaded_crop_store,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.inference_batch_size,
        shuffle=False,
        num_workers=0,
    )
    embeddings = []
    model.eval()
    if preloaded_crop_store is not None:
        print("Embedding crop loading: preloaded RAM store.", flush=True)
    else:
        print(
            f"Embedding crop loading: on demand with a {args.cache_crops}-crop RAM cache.",
            flush=True,
        )
    extraction_started = perf_counter()
    last_report = extraction_started
    with torch.no_grad():
        for batch_number, (images, _, _) in enumerate(loader, start=1):
            vectors = model(images.to(device), project=False)
            embeddings.append(vectors.cpu().numpy())
            now = perf_counter()
            if batch_number == 1 or batch_number == len(loader) or now - last_report >= 10:
                report_progress(
                    "embedding batches",
                    batch_number,
                    len(loader),
                    extraction_started,
                )
                last_report = now
    dataset.close()
    return np.concatenate(embeddings, axis=0)


def aggregate_tracks(embeddings, detections, track_to_label, track_count):
    track_embeddings = np.zeros((track_count, embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(track_count, dtype=np.int64)
    for vector, item in zip(embeddings, detections):
        label = track_to_label[item.track_key]
        track_embeddings[label] += vector
        counts[label] += 1
    track_embeddings /= counts[:, None]
    norms = np.linalg.norm(track_embeddings, axis=1, keepdims=True)
    track_embeddings /= np.maximum(norms, 1e-12)
    return track_embeddings, counts


def export_crop_samples(
    output_dir,
    sample_count,
    detections,
    track_to_label,
    input_size,
    background_store,
    background_threshold,
    background_morph_kernel,
    preloaded_crop_store=None,
):
    """Export deterministic, non-augmented examples of network input crops."""
    sample_count = min(sample_count, len(detections))
    indices = np.unique(
        np.linspace(0, len(detections) - 1, sample_count, dtype=np.int64)
    )
    sample_dir = Path(output_dir) / "crop_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    dataset = VideoCropDataset(
        detections,
        track_to_label,
        input_size=input_size,
        cache_size=0,
        augment=False,
        background_store=background_store,
        background_threshold=background_threshold,
        background_morph_kernel=background_morph_kernel,
        preloaded_crop_store=preloaded_crop_store,
    )
    rows = []
    try:
        for sample_number, index in enumerate(indices, start=1):
            item = detections[int(index)]
            safe_track_id = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in item.track_id
            )
            filename = (
                f"sample_{sample_number:03d}_frame_{item.frame}_"
                f"track_{safe_track_id}.png"
            )
            crop = dataset._read_crop(int(index))
            if not cv2.imwrite(
                str(sample_dir / filename), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            ):
                raise RuntimeError(f"Could not write crop sample: {sample_dir / filename}")
            rows.append(
                {
                    "crop_file": filename,
                    "source_video": item.video,
                    "frame": item.frame,
                    "track_id": item.track_id,
                    "x1": item.bbox[0],
                    "y1": item.bbox[1],
                    "x2": item.bbox[2],
                    "y2": item.bbox[3],
                }
            )
    finally:
        dataset.close()

    manifest_path = sample_dir / "crop_samples.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} crop samples: {sample_dir}")
    return sample_dir


def _normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def cosine_kmeans(
    vectors,
    number_of_individuals,
    seed,
    initial_centers=None,
    restarts=20,
):
    """Cluster unit vectors and return assignments, centers, and similarities."""
    if number_of_individuals < 1:
        raise ValueError("--num-individuals must be at least 1.")
    if number_of_individuals > len(vectors):
        raise ValueError(
            f"Requested {number_of_individuals} individuals but only "
            f"{len(vectors)} tracks are available."
        )

    vectors = _normalize_rows(vectors)
    generator = np.random.default_rng(seed)
    best_assignments = None
    best_centers = None
    best_cost = np.inf
    attempts = 1 if initial_centers is not None else restarts
    for _ in range(attempts):
        if initial_centers is None:
            initial = generator.choice(
                len(vectors), size=number_of_individuals, replace=False
            )
            centers = vectors[initial].copy()
        else:
            centers = _normalize_rows(initial_centers)
        assignments = np.full(len(vectors), -1, dtype=np.int64)

        for _ in range(200):
            similarities = vectors @ centers.T
            updated = similarities.argmax(axis=1)
            if np.array_equal(updated, assignments):
                break
            assignments = updated

            for identity in range(number_of_individuals):
                members = vectors[assignments == identity]
                if len(members):
                    center = members.mean(axis=0)
                else:
                    # Re-seed an empty cluster with the least well represented track.
                    center = vectors[np.argmin(similarities.max(axis=1))]
                centers[identity] = center / max(np.linalg.norm(center), 1e-12)

        similarities = vectors @ centers.T
        cost = float(np.sum(1.0 - similarities[np.arange(len(assignments)), assignments]))
        if cost < best_cost:
            best_cost = cost
            best_assignments = assignments.copy()
            best_centers = centers.copy()

    similarities = vectors @ best_centers.T
    return best_assignments, best_centers, similarities


def cluster_identities(track_embeddings, number_of_individuals, seed):
    """Original unconstrained track-level clustering baseline."""
    assignments, _, _ = cosine_kmeans(
        track_embeddings, number_of_individuals, seed
    )
    return assignments


def find_reference_global_fragment(
    detections,
    track_to_label,
    number_of_individuals,
    minimum_frames,
):
    """Return the strongest full-animal group whose tracks are long enough."""
    tracks_by_frame = defaultdict(set)
    track_lengths = defaultdict(int)
    for item in detections:
        label = track_to_label[item.track_key]
        tracks_by_frame[(item.video, item.frame)].add(label)
        track_lengths[label] += 1

    frames_by_group = defaultdict(list)
    for frame_key, labels in tracks_by_frame.items():
        if len(labels) == number_of_individuals:
            frames_by_group[(frame_key[0], tuple(sorted(labels)))].append(frame_key[1])

    candidates = [
        (key, frames)
        for key, frames in frames_by_group.items()
        if min(track_lengths[label] for label in key[1]) >= minimum_frames
    ]
    if not candidates:
        return None
    (video, labels), frames = max(
        candidates,
        key=lambda item: (
            len(item[1]),
            min(track_lengths[label] for label in item[0][1]),
        ),
    )
    return {
        "video": video,
        "track_labels": labels,
        "coexisting_frame_count": len(frames),
        "minimum_track_detections": min(track_lengths[label] for label in labels),
        "first_frame": min(frames),
        "last_frame": max(frames),
    }


def fragment_identity_probabilities(
    detection_assignments,
    detection_track_labels,
    track_count,
    number_of_individuals,
):
    """Convert per-image cluster votes into idtracker-style fragment P1 values."""
    frequencies = np.zeros((track_count, number_of_individuals), dtype=np.float64)
    np.add.at(frequencies, (detection_track_labels, detection_assignments), 1)
    logits = frequencies * np.log(2.0)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def assign_with_cooccurrence(p1, known_different, fixed_assignments=None):
    """Assign fragments in confidence order while enforcing temporal exclusivity."""
    p1 = np.asarray(p1, dtype=np.float64).copy()
    track_count, number_of_individuals = p1.shape
    assignments = np.full(track_count, -1, dtype=np.int64)
    confidences = np.zeros(track_count, dtype=np.float64)
    remaining = set(range(track_count))

    for track, identity in (fixed_assignments or {}).items():
        assignments[track] = identity
        confidences[track] = 1.0
        p1[track] = 0.0
        p1[track, identity] = 1.0
        remaining.discard(track)

    while remaining:
        best = None
        for track in remaining:
            neighbors = np.flatnonzero(known_different[track])
            if len(neighbors):
                adjusted = p1[track] * np.prod(1.0 - p1[neighbors], axis=0)
            else:
                adjusted = p1[track].copy()
            total = adjusted.sum()
            if total > 0:
                adjusted /= total
            ordered = np.sort(adjusted)
            certainty = (
                np.inf
                if number_of_individuals == 1 or ordered[-2] == 0 < ordered[-1]
                else (ordered[-1] / ordered[-2] if ordered[-2] > 0 else 0.0)
            )
            candidate = (certainty, float(adjusted.max()), -track, track, adjusted)
            if best is None or candidate[:3] > best[:3]:
                best = candidate

        _, max_probability, _, track, adjusted = best
        winners = np.flatnonzero(np.isclose(adjusted, max_probability))
        if max_probability > 0 and len(winners) == 1:
            identity = int(winners[0])
            assignments[track] = identity
            confidences[track] = max_probability
            p1[track] = 0.0
            p1[track, identity] = 1.0
        remaining.remove(track)

    return assignments, confidences


def enforce_identity_exclusivity(assignments, confidences, known_different):
    """Hard final guard: coexisting fragments may not retain the same identity."""
    assignments = np.asarray(assignments, dtype=np.int64).copy()
    confidences = np.asarray(confidences, dtype=np.float64).copy()
    demoted = []
    for identity in sorted(set(assignments[assignments >= 0].tolist())):
        candidates = np.flatnonzero(assignments == identity)
        candidates = sorted(candidates, key=lambda track: (-confidences[track], track))
        accepted = []
        for track in candidates:
            if accepted and known_different[track, accepted].any():
                assignments[track] = -1
                confidences[track] = 0.0
                demoted.append(track)
            else:
                accepted.append(track)

    assigned_tracks = np.flatnonzero(assignments >= 0)
    for position, first in enumerate(assigned_tracks):
        conflicts = assigned_tracks[position + 1 :]
        same_identity = assignments[conflicts] == assignments[first]
        assert not known_different[first, conflicts[same_identity]].any()
    return assignments, confidences, demoted


def idtracker_identity_assignment(
    embeddings,
    track_embeddings,
    detections,
    track_to_label,
    known_different,
    number_of_individuals,
    seed,
    global_fragment_min_frames,
):
    """Cluster detections, then assign whole fragments using video constraints."""
    reference = find_reference_global_fragment(
        detections,
        track_to_label,
        number_of_individuals,
        global_fragment_min_frames,
    )
    initial_centers = None
    fixed_assignments = {}
    if reference is not None:
        initial_centers = track_embeddings[list(reference["track_labels"])]
        fixed_assignments = {
            track: identity
            for identity, track in enumerate(reference["track_labels"])
        }

    _, centers, _ = cosine_kmeans(
        track_embeddings,
        number_of_individuals,
        seed,
        initial_centers=initial_centers,
    )
    detection_assignments = (_normalize_rows(embeddings) @ centers.T).argmax(axis=1)
    detection_track_labels = np.asarray(
        [track_to_label[item.track_key] for item in detections], dtype=np.int64
    )
    p1 = fragment_identity_probabilities(
        detection_assignments,
        detection_track_labels,
        len(track_embeddings),
        number_of_individuals,
    )
    assignments, confidences = assign_with_cooccurrence(
        p1, known_different, fixed_assignments
    )
    assignments, confidences, demoted = enforce_identity_exclusivity(
        assignments, confidences, known_different
    )
    if demoted:
        print(
            f"Final identity-exclusivity guard demoted {len(demoted)} conflicting "
            "track fragments to -1."
        )
    return assignments, confidences, p1, reference


def project_embeddings(
    detection_embeddings,
    track_embeddings,
    method,
    dimensions,
    fit_sample_limit,
    umap_neighbors,
    umap_min_dist,
    seed,
):
    """Project detections and track means into one shared visualization space."""
    combined = np.concatenate((detection_embeddings, track_embeddings), axis=0)
    generator = np.random.default_rng(seed)
    if len(combined) > fit_sample_limit:
        fit_indices = generator.choice(len(combined), fit_sample_limit, replace=False)
        fit_vectors = combined[fit_indices]
    else:
        fit_vectors = combined

    if method == "umap":
        if len(fit_vectors) < 3:
            print("Fewer than three embeddings are available; using PCA instead of UMAP.")
            method = "pca"
        else:
            try:
                import umap
            except Exception as exc:
                raise SystemExit(
                    "UMAP latent export requires a working optional 'umap-learn' "
                    "installation. Install it with 'pip install umap-learn', or use "
                    f"--latent-projection pca. UMAP import failed with: {exc}"
                ) from exc

            reducer = umap.UMAP(
                n_components=dimensions,
                n_neighbors=min(umap_neighbors, len(fit_vectors) - 1),
                min_dist=umap_min_dist,
                metric="cosine",
                random_state=seed,
            )
            reducer.fit(fit_vectors)
            projected = reducer.transform(combined)
            split = len(detection_embeddings)
            return projected[:split], projected[split:], "umap"

    if method == "pca":
        mean = fit_vectors.mean(axis=0, keepdims=True)
        _, _, axes = np.linalg.svd(fit_vectors - mean, full_matrices=False)
        available_dimensions = min(dimensions, len(axes))
        projected = (combined - mean) @ axes[:available_dimensions].T
        if available_dimensions < dimensions:
            projected = np.pad(
                projected,
                ((0, 0), (0, dimensions - available_dimensions)),
            )
        split = len(detection_embeddings)
        return projected[:split], projected[split:], "pca"

    raise ValueError(f"Unsupported latent projection method: {method}")


def write_identity_rows(path, raw_rows, detections, track_to_label, identity_by_track):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(raw_rows[0]) + ["predicted_identity"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, item in zip(raw_rows, detections):
            output = dict(row)
            output["predicted_identity"] = int(identity_by_track[track_to_label[item.track_key]])
            writer.writerow(output)


def write_latents(
    path,
    embeddings,
    detections,
    track_to_label,
    identity_by_track=None,
    projection=None,
    projection_name=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coordinate_names = [f"latent_{index:03d}" for index in range(embeddings.shape[1])]
    fields = ["source_video", "frame", "track_id", "x1", "y1", "x2", "y2"]
    if identity_by_track is not None:
        fields.append("predicted_identity")
    if projection is not None:
        fields.extend(
            f"{projection_name}_{index + 1}" for index in range(projection.shape[1])
        )
    fields.extend(coordinate_names)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for index, (vector, item) in enumerate(zip(embeddings, detections)):
            row = [item.video, item.frame, item.track_id, *item.bbox]
            if identity_by_track is not None:
                row.append(int(identity_by_track[track_to_label[item.track_key]]))
            if projection is not None:
                row.extend(f"{value:.8g}" for value in projection[index])
            row.extend(f"{value:.8g}" for value in vector)
            writer.writerow(row)


def write_track_latents(
    path,
    track_embeddings,
    track_keys,
    counts,
    identity_by_track=None,
    projection=None,
    projection_name=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coordinate_names = [f"latent_{index:03d}" for index in range(track_embeddings.shape[1])]
    fields = ["source_video", "track_id", "detection_count"]
    if identity_by_track is not None:
        fields.append("predicted_identity")
    if projection is not None:
        fields.extend(
            f"{projection_name}_{index + 1}" for index in range(projection.shape[1])
        )
    fields.extend(coordinate_names)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for label, ((video, track_id), vector) in enumerate(zip(track_keys, track_embeddings)):
            row = [video, track_id, int(counts[label])]
            if identity_by_track is not None:
                row.append(int(identity_by_track[label]))
            if projection is not None:
                row.extend(f"{value:.8g}" for value in projection[label])
            row.extend(f"{value:.8g}" for value in vector)
            writer.writerow(row)


def write_assignment_diagnostics(
    path,
    track_keys,
    counts,
    assignments,
    confidences,
    p1=None,
):
    path = Path(path)
    fields = [
        "source_video",
        "track_id",
        "detection_count",
        "predicted_identity",
        "assignment_confidence",
    ]
    if p1 is not None:
        fields.extend(f"cluster_probability_{index}" for index in range(p1.shape[1]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for label, (video, track_id) in enumerate(track_keys):
            row = [
                video,
                track_id,
                int(counts[label]),
                int(assignments[label]),
                f"{confidences[label]:.8g}",
            ]
            if p1 is not None:
                row.extend(f"{value:.8g}" for value in p1[label])
            writer.writerow(row)


def assignment_quality(detections, track_to_label, assignments, confidences, reference):
    tracks_by_frame = defaultdict(set)
    for item in detections:
        tracks_by_frame[(item.video, item.frame)].add(track_to_label[item.track_key])

    multi_frames = 0
    collision_frames = 0
    for labels in tracks_by_frame.values():
        if len(labels) < 2:
            continue
        multi_frames += 1
        identities = [assignments[label] for label in labels if assignments[label] >= 0]
        if len(identities) != len(set(identities)):
            collision_frames += 1

    assigned = assignments >= 0
    assigned_detections = sum(
        assignments[track_to_label[item.track_key]] >= 0 for item in detections
    )
    return {
        "track_count": int(len(assignments)),
        "assigned_tracks": int(assigned.sum()),
        "unassigned_tracks": int((~assigned).sum()),
        "assigned_track_fraction": float(assigned.mean()),
        "detection_count": int(len(detections)),
        "assigned_detections": int(assigned_detections),
        "assigned_detection_fraction": float(assigned_detections / len(detections)),
        "mean_assigned_confidence": (
            float(confidences[assigned].mean()) if assigned.any() else 0.0
        ),
        "multi_detection_frames": int(multi_frames),
        "identity_collision_frames": int(collision_frames),
        "identity_collision_frame_fraction": (
            float(collision_frames / multi_frames) if multi_frames else 0.0
        ),
        "reference_global_fragment": reference,
    }


def run_inference(
    args,
    model,
    input_size,
    detections,
    raw_rows,
    metadata,
    device,
    background_store,
    preloaded_crop_store,
):
    track_keys, track_to_label, _, known_different = metadata
    embeddings = extract_embeddings(
        args,
        model,
        input_size,
        detections,
        metadata,
        device,
        background_store,
        preloaded_crop_store,
    )
    track_embeddings, track_counts = aggregate_tracks(
        embeddings, detections, track_to_label, len(track_keys)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_by_track = None
    if args.mode != "embed":
        if args.num_individuals is None:
            raise ValueError("--num-individuals is required when predicting identities.")
        if args.identity_assignment == "idtracker":
            identity_by_track, assignment_confidence, p1, reference = (
                idtracker_identity_assignment(
                    embeddings,
                    track_embeddings,
                    detections,
                    track_to_label,
                    known_different,
                    args.num_individuals,
                    args.seed,
                    args.global_fragment_min_frames,
                )
            )
            if reference is None:
                print(
                    "No sufficiently large full-animal frame group was found; "
                    "using K-means++ initialization with co-occurrence-aware assignment."
                )
            else:
                print(
                    "Initialized identities from a reference group containing "
                    "all animals; its shortest track has "
                    f"{reference['minimum_track_detections']} detections."
                )
        else:
            identity_by_track = cluster_identities(
                track_embeddings, args.num_individuals, args.seed
            )
            assignment_confidence = np.ones(len(track_embeddings), dtype=np.float64)
            p1 = None
            reference = None
        identity_path = output_dir / "identity_predictions.csv"
        write_identity_rows(
            identity_path, raw_rows, detections, track_to_label, identity_by_track
        )
        diagnostics_path = output_dir / "identity_assignment_tracks.csv"
        write_assignment_diagnostics(
            diagnostics_path,
            track_keys,
            track_counts,
            identity_by_track,
            assignment_confidence,
            p1,
        )
        quality = assignment_quality(
            detections,
            track_to_label,
            identity_by_track,
            assignment_confidence,
            reference,
        )
        quality["assignment_method"] = args.identity_assignment
        quality_path = output_dir / "identity_assignment_quality.json"
        quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
        print(f"Saved identity predictions: {identity_path}")
        print(f"Saved assignment diagnostics: {diagnostics_path}")
        print(f"Saved assignment quality report: {quality_path}")
        print(
            f"Assigned {quality['assigned_tracks']}/{quality['track_count']} tracks; "
            f"identity collisions occur in "
            f"{quality['identity_collision_frame_fraction']:.1%} of multi-detection frames."
        )

    if args.mode == "embed" or args.export_latents:
        latent_path = Path(args.export_latents) if args.export_latents else output_dir / "latents.csv"
        track_latent_path = latent_path.with_name(latent_path.stem + "_tracks" + latent_path.suffix)
        detection_projection = None
        track_projection = None
        projection_name = None
        if args.latent_projection != "none":
            print(
                f"Computing {args.latent_projection.upper()} projection of latent vectors...",
                flush=True,
            )
            detection_projection, track_projection, projection_name = project_embeddings(
                embeddings,
                track_embeddings,
                method=args.latent_projection,
                dimensions=args.visualization_dimensions,
                fit_sample_limit=args.projection_fit_samples,
                umap_neighbors=args.umap_neighbors,
                umap_min_dist=args.umap_min_dist,
                seed=args.seed,
            )
        write_latents(
            latent_path,
            embeddings,
            detections,
            track_to_label,
            identity_by_track,
            detection_projection,
            projection_name,
        )
        write_track_latents(
            track_latent_path,
            track_embeddings,
            track_keys,
            track_counts,
            identity_by_track,
            track_projection,
            projection_name,
        )
        print(f"Saved detection embeddings: {latent_path}")
        print(f"Saved track-mean embeddings: {track_latent_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train embeddings from BehaveAI tracks and cluster them into identities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("train-predict", "train", "predict", "embed", "sample-crops"),
        default="train-predict",
        help="operation to perform",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="crop manifest; repeat this option to train on additional videos",
    )
    parser.add_argument(
        "--video",
        action="append",
        help="source video override; provide one per manifest, or use paths stored in manifests",
    )
    parser.add_argument("--output-dir", default="./contrastive_output")
    parser.add_argument(
        "--checkpoint",
        help="trained network used by predict, embed, or sample-crops mode",
    )
    parser.add_argument(
        "--init-checkpoint",
        help="network used to initialize training on the supplied data",
    )
    parser.add_argument("--num-individuals", type=int, help="number of identity clusters")
    parser.add_argument(
        "--export-crop-samples",
        nargs="?",
        const=100,
        type=int,
        metavar="COUNT",
        help=(
            "export evenly spaced, non-augmented network input crops; specifying "
            "the option without a count exports 100"
        ),
    )
    parser.add_argument(
        "--identity-assignment",
        choices=("idtracker", "kmeans"),
        default="idtracker",
        help=(
            "identity assignment strategy; idtracker aggregates detection votes and "
            "enforces that coexisting tracks have different identities"
        ),
    )
    parser.add_argument(
        "--global-fragment-min-frames",
        type=int,
        default=30,
        help=(
            "minimum detections required in every track of a complete coexisting "
            "track group before it can initialize and anchor identity clusters"
        ),
    )
    parser.add_argument(
        "--export-latents",
        metavar="CSV",
        help="also export per-detection and track-mean embedding CSV files",
    )
    parser.add_argument(
        "--latent-projection",
        choices=("umap", "pca", "none"),
        default="umap",
        help="low-dimensional coordinates included with exported latent vectors",
    )
    parser.add_argument(
        "--visualization-dimensions",
        type=int,
        default=2,
        help="number of UMAP or PCA visualization coordinates",
    )
    parser.add_argument(
        "--projection-fit-samples",
        type=int,
        default=50000,
        help="maximum embeddings used to fit the visualization projection",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="UMAP neighborhood size",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="minimum spacing in the UMAP projection",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help=(
            "total images in each training batch; must be divisible by "
            "--samples-per-track"
        ),
    )
    parser.add_argument(
        "--tracks-per-batch",
        type=int,
        help=(
            "advanced override for the number of tracks per batch; when supplied, "
            "the effective batch size is this value times --samples-per-track"
        ),
    )
    parser.add_argument("--samples-per-track", type=int, default=4)
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        help="automatic when omitted",
    )
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument(
        "--crop-storage",
        choices=("auto", "ram", "disk", "on-demand"),
        default="auto",
        help=(
            "training crop storage: auto uses RAM when enough is available and a "
            "disk memory map otherwise; on-demand retains the old video-seeking behavior"
        ),
    )
    parser.add_argument(
        "--cache-crops",
        type=int,
        default=2048,
        help="maximum resized crops retained in RAM; set to 0 to disable",
    )
    parser.add_argument(
        "--preload-crops",
        action="store_true",
        help=(
            "compatibility alias for --crop-storage ram; automatic RAM/disk crop "
            "materialization is now the default"
        ),
    )
    parser.add_argument(
        "--background-subtraction",
        nargs="?",
        const="median",
        choices=("none", "median"),
        help=(
            "mask each crop using a video-wide median background; specifying the "
            "option without a value selects median"
        ),
    )
    parser.add_argument(
        "--background-samples",
        type=int,
        help="frames sampled across each complete video for median estimation",
    )
    parser.add_argument(
        "--background-threshold",
        type=int,
        help="minimum per-channel pixel difference retained as foreground",
    )
    parser.add_argument(
        "--background-morph-kernel",
        type=int,
        help="foreground-mask cleanup kernel; 0 or 1 disables cleanup",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args, parser):
    if args.mode in {"predict", "embed"} and not args.checkpoint:
        parser.error(f"--checkpoint is required in {args.mode} mode")
    if args.mode in {"train-predict", "predict"} and args.num_individuals is None:
        parser.error(f"--num-individuals is required in {args.mode} mode")
    if args.checkpoint and args.mode in {"train", "train-predict"}:
        parser.error("Use --init-checkpoint, not --checkpoint, when training.")
    if args.samples_per_track < 2:
        parser.error("--samples-per-track must be at least 2 to form positive pairs")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.tracks_per_batch is None:
        if args.batch_size % args.samples_per_track:
            parser.error("--batch-size must be divisible by --samples-per-track")
        args.tracks_per_batch = args.batch_size // args.samples_per_track
    elif args.tracks_per_batch < 1:
        parser.error("--tracks-per-batch must be at least 1")
    if args.background_samples is not None and args.background_samples < 1:
        parser.error("--background-samples must be at least 1")
    if args.background_threshold is not None and not 0 <= args.background_threshold <= 255:
        parser.error("--background-threshold must be between 0 and 255")
    if args.background_morph_kernel is not None and args.background_morph_kernel < 0:
        parser.error("--background-morph-kernel cannot be negative")
    if args.visualization_dimensions < 1:
        parser.error("--visualization-dimensions must be at least 1")
    if args.projection_fit_samples < 3:
        parser.error("--projection-fit-samples must be at least 3")
    if args.umap_neighbors < 2:
        parser.error("--umap-neighbors must be at least 2")
    if not 0 <= args.umap_min_dist:
        parser.error("--umap-min-dist cannot be negative")
    if args.global_fragment_min_frames < 1:
        parser.error("--global-fragment-min-frames must be at least 1")
    if args.export_crop_samples is not None and args.export_crop_samples < 1:
        parser.error("--export-crop-samples must be at least 1")
    if args.mode == "sample-crops" and args.export_crop_samples is None:
        args.export_crop_samples = 100
    if args.preload_crops:
        if args.crop_storage != "auto":
            parser.error("Use either --preload-crops or --crop-storage, not both")
        args.crop_storage = "ram"


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    require_torch()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    preprocessing_checkpoint = (
        args.init_checkpoint
        if args.mode in {"train", "train-predict"}
        else args.checkpoint
    )
    resolve_preprocessing(args, preprocessing_checkpoint, device)
    background_store = None
    if args.background_subtraction == "median":
        background_store = MedianBackgroundStore(args.background_samples, args.seed)

    detections, raw_rows = read_manifests(args.manifest, args.video)
    metadata = build_track_metadata(detections)
    preloaded_crop_store = None

    if args.mode in {"train", "train-predict"}:
        model, input_size, _, preloaded_crop_store = train_model(
            args,
            detections,
            metadata,
            device,
            background_store,
            preloaded_crop_store,
        )
    else:
        model, _, _, input_size = make_model(args, device, args.checkpoint)
        # A single inference pass is already ordered by the manifest, so AUTO does
        # not spend time and storage materializing every crop. Explicit RAM or disk
        # modes remain available when repeated access is desired.
        if args.crop_storage in {"ram", "disk"}:
            preloaded_crop_store = make_preloaded_crop_store(
                args, detections, input_size, background_store
            )

    if args.export_crop_samples:
        export_crop_samples(
            args.output_dir,
            args.export_crop_samples,
            detections,
            metadata[1],
            input_size,
            background_store,
            args.background_threshold,
            args.background_morph_kernel,
            preloaded_crop_store,
        )

    if args.mode in {"train-predict", "predict", "embed"}:
        run_inference(
            args,
            model,
            input_size,
            detections,
            raw_rows,
            metadata,
            device,
            background_store,
            preloaded_crop_store,
        )


if __name__ == "__main__":
    main()
