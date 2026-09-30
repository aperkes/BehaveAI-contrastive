#!/usr/bin/env python3
"""Train and apply a track-supervised contrastive identity model.

The crop manifest produced by ``simple_track.py --export-crop-manifest`` is
used as weak supervision:

* detections in the same track are positive examples;
* tracks that coexist in at least one video frame are negative examples;
* non-overlapping tracks are not assumed to be different animals.

Pixels are read from the source video on demand. Resized crops are held in a
bounded RAM cache, so no directory of crop images is required.

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
import os
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

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
DEFAULT_INPUT_SIZE = 128


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

    for index, item in enumerate(detections):
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
    """Decode each video sequentially once and retain fixed-size uint8 crops."""

    def __init__(
        self,
        detections,
        input_size,
        background_store=None,
        background_threshold=25,
        background_morph_kernel=3,
    ):
        self.detections = detections
        self.input_size = input_size
        self.background_store = background_store
        self.background_threshold = background_threshold
        self.background_morph_kernel = background_morph_kernel
        self.crops = None

    def get(self, index):
        if self.crops is None:
            self._build()
        return self.crops[index]

    def _build(self):
        shape = (len(self.detections), self.input_size, self.input_size, 3)
        gib = np.prod(shape, dtype=np.int64) / 1024 ** 3
        print(f"Preloading {len(self.detections)} crops into {gib:.2f} GiB of RAM...")
        self.crops = np.empty(shape, dtype=np.uint8)

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
        print("Crop preload complete.")


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
    if args.preload_crops and preloaded_crop_store is None:
        preloaded_crop_store = PreloadedCropStore(
            detections,
            input_size,
            background_store,
            args.background_threshold,
            args.background_morph_kernel,
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
    batches_per_epoch = args.batches_per_epoch or max(
        1, int(np.ceil(len(detections) / (args.tracks_per_batch * args.samples_per_track)))
    )
    sampler = TrackBatchSampler(
        indices_by_track,
        known_different,
        args.tracks_per_batch,
        args.samples_per_track,
        batches_per_epoch,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    known_different_tensor = torch.from_numpy(known_different)

    if not known_different.any():
        print(
            "Warning: no temporally overlapping tracks were found. Training has positives "
            "but no reliable negative pairs, so identity separation will be weak."
        )

    print(
        f"Training on {len(detections)} detections from {len(track_keys)} tracks "
        f"using {device}."
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = np.zeros(3, dtype=float)
        for images, labels, _ in loader:
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

        means = totals / len(loader)
        print(
            f"Epoch {epoch:03d}/{args.epochs}: loss={means[0]:.5f} "
            f"positive={means[1]:.5f} negative={means[2]:.5f}"
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
    with torch.no_grad():
        for images, _, _ in loader:
            vectors = model(images.to(device), project=False)
            embeddings.append(vectors.cpu().numpy())
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


def cluster_identities(track_embeddings, number_of_individuals, seed):
    if number_of_individuals < 1:
        raise ValueError("--num-individuals must be at least 1.")
    if number_of_individuals > len(track_embeddings):
        raise ValueError(
            f"Requested {number_of_individuals} individuals but only "
            f"{len(track_embeddings)} tracks are available."
        )

    # The embeddings are unit normalized, so cosine k-means is a natural fit
    # and avoids adding a scikit-learn dependency to the BehaveAI environment.
    generator = np.random.default_rng(seed)
    best_assignments = None
    best_cost = np.inf
    for _ in range(20):
        initial = generator.choice(
            len(track_embeddings), size=number_of_individuals, replace=False
        )
        centers = track_embeddings[initial].copy()
        assignments = np.full(len(track_embeddings), -1, dtype=np.int64)

        for _ in range(200):
            similarities = track_embeddings @ centers.T
            updated = similarities.argmax(axis=1)
            if np.array_equal(updated, assignments):
                break
            assignments = updated

            for identity in range(number_of_individuals):
                members = track_embeddings[assignments == identity]
                if len(members):
                    center = members.mean(axis=0)
                else:
                    # Re-seed an empty cluster with the least well represented track.
                    center = track_embeddings[np.argmin(similarities.max(axis=1))]
                centers[identity] = center / max(np.linalg.norm(center), 1e-12)

        similarities = track_embeddings @ centers.T
        cost = float(np.sum(1.0 - similarities[np.arange(len(assignments)), assignments]))
        if cost < best_cost:
            best_cost = cost
            best_assignments = assignments.copy()

    return best_assignments


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
    track_keys, track_to_label, _, _ = metadata
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
        identity_by_track = cluster_identities(track_embeddings, args.num_individuals, args.seed)
        identity_path = output_dir / "identity_predictions.csv"
        write_identity_rows(
            identity_path, raw_rows, detections, track_to_label, identity_by_track
        )
        print(f"Saved identity predictions: {identity_path}")

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
        choices=("train-predict", "train", "predict", "embed"),
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
    parser.add_argument("--checkpoint", help="trained network used by predict or embed mode")
    parser.add_argument(
        "--init-checkpoint",
        help="network used to initialize training on the supplied data",
    )
    parser.add_argument("--num-individuals", type=int, help="number of identity clusters")
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
    parser.add_argument("--tracks-per-batch", type=int, default=8)
    parser.add_argument("--samples-per-track", type=int, default=4)
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        help="automatic when omitted",
    )
    parser.add_argument("--inference-batch-size", type=int, default=128)
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
            "decode videos sequentially once and retain every resized uint8 crop in RAM; "
            "much faster for training when sufficient memory is available"
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
    if args.tracks_per_batch < 1:
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
        if args.preload_crops:
            preloaded_crop_store = PreloadedCropStore(
                detections,
                input_size,
                background_store,
                args.background_threshold,
                args.background_morph_kernel,
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
