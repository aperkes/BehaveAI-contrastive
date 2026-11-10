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
    from torchvision.models.resnet import BasicBlock, ResNet
except ImportError as exc:  # Keep --help and syntax inspection useful without PyTorch.
    torch = None
    nn = None
    F = None
    BatchSampler = object
    DataLoader = None
    Dataset = object
    BasicBlock = None
    ResNet = object
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


DEFAULT_EMBEDDING_DIM = 128
DEFAULT_IDTRACKER_EMBEDDING_DIM = 8
DEFAULT_INPUT_SIZE = 96


@dataclass(frozen=True)
class Detection:
    video: str
    frame: int
    track_id: str
    bbox: tuple
    manifest_index: int
    oriented_bbox: tuple | None = None
    frame_step: int = 1

    @property
    def track_key(self):
        # Track IDs are local to a video, so the video must be part of the key.
        return self.video, self.track_id


@dataclass
class TrainingFragmentMetadata:
    """Conservative contiguous fragments used only as contrastive supervision."""

    keys: list
    indices_by_fragment: dict
    known_different: np.ndarray
    original_track_by_fragment: dict
    excluded_detection_count: int
    uncertainty_event_count: int


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


def read_manifests(manifest_paths, video_overrides=None, frame_step_overrides=None):
    """Read one or more crop manifests and return detections plus raw rows."""
    if video_overrides and len(video_overrides) != len(manifest_paths):
        raise ValueError("Provide one --video for each --manifest, or omit --video.")
    if frame_step_overrides and len(frame_step_overrides) not in {1, len(manifest_paths)}:
        raise ValueError(
            "Provide --frame-step once for all manifests or once per manifest."
        )

    detections = []
    raw_rows = []
    for manifest_number, manifest_path in enumerate(manifest_paths):
        override = None
        if video_overrides:
            override = os.path.abspath(video_overrides[manifest_number])
        frame_step_override = None
        if frame_step_overrides:
            frame_step_override = (
                frame_step_overrides[0]
                if len(frame_step_overrides) == 1
                else frame_step_overrides[manifest_number]
            )

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
                frame_step = int(
                    frame_step_override
                    if frame_step_override is not None
                    else row.get("frame_step") or 1
                )
                if frame_step < 1:
                    raise ValueError(
                        f"{manifest_path} row {reader.line_num} has invalid "
                        f"frame_step={frame_step}"
                    )
                bbox = tuple(int(float(row[name])) for name in ("x1", "y1", "x2", "y2"))
                oriented_fields = (
                    "obb_cx",
                    "obb_cy",
                    "obb_width",
                    "obb_height",
                    "obb_angle_radians",
                )
                oriented_values = [row.get(name, "").strip() for name in oriented_fields]
                if any(oriented_values) and not all(oriented_values):
                    raise ValueError(
                        f"{manifest_path} row {reader.line_num} has incomplete "
                        "oriented-box coordinates."
                    )
                oriented_bbox = (
                    tuple(float(value) for value in oriented_values)
                    if all(oriented_values)
                    else None
                )
                detection = Detection(
                    video=video,
                    frame=int(row["frame"]),
                    track_id=str(row["track_id"]),
                    bbox=bbox,
                    manifest_index=len(raw_rows),
                    oriented_bbox=oriented_bbox,
                    frame_step=frame_step,
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


def build_conservative_training_fragments(
    detections,
    close_distance=0.75,
    assignment_margin=0.35,
    motion_jump=1.5,
    uncertainty_buffer=5,
    interpolation_gap=10,
    min_fragment_length=4,
):
    """Split tracks around locally ambiguous encounters without assigning identity.

    Linear interpolation is used only to expose close encounters during short
    detection gaps. Interpolated locations never become crops and never bridge
    positive fragments. Continuity and buffer sizes use each video's processed
    frame step, so regularly skipped source frames remain contiguous.
    """
    observations_by_track = defaultdict(list)
    states_by_video_frame = defaultdict(list)
    frame_step_by_video = {}
    for index, item in enumerate(detections):
        previous_step = frame_step_by_video.setdefault(item.video, item.frame_step)
        if previous_step != item.frame_step:
            raise ValueError(
                f"Inconsistent frame_step values for {item.video}: "
                f"{previous_step} and {item.frame_step}"
            )
        x1, y1, x2, y2 = item.bbox
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        scale = max(x2 - x1, y2 - y1, 1)
        state = {
            "track": item.track_key,
            "center": center,
            "scale": float(scale),
            "bbox": item.bbox,
            "observed": True,
        }
        observations_by_track[item.track_key].append((item.frame, index, state))
        states_by_video_frame[(item.video, item.frame)].append(state)

    # Interpolate only bounded internal gaps. This helps flag a visible fish
    # approaching a temporarily missed track, but does not create supervision.
    for track, observations in observations_by_track.items():
        observations.sort()
        video = track[0]
        frame_step = frame_step_by_video[video]
        for (first_frame, _, first), (last_frame, _, last) in zip(
            observations, observations[1:]
        ):
            raw_gap = last_frame - first_frame
            if raw_gap % frame_step:
                continue
            gap_steps = raw_gap // frame_step
            if not 1 < gap_steps <= interpolation_gap + 1:
                continue
            for frame in range(first_frame + frame_step, last_frame, frame_step):
                fraction = (frame - first_frame) / raw_gap
                center = (
                    first["center"][0]
                    + fraction * (last["center"][0] - first["center"][0]),
                    first["center"][1]
                    + fraction * (last["center"][1] - first["center"][1]),
                )
                scale = first["scale"] + fraction * (
                    last["scale"] - first["scale"]
                )
                states_by_video_frame[(video, frame)].append(
                    {
                        "track": track,
                        "center": center,
                        "scale": scale,
                        "bbox": None,
                        "observed": False,
                    }
                )

    uncertain = defaultdict(set)

    def mark(state, frame):
        uncertain[state["track"]].add(frame)

    def distance(first, second):
        return float(np.hypot(
            first["center"][0] - second["center"][0],
            first["center"][1] - second["center"][1],
        ))

    # Very close detections are locally ambiguous regardless of how many other
    # animals happen to be visible in the frame.
    for (_, frame), states in states_by_video_frame.items():
        for position, first in enumerate(states):
            for second in states[position + 1:]:
                if first["track"] == second["track"]:
                    continue
                scale = max((first["scale"] + second["scale"]) / 2.0, 1.0)
                if distance(first, second) / scale < close_distance:
                    mark(first, frame)
                    mark(second, frame)

    # Compare the recorded continuation with the alternative swapped matching.
    # A small cost advantage means the motion-only association was not decisive.
    frames_by_video = defaultdict(list)
    for video, frame in states_by_video_frame:
        frames_by_video[video].append(frame)
    for video, frames in frames_by_video.items():
        frame_set = set(frames)
        frame_step = frame_step_by_video[video]
        for frame in sorted(frame_set):
            if frame - frame_step not in frame_set:
                continue
            previous = {
                state["track"]: state
                for state in states_by_video_frame[(video, frame - frame_step)]
            }
            current = {
                state["track"]: state
                for state in states_by_video_frame[(video, frame)]
            }
            common = sorted(set(previous).intersection(current))
            for position, first_track in enumerate(common):
                for second_track in common[position + 1:]:
                    first_previous, first_current = (
                        previous[first_track], current[first_track]
                    )
                    second_previous, second_current = (
                        previous[second_track], current[second_track]
                    )
                    scale = max(
                        np.mean(
                            [
                                first_previous["scale"],
                                first_current["scale"],
                                second_previous["scale"],
                                second_current["scale"],
                            ]
                        ),
                        1.0,
                    )
                    proximity = min(
                        distance(first_current, second_current),
                        distance(first_previous, second_previous),
                    ) / scale
                    if proximity > 2.0:
                        continue
                    direct = distance(first_previous, first_current) + distance(
                        second_previous, second_current
                    )
                    swapped = distance(first_previous, second_current) + distance(
                        second_previous, first_current
                    )
                    if swapped - direct < assignment_margin * scale:
                        mark(first_current, frame)
                        mark(second_current, frame)

    # Abrupt deviations from constant velocity are another indication that an
    # online track may have transferred to a neighboring animal.
    for track, observations in observations_by_track.items():
        frame_step = frame_step_by_video[track[0]]
        for first, second, third in zip(
            observations, observations[1:], observations[2:]
        ):
            first_frame, _, first_state = first
            second_frame, _, second_state = second
            third_frame, _, third_state = third
            if (
                second_frame != first_frame + frame_step
                or third_frame != second_frame + frame_step
            ):
                continue
            predicted = (
                2 * second_state["center"][0] - first_state["center"][0],
                2 * second_state["center"][1] - first_state["center"][1],
            )
            error = np.hypot(
                third_state["center"][0] - predicted[0],
                third_state["center"][1] - predicted[1],
            )
            if error / max(third_state["scale"], 1.0) > motion_jump:
                uncertain[track].add(third_frame)

    expanded_uncertain = defaultdict(set)
    for track, frames in uncertain.items():
        frame_step = frame_step_by_video[track[0]]
        for frame in frames:
            expanded_uncertain[track].update(
                frame + offset * frame_step
                for offset in range(-uncertainty_buffer, uncertainty_buffer + 1)
            )

    fragment_keys = []
    indices_by_fragment = {}
    original_track_by_fragment = {}
    detection_to_fragment = {}
    excluded = 0
    for track, observations in sorted(observations_by_track.items()):
        frame_step = frame_step_by_video[track[0]]
        current = []
        previous_frame = None

        def finish_fragment():
            nonlocal current, excluded
            if not current:
                return
            if len(current) < min_fragment_length:
                excluded += len(current)
                current = []
                return
            fragment = len(fragment_keys)
            key = (track[0], track[1], fragment)
            fragment_keys.append(key)
            indices_by_fragment[fragment] = current
            original_track_by_fragment[fragment] = track
            for detection_index in current:
                detection_to_fragment[detection_index] = fragment
            current = []

        for frame, index, _ in observations:
            unsafe = frame in expanded_uncertain[track]
            # Positive fragments contain consecutive observed detections only;
            # interpolation is deliberately never used to join them.
            if unsafe or (
                previous_frame is not None
                and frame != previous_frame + frame_step
            ):
                finish_fragment()
            if unsafe:
                excluded += 1
            else:
                current.append(index)
            previous_frame = frame
        finish_fragment()

    fragments_by_frame = defaultdict(set)
    for index, item in enumerate(detections):
        fragment = detection_to_fragment.get(index)
        if fragment is not None:
            fragments_by_frame[(item.video, item.frame)].add(fragment)
    known_different = np.zeros((len(fragment_keys), len(fragment_keys)), dtype=bool)
    for fragments in fragments_by_frame.values():
        fragments = sorted(fragments)
        for position, first in enumerate(fragments):
            for second in fragments[position + 1:]:
                known_different[first, second] = True
                known_different[second, first] = True

    return TrainingFragmentMetadata(
        keys=fragment_keys,
        indices_by_fragment=indices_by_fragment,
        known_different=known_different,
        original_track_by_fragment=original_track_by_fragment,
        excluded_detection_count=excluded,
        uncertainty_event_count=sum(len(frames) for frames in uncertain.values()),
    )


def use_original_tracks_as_training_fragments(metadata):
    track_keys, _, indices_by_track, known_different = metadata
    return TrainingFragmentMetadata(
        keys=[(video, track_id, label) for label, (video, track_id) in enumerate(track_keys)],
        indices_by_fragment=dict(indices_by_track),
        known_different=known_different,
        original_track_by_fragment={
            label: track for label, track in enumerate(track_keys)
        },
        excluded_detection_count=0,
        uncertainty_event_count=0,
    )


def write_training_fragment_summary(path, fragments, detections):
    """Record exactly which contiguous fragments are eligible for supervision."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "training_fragment",
        "source_video",
        "source_track_id",
        "first_frame",
        "last_frame",
        "detection_count",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fragment, indices in fragments.indices_by_fragment.items():
            items = [detections[index] for index in indices]
            video, track_id = fragments.original_track_by_fragment[fragment]
            writer.writerow(
                {
                    "training_fragment": fragment,
                    "source_video": video,
                    "source_track_id": track_id,
                    "first_frame": items[0].frame,
                    "last_frame": items[-1].frame,
                    "detection_count": len(items),
                }
            )
    return path


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


def extract_oriented_crop(image, oriented_bbox):
    """Rotate an OBB into a tight crop, with its long axis horizontal."""
    cx, cy, box_width, box_height, angle_radians = oriented_bbox
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"Invalid oriented crop dimensions: {oriented_bbox}")
    rotation = cv2.getRotationMatrix2D(
        (float(cx), float(cy)), np.degrees(angle_radians), 1.0
    )
    rotated = cv2.warpAffine(
        image,
        rotation,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    crop = cv2.getRectSubPix(
        rotated,
        (max(1, int(round(box_width))), max(1, int(round(box_height)))),
        (float(cx), float(cy)),
    )
    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


def prepare_detection_crop(
    frame,
    item,
    input_size,
    background=None,
    background_threshold=25,
    background_morph_kernel=3,
    input_channels=3,
):
    """Extract, optionally mask, and resize one detection crop."""
    if background is not None:
        if background.shape != frame.shape:
            raise ValueError(
                f"Background shape {background.shape} does not match frame shape "
                f"{frame.shape} for {item.video}"
            )
    if item.oriented_bbox is not None:
        crop = extract_oriented_crop(frame, item.oriented_bbox)
        background_crop = (
            extract_oriented_crop(background, item.oriented_bbox)
            if background is not None
            else None
        )
    else:
        x1, y1, x2, y2 = item.bbox
        height, width = frame.shape[:2]
        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid crop at frame {item.frame}, track {item.track_id}: {item.bbox}"
            )
        crop = frame[y1:y2, x1:x2]
        background_crop = background[y1:y2, x1:x2] if background is not None else None

    if background_crop is not None:
        crop = mask_to_moving_foreground(
            crop,
            background_crop,
            background_threshold,
            background_morph_kernel,
        )
    crop = letterbox_crop(crop, input_size)
    if input_channels == 1:
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)[..., None]
    return crop


class PreloadedCropStore:
    """Decode videos once into a RAM array or disk-backed memory map."""

    def __init__(
        self,
        detections,
        input_size,
        background_store=None,
        background_threshold=25,
        background_morph_kernel=3,
        input_channels=3,
        storage="auto",
        cache_path=None,
        detection_indices=None,
    ):
        self.detections = detections
        self.detection_indices = np.asarray(
            sorted(
                set(
                    range(len(detections))
                    if detection_indices is None
                    else detection_indices
                )
            ),
            dtype=np.int64,
        )
        if not len(self.detection_indices):
            raise ValueError("Cannot create an empty preloaded crop store.")
        if self.detection_indices[0] < 0 or self.detection_indices[-1] >= len(detections):
            raise IndexError("Preloaded crop detection index is out of range.")
        self.storage_index = np.full(len(detections), -1, dtype=np.int64)
        self.storage_index[self.detection_indices] = np.arange(
            len(self.detection_indices), dtype=np.int64
        )
        self.input_size = input_size
        self.background_store = background_store
        self.background_threshold = background_threshold
        self.background_morph_kernel = background_morph_kernel
        self.input_channels = input_channels
        self.shape = (
            len(self.detection_indices),
            self.input_size,
            self.input_size,
            self.input_channels,
        )
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
        storage_index = int(self.storage_index[index])
        if storage_index < 0:
            raise KeyError(index)
        if self.crops is None:
            self._build()
        return self.crops[storage_index]

    def has(self, index):
        return self.storage_index[index] >= 0

    def release(self):
        if isinstance(self.crops, np.memmap):
            self.crops.flush()
            mmap = getattr(self.crops, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self.crops = None

    def _build(self):
        gib = self.required_bytes / 1024 ** 3
        if self.storage == "ram":
            print(
                f"Preloading {len(self.detection_indices)} selected crops into "
                f"{gib:.2f} GiB of RAM..."
            )
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
                f"Saving {len(self.detection_indices)} selected crops to a {gib:.2f} GiB "
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
        for storage_index, detection_index in enumerate(self.detection_indices):
            item = self.detections[int(detection_index)]
            detections_by_video_and_frame[item.video][item.frame].append(
                (storage_index, item)
            )

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
                        self.input_channels,
                    )
                    completed += 1
                percent = 100 * completed / len(self.detection_indices)
                if percent >= next_report:
                    print(f"  crop preload: {percent:.1f}%", flush=True)
                    next_report += 5
            capture.release()
        if isinstance(self.crops, np.memmap):
            self.crops.flush()
        print("Crop preload complete.")


def make_preloaded_crop_store(
    args,
    detections,
    input_size,
    background_store,
    input_channels=3,
    detection_indices=None,
):
    return PreloadedCropStore(
        detections,
        input_size,
        background_store,
        args.background_threshold,
        args.background_morph_kernel,
        input_channels,
        storage=args.crop_storage,
        cache_path=Path(args.output_dir) / "crop_cache" / "crops.npy",
        detection_indices=detection_indices,
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
        input_channels=3,
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
        self.input_channels = input_channels
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
        if (
            self.preloaded_crop_store is not None
            and self.preloaded_crop_store.has(index)
        ):
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
            target_position = item.frame - 1
            current_position = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
            if current_position != target_position:
                capture.set(cv2.CAP_PROP_POS_FRAMES, target_position)
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
            self.input_channels,
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
            # Reverse both spatial axes together: this is a 180-degree rotation,
            # not a horizontal mirror that would alter the fish's appearance.
            tensor = torch.flip(tensor, dims=(1, 2))
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
        self.architecture = "small-cnn"
        self.input_channels = 3
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


class IdtrackerResNet18(ResNet):
    """idtracker.ai's ResNet18 identity network with grayscale or RGB input."""

    def __init__(
        self, embedding_dim=DEFAULT_IDTRACKER_EMBEDDING_DIM, input_channels=1
    ):
        if nn is None:
            require_torch()
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=embedding_dim)
        self.architecture = "idtracker-resnet18"
        self.input_channels = input_channels
        if input_channels != 3:
            self.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            nn.init.kaiming_normal_(
                self.conv1.weight, mode="fan_out", nonlinearity="relu"
            )
        self.fc = nn.Linear(512, embedding_dim, bias=False)

    def forward(self, images, project=True):
        # idtracker.ai trains and clusters the direct, unnormalised network output.
        return super().forward(images)


class FragmentPairDataset(Dataset):
    """Draw fresh crop pairs from explicit negative and positive fragment pairs."""

    def __init__(self, crop_dataset, indices_by_track, negative_pairs, positive_tracks):
        self.crop_dataset = crop_dataset
        self.indices_by_track = indices_by_track
        self.pairs = list(negative_pairs) + [
            (track, track) for track in positive_tracks
        ]
        self.negative_pair_count = len(negative_pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, pair_index):
        first_track, second_track = self.pairs[pair_index]
        first_index = random.choice(self.indices_by_track[first_track])
        second_index = random.choice(self.indices_by_track[second_track])
        if first_track == second_track and len(self.indices_by_track[first_track]) > 1:
            while second_index == first_index:
                second_index = random.choice(self.indices_by_track[second_track])
        first = self.crop_dataset[first_index][0]
        second = self.crop_dataset[second_index][0]
        return first, second, pair_index


class AdaptivePairBatchSampler(BatchSampler):
    """idtracker-style balanced pair batches with adaptive hard-pair sampling."""

    def __init__(
        self,
        negative_pairs,
        positive_tracks,
        indices_by_track,
        pairs_per_class,
        batches_per_epoch,
        score_decay=0.98,
    ):
        self.negative_count = len(negative_pairs)
        self.positive_count = len(positive_tracks)
        self.pairs_per_class = pairs_per_class
        self.batches_per_epoch = batches_per_epoch
        self.score_decay = score_decay
        negative_sizes = [
            len(indices_by_track[a]) + len(indices_by_track[b])
            for a, b in negative_pairs
        ]
        positive_sizes = [len(indices_by_track[track]) for track in positive_tracks]
        self.size_weights = torch.tensor(
            negative_sizes + positive_sizes, dtype=torch.float64
        )
        self.scores = torch.full((len(self.size_weights),), 10.0, dtype=torch.float64)

    def __len__(self):
        return self.batches_per_epoch

    @staticmethod
    def _normalise(values):
        return values / values.sum().clamp_min(1e-12)

    def _probabilities(self, start, stop):
        sizes = self._normalise(self.size_weights[start:stop])
        scores = self._normalise(self.scores[start:stop])
        return self._normalise(sizes + scores)

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            negative = torch.multinomial(
                self._probabilities(0, self.negative_count),
                self.pairs_per_class,
                replacement=True,
            )
            positive = torch.multinomial(
                self._probabilities(self.negative_count, len(self.scores)),
                self.pairs_per_class,
                replacement=True,
            ) + self.negative_count
            # Keeping negatives first matches the loss convention in idtracker.ai.
            yield torch.cat((negative, positive)).tolist()

    def update(self, pair_indices, losses):
        active = losses > 0
        if np.any(active):
            counts = np.bincount(
                pair_indices[active], minlength=len(self.scores)
            ).astype(np.float64)
            self.scores += torch.from_numpy(counts)
        self.scores.mul_(self.score_decay)


def idtracker_pair_losses(first, second, negative_count, margin):
    """Per-pair Euclidean losses matching idtracker.ai's contrastive criterion."""
    distances = F.pairwise_distance(first, second)
    negative = F.relu(margin - distances[:negative_count]).square()
    positive = F.relu(distances[negative_count:] - 1.0).square()
    return torch.cat((negative, positive)), negative, positive


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
    # Checkpoints predating the architecture field contain the original RGB CNN.
    architecture = stored.get(
        "architecture", "small-cnn" if checkpoint else args.architecture
    )
    default_embedding_dim = (
        DEFAULT_IDTRACKER_EMBEDDING_DIM
        if architecture == "idtracker-resnet18"
        else DEFAULT_EMBEDDING_DIM
    )
    embedding_dim = stored.get("embedding_dim", args.embedding_dim or default_embedding_dim)
    projection_dim = stored.get("projection_dim", args.projection_dim)
    input_size = stored.get("input_size", args.input_size or DEFAULT_INPUT_SIZE)
    if architecture == "idtracker-resnet18":
        requested_color = getattr(args, "input_color", "grayscale")
        input_channels = stored.get(
            "input_channels",
            1 if checkpoint else (1 if requested_color == "grayscale" else 3),
        )
        model = IdtrackerResNet18(embedding_dim, input_channels).to(device)
    else:
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
    training_fragments=None,
):
    track_keys, track_to_label, indices_by_track, known_different = metadata
    model, embedding_dim, projection_dim, input_size = make_model(
        args, device, args.init_checkpoint
    )
    preload_indices = None
    if model.architecture == "idtracker-resnet18" and training_fragments is not None:
        preload_indices = [
            index
            for indices in training_fragments.indices_by_fragment.values()
            for index in indices
        ]
    if args.crop_storage != "on-demand" and preloaded_crop_store is None:
        preloaded_crop_store = make_preloaded_crop_store(
            args,
            detections,
            input_size,
            background_store,
            model.input_channels,
            preload_indices,
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
        input_channels=model.input_channels,
    )

    if preloaded_crop_store is not None:
        print(
            "Crop loading: sequentially materialize selected training crops in "
            f"{preloaded_crop_store.storage.upper()} before the first training batch.",
            flush=True,
        )
    else:
        print(
            f"Crop loading: on demand from video with a {args.cache_crops}-crop RAM "
            "cache. Random video seeks can be slow.",
            flush=True,
        )

    if model.architecture == "idtracker-resnet18":
        fragment_indices = (
            training_fragments.indices_by_fragment
            if training_fragments is not None
            else indices_by_track
        )
        fragment_differences = (
            training_fragments.known_different
            if training_fragments is not None
            else known_different
        )
        best_state = _train_idtracker_style(
            args, model, dataset, fragment_indices, fragment_differences, device
        )
        if best_state is not None:
            model.load_state_dict(best_state)
    else:
        _train_legacy_style(
            args, model, dataset, indices_by_track, known_different, detections, device
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "contrastive_identity.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "architecture": model.architecture,
                "input_channels": model.input_channels,
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
            "training_fragment_config": {
                "strategy": args.training_fragments,
                "close_distance": args.fragment_close_distance,
                "assignment_margin": args.fragment_assignment_margin,
                "motion_jump": args.fragment_motion_jump,
                "uncertainty_buffer": args.fragment_uncertainty_buffer,
                "interpolation_gap": args.fragment_interpolation_gap,
                "minimum_length": args.min_fragment_length,
            },
            "epochs": args.epochs,
        },
        checkpoint_path,
    )
    dataset.close()
    print(f"Saved network: {checkpoint_path}")
    return model, input_size, checkpoint_path, preloaded_crop_store


def _train_legacy_style(
    args, model, dataset, indices_by_track, known_different, detections, device
):
    """Retain the original training loop for old checkpoints and comparisons."""
    effective_tracks = min(args.tracks_per_batch, len(indices_by_track))
    effective_batch_size = effective_tracks * args.samples_per_track
    batches_per_epoch = args.batches_per_epoch or max(
        1, int(np.ceil(len(detections) / effective_batch_size))
    )
    sampler = TrackBatchSampler(
        indices_by_track,
        known_different,
        effective_tracks,
        args.samples_per_track,
        batches_per_epoch,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    known_different_tensor = torch.from_numpy(known_different)
    print(
        f"Legacy small-CNN training on {len(detections)} detections using {device}; "
        f"effective batch {effective_batch_size} images."
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = np.zeros(3, dtype=float)
        started = perf_counter()
        last_report = started
        for batch_number, (images, labels, _) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, positive, negative = track_contrastive_loss(
                model(images), labels, known_different_tensor,
                0.5 if args.margin is None else args.margin,
            )
            loss.backward()
            optimizer.step()
            totals += (loss.item(), positive.item(), negative.item())
            now = perf_counter()
            if batch_number == 1 or batch_number == len(loader) or now - last_report >= 10:
                report_progress(f"epoch {epoch:03d} batches", batch_number, len(loader), started)
                last_report = now
        means = totals / len(loader)
        print(
            f"Epoch {epoch:03d}/{args.epochs}: loss={means[0]:.5f} "
            f"positive_loss={means[1]:.5f} negative_loss={means[2]:.5f}"
        )


def _train_idtracker_style(
    args, model, dataset, indices_by_track, known_different, device
):
    """Train with idtracker.ai's fragment pairs, loss, and adaptive sampling."""
    eligible = sorted(
        track for track, indices in indices_by_track.items()
        if len(indices) >= args.min_fragment_length
    )
    eligible_set = set(eligible)
    negative_pairs = [
        (first, second)
        for first in eligible
        for second in np.flatnonzero(known_different[first])
        if first < second and int(second) in eligible_set
    ]
    if not eligible:
        raise ValueError(
            "No fragments meet --min-fragment-length; lower it or check the manifest."
        )
    if not negative_pairs:
        raise ValueError(
            "No coexisting fragment pairs meet --min-fragment-length. The idTracker "
            "training loss requires reliable negative pairs."
        )
    pair_dataset = FragmentPairDataset(
        dataset, indices_by_track, negative_pairs, eligible
    )
    training_detection_count = sum(
        len(indices_by_track[track]) for track in eligible
    )
    # A batch contains this many negative pairs plus the same number of positives.
    # Each pair contains two images, exactly as in idtracker.ai.
    batches_per_epoch = args.batches_per_epoch or max(
        1, int(np.ceil(training_detection_count / (4 * args.pairs_per_batch)))
    )
    sampler = AdaptivePairBatchSampler(
        negative_pairs,
        eligible,
        indices_by_track,
        args.pairs_per_batch,
        batches_per_epoch,
    )
    loader = DataLoader(pair_dataset, batch_sampler=sampler, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    input_color = "grayscale" if model.input_channels == 1 else "RGB"
    print(
        f"idTracker-style {input_color} training on {training_detection_count} detections "
        f"from {len(eligible)} "
        f"eligible fragments ({len(negative_pairs)} negative and {len(eligible)} "
        f"positive fragment pairs) using {device}. Each optimizer step samples "
        f"{args.pairs_per_batch} pairs per class / {4 * args.pairs_per_batch} images."
    )
    print(
        f"Forward passes are chunked to {args.forward_pair_chunk_size} pairs to bound VRAM."
    )
    validation_interval = args.validation_interval or (
        max(5 * args.num_individuals, 100) if args.num_individuals else None
    )
    best_score = -np.inf
    best_state = None
    validations = stale_validations = total_batches = 0
    stop_training = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = np.zeros(3, dtype=float)
        completed_batches = 0
        started = perf_counter()
        last_report = started
        for batch_number, (first, second, pair_indices) in enumerate(loader, start=1):
            completed_batches = batch_number
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            negative_sum = positive_sum = 0.0
            pair_count = len(pair_indices)
            negative_count = pair_count // 2
            chunk = args.forward_pair_chunk_size
            for start in range(0, pair_count, chunk):
                stop = min(start + chunk, pair_count)
                images = torch.cat((first[start:stop], second[start:stop]), dim=0).to(device)
                vectors = model(images)
                chunk_size = stop - start
                first_vectors, second_vectors = vectors[:chunk_size], vectors[chunk_size:]
                chunk_negative = max(0, min(stop, negative_count) - start)
                losses, negative, positive = idtracker_pair_losses(
                    first_vectors, second_vectors, chunk_negative,
                    10.0 if args.margin is None else args.margin,
                )
                (losses.sum() / pair_count).backward()
                batch_losses.append(losses.detach().cpu().numpy())
                negative_sum += float(negative.sum().detach())
                positive_sum += float(positive.sum().detach())
            optimizer.step()
            total_batches += 1
            loss_values = np.concatenate(batch_losses)
            sampler.update(pair_indices.numpy(), loss_values)
            totals += (
                float(loss_values.mean()),
                positive_sum / args.pairs_per_batch,
                negative_sum / args.pairs_per_batch,
            )
            now = perf_counter()
            if batch_number == 1 or batch_number == len(loader) or now - last_report >= 10:
                report_progress(f"epoch {epoch:03d} batches", batch_number, len(loader), started)
                last_report = now
            if (
                validation_interval
                and total_batches >= args.validation_warmup * validation_interval
                and total_batches % validation_interval == 0
            ):
                score = _embedding_silhouette_score(
                    args, model, dataset, eligible, indices_by_track, device
                )
                validations += 1
                improved = score > best_score
                if improved:
                    best_score = score
                    stale_validations = 0
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                else:
                    stale_validations += 1
                print(
                    f"  validation {validations}: silhouette={score:.4f}, "
                    f"best={best_score:.4f}, stale={stale_validations}",
                    flush=True,
                )
                if (
                    stale_validations > args.validation_patience
                    or (best_score > args.silhouette_target and stale_validations > 1)
                ):
                    reason = (
                        "target reached and stable"
                        if best_score > args.silhouette_target
                        else "patience exhausted"
                    )
                    print(f"Stopping contrastive training: {reason}.")
                    stop_training = True
                    break
        means = totals / completed_batches
        print(
            f"Epoch {epoch:03d}/{args.epochs}: loss={means[0]:.5f} "
            f"positive_loss={means[1]:.5f} negative_loss={means[2]:.5f}"
        )
        if stop_training:
            break
    return best_state


def _embedding_silhouette_score(
    args, model, dataset, eligible_tracks, indices_by_track, device
):
    """Cluster held-out sampled crops and compute an idTracker-style silhouette."""
    limit = args.validation_images_per_individual * args.num_individuals
    candidates = np.concatenate(
        [np.asarray(indices_by_track[track], dtype=np.int64) for track in eligible_tracks]
    )
    generator = np.random.default_rng(args.seed + len(candidates))
    if len(candidates) > limit:
        candidates = generator.choice(candidates, limit, replace=False)
    previous_augment = dataset.augment
    dataset.augment = False
    vectors = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(candidates), args.inference_batch_size):
            images = torch.stack(
                [dataset[int(index)][0] for index in candidates[start:start + args.inference_batch_size]]
            ).to(device)
            vectors.append(model(images).cpu().numpy())
    dataset.augment = previous_augment
    model.train()
    vectors = np.concatenate(vectors)
    assignments, _, _ = euclidean_kmeans(
        vectors, args.num_individuals, args.seed, restarts=5
    )
    query_indices = np.arange(len(vectors))
    if len(query_indices) > args.silhouette_samples:
        query_indices = generator.choice(
            query_indices, args.silhouette_samples, replace=False
        )
    scores = []
    for query_start in range(0, len(query_indices), 128):
        selected = query_indices[query_start:query_start + 128]
        distances = np.sqrt(
            np.maximum(
                ((vectors[selected, None] - vectors[None]) ** 2).sum(axis=2), 0
            )
        )
        for row, index in enumerate(selected):
            own = assignments == assignments[index]
            own_count = int(own.sum())
            if own_count <= 1:
                continue
            a = float(distances[row, own].sum() / (own_count - 1))
            b = min(
                float(distances[row, assignments == identity].mean())
                for identity in range(args.num_individuals)
                if identity != assignments[index]
                and np.any(assignments == identity)
            )
            scores.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(scores)) if scores else -1.0


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
        input_channels=model.input_channels,
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
        selected = len(preloaded_crop_store.detection_indices)
        if selected == len(detections):
            description = f"complete preloaded {preloaded_crop_store.storage.upper()} store"
        else:
            description = (
                f"{selected} preloaded training crops plus on-demand decoding "
                f"for {len(detections) - selected} remaining detections"
            )
        print(f"Embedding crop loading: {description}.", flush=True)
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
    input_channels=3,
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
        input_channels=input_channels,
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
            output_crop = (
                crop[..., 0]
                if crop.shape[-1] == 1
                else cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            )
            if not cv2.imwrite(str(sample_dir / filename), output_crop):
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
                    "obb_cx": item.oriented_bbox[0] if item.oriented_bbox else "",
                    "obb_cy": item.oriented_bbox[1] if item.oriented_bbox else "",
                    "obb_width": item.oriented_bbox[2] if item.oriented_bbox else "",
                    "obb_height": item.oriented_bbox[3] if item.oriented_bbox else "",
                    "obb_angle_radians": (
                        item.oriented_bbox[4] if item.oriented_bbox else ""
                    ),
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


def export_pair_samples(
    output_dir,
    sample_count,
    detections,
    metadata,
    input_size,
    background_store,
    background_threshold,
    background_morph_kernel,
    preloaded_crop_store=None,
    input_channels=1,
    min_fragment_length=4,
    seed=42,
    training_fragments=None,
):
    """Export side-by-side examples of the fragment pairs used for training."""
    _, track_to_label, original_indices_by_track, original_known_different = metadata
    if training_fragments is None:
        indices_by_track = original_indices_by_track
        known_different = original_known_different
    else:
        indices_by_track = training_fragments.indices_by_fragment
        known_different = training_fragments.known_different
    eligible = sorted(
        track
        for track, indices in indices_by_track.items()
        if len(indices) >= min_fragment_length
    )
    eligible_set = set(eligible)
    negative_pairs = [
        (first, int(second))
        for first in eligible
        for second in np.flatnonzero(known_different[first])
        if first < second and int(second) in eligible_set
    ]
    if not eligible or not negative_pairs:
        raise ValueError(
            "Pair export requires eligible fragments and at least one coexisting "
            "negative-fragment pair."
        )

    generator = np.random.default_rng(seed)

    def choose_examples(values, weights):
        count = min(sample_count, len(values))
        probabilities = np.asarray(weights, dtype=np.float64)
        probabilities /= probabilities.sum()
        selected = generator.choice(
            len(values), size=count, replace=False, p=probabilities
        )
        return [values[int(index)] for index in selected]

    selected_positive = choose_examples(
        [(track, track) for track in eligible],
        [len(indices_by_track[track]) for track in eligible],
    )
    selected_negative = choose_examples(
        negative_pairs,
        [
            len(indices_by_track[first]) + len(indices_by_track[second])
            for first, second in negative_pairs
        ],
    )

    pair_dir = Path(output_dir) / "pair_samples"
    positive_dir = pair_dir / "positive"
    negative_dir = pair_dir / "negative"
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)
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
        input_channels=input_channels,
    )
    frame_sets = {}

    def frames_for_track(track):
        if track not in frame_sets:
            frame_sets[track] = {
                detections[index].frame for index in indices_by_track[track]
            }
        return frame_sets[track]

    def choose_detection(track, excluded=None):
        candidates = indices_by_track[track]
        index = int(generator.choice(candidates))
        if excluded is not None and len(candidates) > 1:
            while index == excluded:
                index = int(generator.choice(candidates))
        return index

    def display_crop(crop):
        if crop.shape[-1] == 1:
            return cv2.cvtColor(crop[..., 0], cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

    rows = []
    all_examples = (
        [("positive", pair) for pair in selected_positive]
        + [("negative", pair) for pair in selected_negative]
    )
    try:
        for completed, (pair_type, (first_track, second_track)) in enumerate(
            all_examples, start=1
        ):
            first_index = choose_detection(first_track)
            second_index = choose_detection(
                second_track, first_index if first_track == second_track else None
            )
            first_item = detections[first_index]
            second_item = detections[second_index]
            first_crop = display_crop(dataset._read_crop(first_index))
            second_crop = display_crop(dataset._read_crop(second_index))
            gap = 8
            header = 38
            canvas = np.zeros(
                (input_size + header, 2 * input_size + gap, 3), dtype=np.uint8
            )
            canvas[header:, :input_size] = first_crop
            canvas[header:, input_size + gap:] = second_crop
            color = (80, 220, 80) if pair_type == "positive" else (80, 80, 240)
            cv2.putText(
                canvas,
                f"{pair_type.upper()}  A:{first_item.track_id} f{first_item.frame}",
                (3, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"B:{second_item.track_id} f{second_item.frame}",
                (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                color,
                1,
                cv2.LINE_AA,
            )
            type_number = sum(row["pair_type"] == pair_type for row in rows) + 1
            relative_path = Path(pair_type) / f"pair_{type_number:03d}.png"
            if not cv2.imwrite(str(pair_dir / relative_path), canvas):
                raise RuntimeError(f"Could not write pair sample: {relative_path}")
            shared_frames = len(
                frames_for_track(first_track).intersection(
                    frames_for_track(second_track)
                )
            )
            rows.append(
                {
                    "pair_file": str(relative_path),
                    "pair_type": pair_type,
                    "training_fragment_a": first_track,
                    "training_fragment_b": second_track,
                    "shared_frame_count": shared_frames,
                    "video_a": first_item.video,
                    "frame_a": first_item.frame,
                    "track_id_a": first_item.track_id,
                    "bbox_a": ",".join(map(str, first_item.bbox)),
                    "video_b": second_item.video,
                    "frame_b": second_item.frame,
                    "track_id_b": second_item.track_id,
                    "bbox_b": ",".join(map(str, second_item.bbox)),
                }
            )
            if completed % 50 == 0 or completed == len(all_examples):
                print(
                    f"  pair samples: {completed}/{len(all_examples)}",
                    flush=True,
                )
    finally:
        dataset.close()

    manifest_path = pair_dir / "pair_samples.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Exported {len(selected_positive)} positive and {len(selected_negative)} "
        f"negative pair samples: {pair_dir}"
    )
    return pair_dir


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


def euclidean_kmeans(
    vectors,
    number_of_individuals,
    seed,
    initial_centers=None,
    restarts=20,
    max_iterations=200,
):
    """Dependency-free Euclidean K-means for idTracker network outputs."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if not 1 <= number_of_individuals <= len(vectors):
        raise ValueError(
            f"Cannot form {number_of_individuals} clusters from {len(vectors)} vectors."
        )
    generator = np.random.default_rng(seed)
    best = None
    attempts = 1 if initial_centers is not None else restarts
    for _ in range(attempts):
        if initial_centers is not None:
            centers = np.asarray(initial_centers, dtype=np.float32).copy()
        else:
            # K-means++ seeding, matching the initialization used by idtracker.ai.
            chosen = [int(generator.integers(len(vectors)))]
            while len(chosen) < number_of_individuals:
                distances = ((vectors[:, None] - vectors[chosen][None]) ** 2).sum(2)
                weights = distances.min(axis=1)
                if weights.sum() == 0:
                    remaining = np.setdiff1d(np.arange(len(vectors)), chosen)
                    chosen.append(int(generator.choice(remaining)))
                else:
                    chosen.append(int(generator.choice(len(vectors), p=weights / weights.sum())))
            centers = vectors[chosen].copy()
        assignments = np.full(len(vectors), -1, dtype=np.int64)
        for _ in range(max_iterations):
            squared = ((vectors[:, None] - centers[None]) ** 2).sum(axis=2)
            updated = squared.argmin(axis=1)
            if np.array_equal(updated, assignments):
                break
            assignments = updated
            for identity in range(number_of_individuals):
                members = vectors[assignments == identity]
                centers[identity] = (
                    members.mean(axis=0)
                    if len(members)
                    else vectors[np.argmax(squared.min(axis=1))]
                )
        squared = ((vectors[:, None] - centers[None]) ** 2).sum(axis=2)
        cost = float(squared[np.arange(len(vectors)), assignments].sum())
        if best is None or cost < best[0]:
            best = cost, assignments.copy(), centers.copy(), squared
    _, assignments, centers, squared = best
    return assignments, centers, squared


def cluster_identities(track_embeddings, number_of_individuals, seed, metric="cosine"):
    """Original unconstrained track-level clustering baseline."""
    cluster = euclidean_kmeans if metric == "euclidean" else cosine_kmeans
    assignments, _, _ = cluster(track_embeddings, number_of_individuals, seed)
    return assignments


def find_reference_training_fragments(
    detections,
    fragments,
    number_of_individuals,
    minimum_frames,
):
    """Find a complete coexisting group among conservative fragments."""
    detection_to_fragment = np.full(len(detections), -1, dtype=np.int64)
    fragment_lengths = np.zeros(len(fragments.keys), dtype=np.int64)
    for fragment, indices in fragments.indices_by_fragment.items():
        detection_to_fragment[indices] = fragment
        fragment_lengths[fragment] = len(indices)

    fragments_by_frame = defaultdict(set)
    for index, item in enumerate(detections):
        fragment = detection_to_fragment[index]
        if fragment >= 0:
            fragments_by_frame[(item.video, item.frame)].add(int(fragment))

    frames_by_group = defaultdict(list)
    for (video, frame), labels in fragments_by_frame.items():
        if len(labels) == number_of_individuals:
            frames_by_group[(video, tuple(sorted(labels)))].append(frame)

    candidates = [
        (key, frames)
        for key, frames in frames_by_group.items()
        if min(fragment_lengths[label] for label in key[1]) >= minimum_frames
    ]
    if not candidates:
        return None
    (video, labels), frames = max(
        candidates,
        key=lambda item: (
            len(item[1]),
            min(fragment_lengths[label] for label in item[0][1]),
        ),
    )
    return {
        "video": video,
        "fragment_labels": labels,
        "coexisting_frames": tuple(frames),
        "coexisting_frame_count": len(frames),
        "minimum_fragment_detections": int(
            min(fragment_lengths[label] for label in labels)
        ),
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


def assign_fragments_with_cooccurrence(p1, known_different, fixed_assignments=None):
    """Assign fragments efficiently while forbidding coexisting duplicates."""
    p1 = np.asarray(p1, dtype=np.float64).copy()
    fragment_count, number_of_individuals = p1.shape
    fixed_assignments = fixed_assignments or {}
    for fragment, identity in fixed_assignments.items():
        p1[fragment] = 0.0
        p1[fragment, identity] = 1.0

    # idtracker.ai's P2 combines each fragment's image evidence with evidence
    # that coexisting fragments are not a given identity.  Calculate it once;
    # repeatedly recalculating every row is prohibitively expensive for the
    # thousands of conservative fragments in a long video.
    p2 = np.empty_like(p1)
    for fragment in range(fragment_count):
        neighbors = np.flatnonzero(known_different[fragment])
        if len(neighbors):
            log_scores = np.log(np.clip(p1[fragment], 1e-300, 1.0))
            log_scores += np.log(
                np.clip(1.0 - p1[neighbors], 1e-300, 1.0)
            ).sum(axis=0)
            log_scores -= log_scores.max()
            scores = np.exp(log_scores)
        else:
            scores = p1[fragment].copy()
        total = scores.sum()
        p2[fragment] = scores / total if total else p1[fragment]

    assignments = np.full(fragment_count, -1, dtype=np.int64)
    confidences = np.zeros(fragment_count, dtype=np.float64)
    for fragment, identity in fixed_assignments.items():
        assignments[fragment] = identity
        confidences[fragment] = 1.0

    sorted_probabilities = np.sort(p2, axis=1)
    if number_of_individuals == 1:
        certainty = sorted_probabilities[:, -1]
    else:
        certainty = sorted_probabilities[:, -1] / np.maximum(
            sorted_probabilities[:, -2], 1e-300
        )
    order = sorted(
        (fragment for fragment in range(fragment_count) if fragment not in fixed_assignments),
        key=lambda fragment: (-certainty[fragment], -p2[fragment].max(), fragment),
    )
    for fragment in order:
        neighbors = np.flatnonzero(known_different[fragment])
        forbidden = set(assignments[neighbors][assignments[neighbors] >= 0].tolist())
        available = np.asarray(
            [identity for identity in range(number_of_individuals) if identity not in forbidden],
            dtype=np.int64,
        )
        if not len(available):
            continue
        scores = p2[fragment, available]
        identity = int(available[np.argmax(scores)])
        assignments[fragment] = identity
        confidences[fragment] = float(p2[fragment, identity])

    return assignments, confidences, p2


def conservative_fragment_identity_assignment(
    embeddings,
    detections,
    fragments,
    number_of_individuals,
    seed,
    global_fragment_min_frames,
    metric="euclidean",
):
    """Cluster safe crop embeddings and vote identities per safe fragment."""
    detection_fragments = np.full(len(detections), -1, dtype=np.int64)
    for fragment, indices in fragments.indices_by_fragment.items():
        detection_fragments[indices] = fragment
    safe_indices = np.flatnonzero(detection_fragments >= 0)
    if len(safe_indices) < number_of_individuals:
        raise ValueError(
            "Too few detections survived conservative fragment filtering to "
            "form the requested identity clusters."
        )

    reference = find_reference_training_fragments(
        detections,
        fragments,
        number_of_individuals,
        global_fragment_min_frames,
    )
    initial_centers = None
    fixed_assignments = {}
    if reference is not None:
        reference_frames = set(reference["coexisting_frames"])
        centers = []
        for identity, fragment in enumerate(reference["fragment_labels"]):
            indices = [
                index
                for index in fragments.indices_by_fragment[fragment]
                if detections[index].frame in reference_frames
            ]
            if not indices:
                indices = fragments.indices_by_fragment[fragment]
            if len(indices) > 500:
                positions = np.linspace(0, len(indices) - 1, 500, dtype=np.int64)
                indices = np.asarray(indices)[positions]
            centers.append(embeddings[indices].mean(axis=0))
            fixed_assignments[fragment] = identity
        initial_centers = np.asarray(centers, dtype=np.float32)

    safe_embeddings = embeddings[safe_indices]
    if metric == "euclidean":
        detection_clusters, _, _ = euclidean_kmeans(
            safe_embeddings,
            number_of_individuals,
            seed,
            initial_centers=initial_centers,
            restarts=5,
            max_iterations=50,
        )
    else:
        detection_clusters, _, _ = cosine_kmeans(
            safe_embeddings,
            number_of_individuals,
            seed,
            initial_centers=initial_centers,
            restarts=5,
        )
    p1 = fragment_identity_probabilities(
        detection_clusters,
        detection_fragments[safe_indices],
        len(fragments.keys),
        number_of_individuals,
    )
    assignments, confidences, p2 = assign_fragments_with_cooccurrence(
        p1, fragments.known_different, fixed_assignments
    )
    assignments, confidences, demoted = enforce_identity_exclusivity(
        assignments, confidences, fragments.known_different
    )
    if demoted:
        print(
            f"Final identity-exclusivity guard demoted {len(demoted)} conflicting "
            "training fragments to -1."
        )

    detection_identities = np.full(len(detections), -1, dtype=np.int64)
    assigned_safe = assignments[detection_fragments[safe_indices]]
    detection_identities[safe_indices] = assigned_safe
    if reference is not None:
        reference = dict(reference)
        reference.pop("coexisting_frames")
        reference["fragment_labels"] = list(reference["fragment_labels"])
    return assignments, confidences, p1, p2, reference, detection_identities


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


def write_detection_identity_rows(path, raw_rows, detection_identities):
    """Write identities that may change between fragments of one source track."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(raw_rows[0]) + ["predicted_identity"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, identity in zip(raw_rows, detection_identities):
            output = dict(row)
            output["predicted_identity"] = int(identity)
            writer.writerow(output)


def write_latents(
    path,
    embeddings,
    detections,
    track_to_label,
    identity_by_track=None,
    projection=None,
    projection_name=None,
    identity_by_detection=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coordinate_names = [f"latent_{index:03d}" for index in range(embeddings.shape[1])]
    fields = [
        "source_video", "frame", "track_id", "x1", "y1", "x2", "y2",
        "obb_cx", "obb_cy", "obb_width", "obb_height", "obb_angle_radians",
    ]
    if identity_by_track is not None or identity_by_detection is not None:
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
            row.extend(item.oriented_bbox or ("", "", "", "", ""))
            if identity_by_detection is not None:
                row.append(int(identity_by_detection[index]))
            elif identity_by_track is not None:
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


def write_fragment_assignment_diagnostics(
    path,
    fragments,
    detections,
    assignments,
    confidences,
    p1,
    p2,
):
    path = Path(path)
    fields = [
        "training_fragment",
        "source_video",
        "source_track_id",
        "first_frame",
        "last_frame",
        "detection_count",
        "predicted_identity",
        "assignment_confidence",
    ]
    fields.extend(f"p1_cluster_probability_{index}" for index in range(p1.shape[1]))
    fields.extend(f"p2_identity_probability_{index}" for index in range(p2.shape[1]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for fragment, indices in fragments.indices_by_fragment.items():
            video, track_id = fragments.original_track_by_fragment[fragment]
            row = [
                fragment,
                video,
                track_id,
                detections[indices[0]].frame,
                detections[indices[-1]].frame,
                len(indices),
                int(assignments[fragment]),
                f"{confidences[fragment]:.8g}",
            ]
            row.extend(f"{value:.8g}" for value in p1[fragment])
            row.extend(f"{value:.8g}" for value in p2[fragment])
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


def fragment_assignment_quality(
    detections,
    fragments,
    assignments,
    confidences,
    reference,
    detection_identities,
):
    identities_by_frame = defaultdict(list)
    for item, identity in zip(detections, detection_identities):
        if identity >= 0:
            identities_by_frame[(item.video, item.frame)].append(int(identity))

    detections_by_frame = defaultdict(int)
    for item in detections:
        detections_by_frame[(item.video, item.frame)] += 1
    multi_frames = sum(count >= 2 for count in detections_by_frame.values())
    # Keep the key available while checking all multi-detection frames, including
    # frames where only one safe fragment retained an identity.
    collision_frames = 0
    for frame_key, count in detections_by_frame.items():
        if count < 2:
            continue
        identities = identities_by_frame.get(frame_key, [])
        if len(identities) != len(set(identities)):
            collision_frames += 1

    assigned = assignments >= 0
    assigned_detection_count = int((detection_identities >= 0).sum())
    return {
        "original_track_count": int(len({item.track_key for item in detections})),
        "fragment_count": int(len(assignments)),
        "assigned_fragments": int(assigned.sum()),
        "unassigned_fragments": int((~assigned).sum()),
        "assigned_fragment_fraction": float(assigned.mean()),
        "detection_count": int(len(detections)),
        "assigned_detections": assigned_detection_count,
        "unassigned_detections": int(len(detections) - assigned_detection_count),
        "assigned_detection_fraction": float(
            assigned_detection_count / len(detections)
        ),
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
    training_fragments,
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
    metric = "euclidean" if model.architecture == "idtracker-resnet18" else "cosine"
    if metric == "cosine":
        track_embeddings = _normalize_rows(track_embeddings)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_by_track = None
    identity_by_detection = None
    if args.mode != "embed":
        if args.num_individuals is None:
            raise ValueError("--num-individuals is required when predicting identities.")
        if args.identity_assignment == "idtracker":
            if training_fragments is None:
                raise ValueError(
                    "Conservative fragment metadata is required for idtracker "
                    "identity assignment."
                )
            (
                identity_by_fragment,
                assignment_confidence,
                p1,
                p2,
                reference,
                identity_by_detection,
            ) = conservative_fragment_identity_assignment(
                embeddings,
                detections,
                training_fragments,
                args.num_individuals,
                args.seed,
                args.global_fragment_min_frames,
                metric,
            )
            if reference is None:
                print(
                    "No sufficiently large full-animal safe-fragment group was found; "
                    "using K-means++ initialization with co-occurrence-aware assignment."
                )
            else:
                print(
                    "Initialized identities from a safe reference group containing "
                    "all animals; its shortest fragment has "
                    f"{reference['minimum_fragment_detections']} detections."
                )
        else:
            identity_by_track = cluster_identities(
                track_embeddings, args.num_individuals, args.seed, metric
            )
            assignment_confidence = np.ones(len(track_embeddings), dtype=np.float64)
            p1 = None
            reference = None
        identity_path = output_dir / "identity_predictions.csv"
        if args.identity_assignment == "idtracker":
            write_detection_identity_rows(
                identity_path, raw_rows, identity_by_detection
            )
            diagnostics_path = output_dir / "identity_assignment_fragments.csv"
            write_fragment_assignment_diagnostics(
                diagnostics_path,
                training_fragments,
                detections,
                identity_by_fragment,
                assignment_confidence,
                p1,
                p2,
            )
            quality = fragment_assignment_quality(
                detections,
                training_fragments,
                identity_by_fragment,
                assignment_confidence,
                reference,
                identity_by_detection,
            )
        else:
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
            f"Assigned {quality.get('assigned_fragments', quality.get('assigned_tracks'))}/"
            f"{quality.get('fragment_count', quality.get('track_count'))} "
            f"{'fragments' if args.identity_assignment == 'idtracker' else 'tracks'}; "
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
            identity_by_detection,
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
        choices=(
            "train-predict",
            "train",
            "predict",
            "embed",
            "sample-crops",
            "sample-pairs",
        ),
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
    parser.add_argument(
        "--frame-step",
        action="append",
        type=int,
        help=(
            "processed-frame interval for an older manifest without frame_step; "
            "give once for all manifests or once per manifest"
        ),
    )
    parser.add_argument("--output-dir", default="./contrastive_output")
    parser.add_argument(
        "--checkpoint",
        help="trained network used by predict, embed, or sampling modes",
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
        "--export-pair-samples",
        nargs="?",
        const=300,
        type=int,
        metavar="COUNT",
        help=(
            "export COUNT positive and COUNT negative training-pair examples; "
            "specifying the option without a count exports 300 of each"
        ),
    )
    parser.add_argument(
        "--identity-assignment",
        choices=("idtracker", "kmeans"),
        default="idtracker",
        help=(
            "identity assignment strategy; idtracker clusters safe crop embeddings, "
            "aggregates votes by conservative fragment, and enforces that coexisting "
            "fragments have different identities"
        ),
    )
    parser.add_argument(
        "--global-fragment-min-frames",
        type=int,
        default=30,
        help=(
            "minimum detections required in every conservative fragment of a complete "
            "coexisting group before it can initialize and anchor identity clusters"
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
    parser.add_argument(
        "--architecture",
        choices=("idtracker-resnet18", "small-cnn"),
        default="idtracker-resnet18",
        help="identity network; checkpoints always restore their saved architecture",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        help="latent dimensions (8 for idTracker ResNet18, 128 for the legacy CNN)",
    )
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--input-size", type=int)
    parser.add_argument(
        "--input-color",
        choices=("grayscale", "rgb"),
        default="grayscale",
        help=(
            "ResNet crop color mode; checkpoints restore their saved mode and the "
            "legacy small-CNN remains RGB-only"
        ),
    )
    parser.add_argument(
        "--margin",
        type=float,
        help="contrastive margin (10 for idTracker training, 0.5 for legacy training)",
    )
    parser.add_argument(
        "--pairs-per-batch",
        type=int,
        default=400,
        help="negative pairs per optimizer step; an equal number of positives is added",
    )
    parser.add_argument(
        "--forward-pair-chunk-size",
        type=int,
        default=128,
        help="pairs evaluated at once inside a large idTracker-style optimizer batch",
    )
    parser.add_argument(
        "--min-fragment-length",
        type=int,
        default=4,
        help="minimum detections for a fragment to participate in pair training",
    )
    parser.add_argument(
        "--training-fragments",
        choices=("conservative", "tracks"),
        default="conservative",
        help=(
            "split motion tracks around locally ambiguous events, or retain the "
            "original whole tracks for comparison"
        ),
    )
    parser.add_argument(
        "--fragment-close-distance",
        type=float,
        default=0.75,
        help="uncertain center distance in units of mean bounding-box size",
    )
    parser.add_argument(
        "--fragment-assignment-margin",
        type=float,
        default=0.35,
        help="maximum normalized direct-versus-swapped motion cost advantage",
    )
    parser.add_argument(
        "--fragment-motion-jump",
        type=float,
        default=1.5,
        help="maximum constant-velocity prediction error in box-size units",
    )
    parser.add_argument(
        "--fragment-uncertainty-buffer",
        type=int,
        default=5,
        help="observed frames discarded on each side of an ambiguous event",
    )
    parser.add_argument(
        "--fragment-interpolation-gap",
        type=int,
        default=10,
        help="maximum internal gap interpolated only for uncertainty detection",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        help="optimizer steps between silhouette checks; auto is max(5*N, 100)",
    )
    parser.add_argument("--validation-warmup", type=int, default=5)
    parser.add_argument("--silhouette-target", type=float, default=0.91)
    parser.add_argument("--validation-patience", type=int, default=30)
    parser.add_argument("--validation-images-per-individual", type=int, default=1000)
    parser.add_argument("--silhouette-samples", type=int, default=2000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help=(
            "legacy small-CNN image batch; must be divisible by "
            "--samples-per-track (the ResNet uses --pairs-per-batch)"
        ),
    )
    parser.add_argument(
        "--tracks-per-batch",
        type=int,
        help=(
            "legacy small-CNN override for tracks per batch; when supplied, "
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
    if args.frame_step:
        if len(args.frame_step) not in {1, len(args.manifest)}:
            parser.error("--frame-step must be given once or once per manifest")
        if any(step < 1 for step in args.frame_step):
            parser.error("--frame-step must be at least 1")
    if args.pairs_per_batch < 1:
        parser.error("--pairs-per-batch must be at least 1")
    if args.forward_pair_chunk_size < 1:
        parser.error("--forward-pair-chunk-size must be at least 1")
    if args.min_fragment_length < 2:
        parser.error("--min-fragment-length must be at least 2")
    if args.fragment_close_distance < 0 or args.fragment_assignment_margin < 0:
        parser.error("fragment distance and assignment margin cannot be negative")
    if args.fragment_motion_jump <= 0:
        parser.error("--fragment-motion-jump must be greater than zero")
    if args.fragment_uncertainty_buffer < 0 or args.fragment_interpolation_gap < 0:
        parser.error("fragment buffer and interpolation gap cannot be negative")
    if args.validation_interval is not None and args.validation_interval < 1:
        parser.error("--validation-interval must be at least 1")
    if args.validation_warmup < 0 or args.validation_patience < 1:
        parser.error("validation warmup/patience values are invalid")
    if args.validation_images_per_individual < 1 or args.silhouette_samples < 1:
        parser.error("validation sample limits must be at least 1")
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
    if args.export_pair_samples is not None and args.export_pair_samples < 1:
        parser.error("--export-pair-samples must be at least 1")
    if args.mode == "sample-crops" and args.export_crop_samples is None:
        args.export_crop_samples = 100
    if args.mode == "sample-pairs" and args.export_pair_samples is None:
        args.export_pair_samples = 300
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

    detections, raw_rows = read_manifests(
        args.manifest, args.video, args.frame_step
    )
    metadata = build_track_metadata(detections)
    training_fragments = None
    if (
        args.mode in {"train", "train-predict", "sample-pairs"}
        or (args.mode == "predict" and args.identity_assignment == "idtracker")
        or args.export_pair_samples
    ):
        if args.training_fragments == "conservative":
            print("Building conservative contiguous training fragments...", flush=True)
            training_fragments = build_conservative_training_fragments(
                detections,
                close_distance=args.fragment_close_distance,
                assignment_margin=args.fragment_assignment_margin,
                motion_jump=args.fragment_motion_jump,
                uncertainty_buffer=args.fragment_uncertainty_buffer,
                interpolation_gap=args.fragment_interpolation_gap,
                min_fragment_length=args.min_fragment_length,
            )
        else:
            training_fragments = use_original_tracks_as_training_fragments(metadata)
        retained = sum(
            len(indices)
            for indices in training_fragments.indices_by_fragment.values()
        )
        print(
            f"Training fragments: {len(training_fragments.keys)} fragments retain "
            f"{retained}/{len(detections)} detections ({retained / len(detections):.1%}); "
            f"{training_fragments.excluded_detection_count} detections excluded.",
            flush=True,
        )
        fragment_summary = write_training_fragment_summary(
            Path(args.output_dir) / "training_fragments.csv",
            training_fragments,
            detections,
        )
        print(f"Saved training fragment summary: {fragment_summary}")
    preloaded_crop_store = None

    if args.mode in {"train", "train-predict"}:
        model, input_size, _, preloaded_crop_store = train_model(
            args,
            detections,
            metadata,
            device,
            background_store,
            preloaded_crop_store,
            training_fragments,
        )
    else:
        model, _, _, input_size = make_model(args, device, args.checkpoint)
        # A single inference pass is already ordered by the manifest, so AUTO does
        # not spend time and storage materializing every crop. Explicit RAM or disk
        # modes remain available when repeated access is desired.
        if args.crop_storage in {"ram", "disk"}:
            preloaded_crop_store = make_preloaded_crop_store(
                args, detections, input_size, background_store, model.input_channels
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
            model.input_channels,
        )

    if args.export_pair_samples:
        export_pair_samples(
            args.output_dir,
            args.export_pair_samples,
            detections,
            metadata,
            input_size,
            background_store,
            args.background_threshold,
            args.background_morph_kernel,
            preloaded_crop_store,
            model.input_channels,
            args.min_fragment_length,
            args.seed,
            training_fragments,
        )

    if args.mode in {"train-predict", "predict", "embed"}:
        if (
            preloaded_crop_store is not None
            and len(preloaded_crop_store.detection_indices) < len(detections)
        ):
            print(
                "Releasing the training-only crop cache; full-video embedding "
                "will stream frames sequentially.",
                flush=True,
            )
            preloaded_crop_store.release()
            preloaded_crop_store = None
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
            training_fragments,
        )


if __name__ == "__main__":
    main()
