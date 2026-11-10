#!/usr/bin/env python3
"""Classify a small crop sample from each contrastive identity.

The winning secondary class is chosen by averaging classifier probabilities
across the sampled crops, then written onto every detection with that assigned
identity. Track IDs and contrastive identity assignments are left unchanged.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

import simple_track
from contrastive_identity import extract_oriented_crop, read_manifests


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Apply a secondary-static classifier to sampled crops from each "
            "contrastive identity and add secondary_class to the prediction CSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="source video")
    parser.add_argument(
        "--predictions",
        required=True,
        help="identity_predictions.csv produced by contrastive_identity.py",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="secondary-static best.pt file or model directory",
    )
    parser.add_argument(
        "--output",
        help="output CSV (default: identity_predictions_secondary.csv beside input)",
    )
    parser.add_argument(
        "--samples-per-identity",
        type=int,
        default=100,
        help="maximum evenly spaced crops classified for each assigned identity",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device",
        default="auto",
        help="Ultralytics device, such as 0 or cpu; auto uses its default",
    )
    return parser


def resolve_model_path(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path
    candidates = (path / "train/weights/best.pt", path / "weights/best.pt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find classifier weights at {path} or beneath its weights directory"
    )


def sample_indices_by_identity(raw_rows, samples_per_identity):
    indices_by_identity = defaultdict(list)
    for index, row in enumerate(raw_rows):
        identity = int(row["predicted_identity"])
        if identity >= 0:
            indices_by_identity[identity].append(index)
    if not indices_by_identity:
        raise ValueError("The prediction CSV contains no assigned identities")

    selected = {}
    for identity, indices in sorted(indices_by_identity.items()):
        positions = np.linspace(
            0,
            len(indices) - 1,
            min(samples_per_identity, len(indices)),
            dtype=np.int64,
        )
        selected[identity] = [indices[int(position)] for position in np.unique(positions)]
    return selected


def crop_detection(frame, detection):
    if detection.oriented_bbox is not None:
        return extract_oriented_crop(frame, detection.oriented_bbox)
    x1, y1, x2, y2 = detection.bbox
    height, width = frame.shape[:2]
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid crop at frame {detection.frame}, track {detection.track_id}"
        )
    return frame[y1:y2, x1:x2]


def read_sample_crops(video, detections, selected):
    requests = sorted(
        (detections[index].frame, identity, index)
        for identity, indices in selected.items()
        for index in indices
    )
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {video}")

    crops = []
    crop_identities = []
    current_frame_number = None
    current_frame = None
    try:
        for request_number, (frame_number, identity, index) in enumerate(requests, start=1):
            if frame_number != current_frame_number:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                success, current_frame = capture.read()
                if not success:
                    raise RuntimeError(f"Could not read frame {frame_number} from {video}")
                current_frame_number = frame_number
            crop = crop_detection(current_frame, detections[index])
            if crop is None or crop.size == 0:
                raise ValueError(
                    f"Empty crop at frame {frame_number}, track "
                    f"{detections[index].track_id}"
                )
            crops.append(crop)
            crop_identities.append(identity)
            if request_number % 100 == 0 or request_number == len(requests):
                print(
                    f"  Loaded sample crops: {request_number}/{len(requests)}",
                    flush=True,
                )
    finally:
        capture.release()
    return crops, crop_identities


def classify_identities(model, crops, crop_identities, batch_size, device):
    predict_args = {"source": crops, "batch": batch_size, "verbose": False}
    if device != "auto":
        predict_args["device"] = device
    started = perf_counter()
    results = model.predict(**predict_args)
    probability_sums = {}
    sample_counts = defaultdict(int)
    for identity, result in zip(crop_identities, results):
        if result.probs is None:
            raise RuntimeError("The supplied model did not return classification probabilities")
        probabilities = result.probs.data.detach().cpu().numpy().astype(np.float64)
        if identity not in probability_sums:
            probability_sums[identity] = np.zeros_like(probabilities)
        probability_sums[identity] += probabilities
        sample_counts[identity] += 1

    assignments = {}
    for identity in sorted(probability_sums):
        mean_probabilities = probability_sums[identity] / sample_counts[identity]
        class_index = int(np.argmax(mean_probabilities))
        class_name = str(model.names[class_index])
        confidence = float(mean_probabilities[class_index])
        assignments[identity] = class_name
        print(
            f"  Identity {identity}: {class_name} "
            f"({confidence:.1%}, {sample_counts[identity]} samples)",
            flush=True,
        )
    print(f"Classified {len(crops)} crops in {perf_counter() - started:.1f}s")
    return assignments


def write_output(path, raw_rows, assignments):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(raw_rows[0])
    if "secondary_class" not in fieldnames:
        fieldnames.append("secondary_class")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in raw_rows:
            output = dict(row)
            identity = int(row["predicted_identity"])
            output["secondary_class"] = assignments.get(identity, "")
            writer.writerow(output)
    print(f"Saved secondary classifications: {path}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.samples_per_identity < 1:
        parser.error("--samples-per-identity must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    video = Path(args.video).expanduser().resolve()
    predictions = Path(args.predictions).expanduser().resolve()
    model_path = resolve_model_path(args.model)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else predictions.with_name("identity_predictions_secondary.csv")
    )

    detections, raw_rows = read_manifests([predictions], [video])
    if "predicted_identity" not in raw_rows[0]:
        parser.error("--predictions must contain a predicted_identity column")
    selected = sample_indices_by_identity(raw_rows, args.samples_per_identity)
    print(
        f"Sampling {sum(map(len, selected.values()))} crops across "
        f"{len(selected)} identities.",
        flush=True,
    )
    crops, crop_identities = read_sample_crops(video, detections, selected)

    simple_track.load_runtime_dependencies()
    model = simple_track.YOLO(str(model_path), task="classify")
    assignments = classify_identities(
        model, crops, crop_identities, args.batch_size, args.device
    )
    write_output(output, raw_rows, assignments)


if __name__ == "__main__":
    main()
