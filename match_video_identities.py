#!/usr/bin/env python3
"""Train one encoder and match contrastive identities across two or three videos.

The inputs are crop manifests produced by simple_track.py. Each video is clustered
independently with the shared encoder, then the small set of possible cluster-to-fish
mappings is scored using known tank locations and cross-video embedding similarity.
"""

import argparse
import csv
import itertools
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from findCenter import find_video_center


SCRIPT_DIR = Path(__file__).resolve().parent
ROLE_ORDER = {"B": 0, "T": 1, "F": 2, "A": 3}
VIDEO_NAME = re.compile(
    r"^pi(?P<pi>\d+)\.(?P<year>\d{4})\.(?P<month>\d{2})\."
    r"(?P<day>\d{2})\.(?P<hour>\d{2})\.(?P<minute>\d{2})"
)
DEFAULT_INCLUSION_SIZE = (880.0, 1030.0)


@dataclass(frozen=True)
class AssignmentInterval:
    start: float
    stop: float
    fish_ids: tuple
    fish_locations: tuple


@dataclass
class VideoData:
    manifest: Path
    video: Path
    stem: str
    pi: str
    day: str
    clock_start: float
    fps: float
    width: int
    height: int
    frame_count: int
    intervals: list
    fish_ids: tuple
    center_x: float = None
    center_y: float = None
    local_output: Path = None
    prediction_csv: Path = None
    latent_csv: Path = None
    cluster_ids: tuple = ()
    prototypes: dict = None
    compatibility: dict = None
    cluster_counts: dict = None


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Use one shared contrastive encoder to match identities across two or "
            "three simple_track crop manifests."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifests",
        nargs="+",
        required=True,
        help="two or three crop manifests produced by simple_track.py",
    )
    parser.add_argument(
        "--assignments",
        required=True,
        help="video/fish location CSV, such as SurpriseWinnerEffectVideoAssignments.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--input-color", choices=("grayscale", "rgb"), default="grayscale"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--init-checkpoint",
        help="optional existing contrastive model used to initialize shared training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="maximum training epochs; omit to use contrastive_identity.py's default",
    )
    parser.add_argument(
        "--background-subtraction",
        action="store_true",
        help="use median-background masking while training and embedding",
    )
    parser.add_argument(
        "--min-location-fraction",
        type=float,
        default=0.5,
        help=(
            "minimum fraction of a local identity's detections compatible with a fish's "
            "known locations"
        ),
    )
    parser.add_argument(
        "--center",
        nargs=2,
        type=float,
        action="append",
        metavar=("X", "Y"),
        help=(
            "arena center in original-video pixels; give once for all videos or once "
            "per manifest (default: detect each video's precise center)"
        ),
    )
    parser.add_argument(
        "--exclude-edges",
        action="store_true",
        help="exclude detections outside the centered inclusion area",
    )
    parser.add_argument(
        "--inclusion-size",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        help=(
            "centered inclusion size in pixels; implies --exclude-edges "
            f"(default when excluding: {DEFAULT_INCLUSION_SIZE[0]:g} "
            f"{DEFAULT_INCLUSION_SIZE[1]:g})"
        ),
    )
    return parser


def run_command(label, command):
    command = [str(part) for part in command]
    print(f"\n=== {label} ===", flush=True)
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def manifest_video(path):
    videos = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "source_video" not in (reader.fieldnames or ()):
            raise ValueError(f"Manifest has no source_video column: {path}")
        for row in reader:
            if row["source_video"]:
                videos.add(Path(row["source_video"]).expanduser().resolve())
            if len(videos) > 1:
                raise ValueError(f"Each manifest must contain exactly one video: {path}")
    if not videos:
        raise ValueError(f"Manifest contains no detections: {path}")
    return videos.pop()


def read_video_metadata(manifest):
    video = manifest_video(manifest)
    match = VIDEO_NAME.match(video.name)
    if match is None:
        raise ValueError(
            "Video names must begin with piXX.YYYY.MM.DD.HH.MM: " + video.name
        )
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError(f"Video reports invalid metadata: {video}")
    values = match.groupdict()
    clock_start = int(values["hour"]) * 3600 + int(values["minute"]) * 60
    return VideoData(
        manifest=manifest,
        video=video,
        stem=video.stem,
        pi=f"pi{int(values['pi']):02d}",
        day=f"{values['year']}-{values['month']}-{values['day']}",
        clock_start=clock_start,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        intervals=[],
        fish_ids=(),
    )


def clock_seconds(value):
    parsed = datetime.strptime(value, "%H:%M").time()
    return parsed.hour * 3600 + parsed.minute * 60


def read_assignment_rows(path):
    rows = []
    required = {
        "pi",
        "date",
        "timeStart",
        "timeStop",
        "fishIDs",
        "fishLocs",
    }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Assignment CSV is missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            fish_ids = tuple(json.loads(row["fishIDs"]))
            fish_locations = tuple(json.loads(row["fishLocs"]))
            if len(fish_ids) != 4 or len(fish_locations) != 4:
                raise ValueError("fishIDs and fishLocs must each contain four entries")
            rows.append(
                (
                    row["pi"],
                    row["date"],
                    AssignmentInterval(
                        clock_seconds(row["timeStart"]),
                        clock_seconds(row["timeStop"]),
                        fish_ids,
                        fish_locations,
                    ),
                )
            )
    return rows


def attach_assignments(video, assignment_rows):
    clip_stop = video.clock_start + video.frame_count / video.fps
    intervals = [
        interval
        for pi, day, interval in assignment_rows
        if pi == video.pi
        and day == video.day
        and interval.stop > video.clock_start
        and interval.start < clip_stop
    ]
    if not intervals:
        raise ValueError(
            f"No assignment intervals overlap {video.video.name} ({video.pi}, {video.day})"
        )
    intervals.sort(key=lambda interval: interval.start)

    # A role may be absent for part of a clip, but it cannot change from one fish
    # to another. Splitting at replacements remains a deliberate prerequisite.
    for role_index, role_name in enumerate(("Big", "Treatment", "Focal", "Assay")):
        role_ids = {
            interval.fish_ids[role_index]
            for interval in intervals
            if interval.fish_ids[role_index] is not None
        }
        if len(role_ids) > 1:
            raise ValueError(
                f"{video.video.name} contains a {role_name} replacement ({sorted(role_ids)}). "
                "Split the video at the replacement time and regenerate its crop manifest."
            )

    fish_ids = tuple(
        sorted(
            {
                fish_id
                for interval in intervals
                for fish_id in interval.fish_ids
                if fish_id is not None
            },
            key=fish_sort_key,
        )
    )
    if len(fish_ids) < 2:
        raise ValueError(
            f"At least two known fish are required in {video.video.name}; found {fish_ids}"
        )
    video.intervals = intervals
    video.fish_ids = fish_ids


def fish_sort_key(fish_id):
    match = re.fullmatch(r"([BTFA])(\d+)\.(\d+)\.(\d+)", fish_id)
    if match is None:
        return (99, fish_id)
    role, pi, block, index = match.groups()
    return (ROLE_ORDER[role], int(pi), int(block), int(index))


def interval_at(video, frame):
    moment = video.clock_start + (int(frame) - 1) / video.fps
    for interval in video.intervals:
        if interval.start <= moment < interval.stop:
            return interval
    raise ValueError(
        f"Frame {frame} of {video.video.name} falls outside assignment intervals"
    )


def detection_quadrant(row, video):
    center_x, center_y = detection_center(row)
    horizontal = "Left" if center_x < video.center_x else "Right"
    vertical = "top" if center_y < video.center_y else "bottom"
    return vertical + horizontal


def detection_center(row):
    if row.get("obb_cx") not in {None, ""} and row.get("obb_cy") not in {None, ""}:
        return float(row["obb_cx"]), float(row["obb_cy"])
    return (
        (float(row["x1"]) + float(row["x2"])) / 2,
        (float(row["y1"]) + float(row["y2"])) / 2,
    )


def detection_is_included(row, video, inclusion_size):
    center_x, center_y = detection_center(row)
    half_width = inclusion_size[0] / 2
    half_height = inclusion_size[1] / 2
    return (
        video.center_x - half_width <= center_x <= video.center_x + half_width
        and video.center_y - half_height <= center_y <= video.center_y + half_height
    )


def filter_manifest(video, destination, inclusion_size):
    destination.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    excluded = 0
    with video.manifest.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        with destination.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if detection_is_included(row, video, inclusion_size):
                    writer.writerow(row)
                    kept += 1
                else:
                    excluded += 1
    if kept == 0:
        raise ValueError(
            f"Edge exclusion removed every detection from {video.video.name}"
        )
    video.manifest = destination
    print(
        f"{video.video.name}: retained {kept:,} detections and excluded "
        f"{excluded:,} outside the {inclusion_size[0]:g}x{inclusion_size[1]:g} area."
    )


def location_allows(location, quadrant):
    if location is None:
        return False
    if location in {"topLeft", "topRight", "bottomLeft", "bottomRight"}:
        return location == quadrant
    if location == "Left":
        return quadrant in {"topLeft", "bottomLeft"}
    if location == "Right":
        return quadrant in {"topRight", "bottomRight"}
    if location == "Bottom":
        return quadrant in {"bottomLeft", "bottomRight"}
    raise ValueError(f"Unknown fish location: {location!r}")


def load_cluster_data(video):
    with video.latent_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        latent_fields = sorted(
            name for name in (reader.fieldnames or ()) if name.startswith("latent_")
        )
        if not latent_fields:
            raise ValueError(f"No latent coordinates found in {video.latent_csv}")
        vectors = {}
        compatibility_counts = {}
        cluster_counts = {}
        for row in reader:
            cluster = int(row["predicted_identity"])
            if cluster < 0:
                continue
            vectors.setdefault(cluster, []).append(
                np.asarray([float(row[name]) for name in latent_fields], dtype=np.float32)
            )
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            interval = interval_at(video, row["frame"])
            quadrant = detection_quadrant(row, video)
            for fish_id in video.fish_ids:
                matches = [
                    location_allows(location, quadrant)
                    for known_id, location in zip(
                        interval.fish_ids, interval.fish_locations
                    )
                    if known_id == fish_id
                ]
                key = (cluster, fish_id)
                compatible, total = compatibility_counts.get(key, (0, 0))
                compatibility_counts[key] = (
                    compatible + int(bool(matches) and matches[0]),
                    total + 1,
                )
    if not vectors:
        raise ValueError(f"No assigned identities found in {video.latent_csv}")
    video.cluster_ids = tuple(sorted(vectors))
    video.prototypes = {
        cluster: np.median(np.stack(items), axis=0)
        for cluster, items in vectors.items()
    }
    video.compatibility = {
        key: compatible / total
        for key, (compatible, total) in compatibility_counts.items()
    }
    video.cluster_counts = cluster_counts
    if len(video.cluster_ids) > len(video.fish_ids):
        raise ValueError(
            f"{video.video.name} produced {len(video.cluster_ids)} local identities but "
            f"only {len(video.fish_ids)} fish occur in the assignment table"
        )


def feasible_mappings(video, minimum_fraction):
    mappings = []
    for fish_permutation in itertools.permutations(
        video.fish_ids, len(video.cluster_ids)
    ):
        mapping = dict(zip(video.cluster_ids, fish_permutation))
        fractions = [
            video.compatibility[(cluster, fish_id)]
            for cluster, fish_id in mapping.items()
        ]
        if all(fraction >= minimum_fraction for fraction in fractions):
            mappings.append((mapping, sum(1 - fraction for fraction in fractions)))
    if mappings:
        return mappings

    best = []
    for cluster in video.cluster_ids:
        ranked = sorted(
            (
                (video.compatibility[(cluster, fish_id)], fish_id)
                for fish_id in video.fish_ids
            ),
            reverse=True,
        )
        best.append(f"local {cluster}: {ranked[0][1]}={ranked[0][0]:.1%}")
    raise ValueError(
        f"No location-compatible identity mapping for {video.video.name} at "
        f"--min-location-fraction {minimum_fraction:.2f}. Best matches: "
        + "; ".join(best)
    )


def mapping_score(videos, choices, appearance_scale):
    location_cost = sum(choice[1] for choice in choices)
    appearance_cost = 0.0
    comparisons = 0
    fish_to_clusters = [
        {fish_id: cluster for cluster, fish_id in choice[0].items()}
        for choice in choices
    ]
    for first_index, second_index in itertools.combinations(range(len(videos)), 2):
        common = set(fish_to_clusters[first_index]).intersection(
            fish_to_clusters[second_index]
        )
        for fish_id in common:
            first_cluster = fish_to_clusters[first_index][fish_id]
            second_cluster = fish_to_clusters[second_index][fish_id]
            appearance_cost += float(
                np.linalg.norm(
                    videos[first_index].prototypes[first_cluster]
                    - videos[second_index].prototypes[second_cluster]
                )
            ) / appearance_scale
            comparisons += 1
    return appearance_cost + location_cost, comparisons


def solve_mappings(videos, minimum_fraction):
    candidates = [feasible_mappings(video, minimum_fraction) for video in videos]
    all_distances = []
    for first, second in itertools.combinations(videos, 2):
        for first_vector in first.prototypes.values():
            for second_vector in second.prototypes.values():
                all_distances.append(float(np.linalg.norm(first_vector - second_vector)))
    positive = [distance for distance in all_distances if distance > 1e-12]
    appearance_scale = float(np.median(positive)) if positive else 1.0

    best = None
    second_best = None
    for choices in itertools.product(*candidates):
        score, comparisons = mapping_score(videos, choices, appearance_scale)
        result = (score, choices, comparisons)
        if best is None or score < best[0]:
            second_best = best
            best = result
        elif second_best is None or score < second_best[0]:
            second_best = result
    if best is None:
        raise ValueError("No feasible cross-video identity mapping was found")
    print(
        f"Selected constrained mapping score {best[0]:.4f} using "
        f"{best[2]} cross-video appearance comparisons."
    )
    if second_best is not None:
        print(f"Next-best score: {second_best[0]:.4f}")
    return [choice[0] for choice in best[1]]


def write_outputs(output_dir, videos, mappings):
    all_fish = sorted(
        {fish_id for video in videos for fish_id in video.fish_ids},
        key=fish_sort_key,
    )
    global_identity = {fish_id: index for index, fish_id in enumerate(all_fish)}

    mapping_path = output_dir / "identity_mapping.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("predicted_identity", "fish_id"))
        for fish_id, identity in global_identity.items():
            writer.writerow((identity, fish_id))

    diagnostic_path = output_dir / "video_identity_mapping.csv"
    with diagnostic_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "video",
                "local_identity",
                "predicted_identity",
                "fish_id",
                "location_compatibility",
                "detection_count",
            )
        )
        for video, mapping in zip(videos, mappings):
            for cluster, fish_id in sorted(mapping.items()):
                writer.writerow(
                    (
                        video.video,
                        cluster,
                        global_identity[fish_id],
                        fish_id,
                        f"{video.compatibility[(cluster, fish_id)]:.6f}",
                        video.cluster_counts[cluster],
                    )
                )

    prediction_paths = []
    for video, mapping in zip(videos, mappings):
        destination = output_dir / video.stem / "identity_predictions.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with video.prediction_csv.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            with destination.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    local_identity = int(row["predicted_identity"])
                    if local_identity >= 0:
                        fish_id = mapping.get(local_identity)
                        row["predicted_identity"] = (
                            global_identity[fish_id] if fish_id is not None else -1
                        )
                    writer.writerow(row)
        prediction_paths.append(destination)
        print(f"Saved matched predictions: {destination}")
    print(f"Saved global identity mapping: {mapping_path}")
    print(f"Saved per-video mapping diagnostics: {diagnostic_path}")
    return prediction_paths


def main():
    parser = build_parser()
    args = parser.parse_args()
    if len(args.manifests) not in {2, 3}:
        parser.error("--manifests requires exactly two or three files")
    if not 0 <= args.min_location_fraction <= 1:
        parser.error("--min-location-fraction must be between 0 and 1")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.inclusion_size and any(value <= 0 for value in args.inclusion_size):
        parser.error("--inclusion-size WIDTH and HEIGHT must both be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = [Path(path).expanduser().resolve() for path in args.manifests]
    videos = [read_video_metadata(manifest) for manifest in manifests]
    if len({video.video for video in videos}) != len(videos):
        raise ValueError("Each manifest must refer to a different source video")
    if len({video.pi for video in videos}) != 1:
        raise ValueError("All videos in one matching run must come from the same Pi")

    supplied_centers = args.center or []
    if len(supplied_centers) not in {0, 1, len(videos)}:
        parser.error("--center must be given once or once per manifest")
    for position, video in enumerate(videos):
        if supplied_centers:
            center = (
                supplied_centers[0]
                if len(supplied_centers) == 1
                else supplied_centers[position]
            )
            score_text = "provided"
        else:
            center, score = find_video_center(video.video)
            score_text = f"detected; match score {score:.4f}"
        video.center_x, video.center_y = center
        print(
            f"{video.video.name}: arena center "
            f"({video.center_x:.1f}, {video.center_y:.1f}) [{score_text}]"
        )

    exclude_edges = args.exclude_edges or args.inclusion_size is not None
    inclusion_size = tuple(args.inclusion_size or DEFAULT_INCLUSION_SIZE)
    if exclude_edges:
        filtered_dir = output_dir / "filtered_manifests"
        for video in videos:
            destination = filtered_dir / f"{video.stem}_crop_manifest.csv"
            filter_manifest(video, destination, inclusion_size)

    assignment_rows = read_assignment_rows(
        Path(args.assignments).expanduser().resolve()
    )
    for video in videos:
        attach_assignments(video, assignment_rows)
        print(
            f"{video.video.name}: {len(video.fish_ids)} fish in analyzed interval: "
            + ", ".join(video.fish_ids)
        )

    all_fish = {fish_id for video in videos for fish_id in video.fish_ids}
    shared_dir = output_dir / "shared_model"
    train_command = [
        sys.executable,
        SCRIPT_DIR / "contrastive_identity.py",
        "--mode",
        "train",
        "--output-dir",
        shared_dir,
        "--num-individuals",
        len(all_fish),
        "--input-color",
        args.input_color,
        "--device",
        args.device,
    ]
    for video in videos:
        train_command.extend(("--manifest", video.manifest))
    if args.init_checkpoint:
        train_command.extend(("--init-checkpoint", args.init_checkpoint))
    if args.epochs is not None:
        train_command.extend(("--epochs", args.epochs))
    if args.background_subtraction:
        train_command.extend(("--background-subtraction", "median"))
    run_command("Training shared contrastive encoder", train_command)

    checkpoint = shared_dir / "contrastive_identity.pt"
    for position, video in enumerate(videos, start=1):
        local_output = output_dir / "local" / video.stem
        latent_csv = local_output / "latents.csv"
        predict_command = [
            sys.executable,
            SCRIPT_DIR / "contrastive_identity.py",
            "--mode",
            "predict",
            "--manifest",
            video.manifest,
            "--output-dir",
            local_output,
            "--checkpoint",
            checkpoint,
            "--num-individuals",
            len(video.fish_ids),
            "--export-latents",
            latent_csv,
            "--latent-projection",
            "none",
            "--device",
            args.device,
        ]
        run_command(
            f"Predicting local identities {position}/{len(videos)}", predict_command
        )
        video.local_output = local_output
        video.prediction_csv = local_output / "identity_predictions.csv"
        video.latent_csv = latent_csv
        load_cluster_data(video)

    mappings = solve_mappings(videos, args.min_location_fraction)
    write_outputs(output_dir, videos, mappings)
    print(f"\nCross-video identity matching complete: {output_dir}")


if __name__ == "__main__":
    main()
