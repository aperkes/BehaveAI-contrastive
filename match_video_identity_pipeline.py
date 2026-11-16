#!/usr/bin/env python3
"""Track two or three videos, then match their identities across videos."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MAX_TRAINING_CROPS = 250_000
DEFAULT_MAX_EMBEDDING_CROPS = 250_000


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run simple_track.py on two or three videos, then match their identities "
            "with match_video_identities.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        help="BehaveAI project containing settings and trained detection weights",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="two or three source MP4 files",
    )
    parser.add_argument(
        "--assignments",
        required=True,
        help="video/fish location assignment CSV",
    )
    parser.add_argument(
        "--output-dir",
        help="output root; defaults to PROJECT/output/START_to_END_identity_match",
    )
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
        help="maximum contrastive training epochs; omit to use the standard default",
    )
    parser.add_argument(
        "--background-subtraction",
        action="store_true",
        help="use median-background masking during contrastive learning",
    )
    parser.add_argument(
        "--min-location-fraction",
        type=float,
        default=0.5,
        help="minimum location compatibility required for an identity match",
    )
    parser.add_argument(
        "--max-training-crops",
        type=int,
        default=DEFAULT_MAX_TRAINING_CROPS,
        help="shared contrastive training crop cap; 0 disables the cap",
    )
    parser.add_argument(
        "--max-embedding-crops",
        type=int,
        default=DEFAULT_MAX_EMBEDDING_CROPS,
        help="per-video identity embedding crop cap; 0 disables the cap",
    )
    parser.add_argument(
        "--center",
        nargs=2,
        type=float,
        action="append",
        metavar=("X", "Y"),
        help="center for all videos, or repeat once per video (default: auto-detect)",
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
        help="custom inclusion size; implies --exclude-edges",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="rerun tracking, training, and embedding even when outputs exist",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="render the first 10 minutes of each video with matched identities",
    )
    return parser


def run_stage(label, command):
    command = [str(part) for part in command]
    print(f"\n=== {label} ===", flush=True)
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if len(args.videos) not in {2, 3}:
        parser.error("--videos requires exactly two or three MP4 files")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if not 0 <= args.min_location_fraction <= 1:
        parser.error("--min-location-fraction must be between 0 and 1")
    if args.center and len(args.center) not in {1, len(args.videos)}:
        parser.error("--center must be given once or once per video")
    if args.inclusion_size and any(value <= 0 for value in args.inclusion_size):
        parser.error("--inclusion-size WIDTH and HEIGHT must both be positive")
    if args.max_training_crops < 0 or args.max_embedding_crops < 0:
        parser.error("crop caps must be zero (uncapped) or positive")

    project_dir = Path(args.project_dir).expanduser().resolve()
    videos = [Path(video).expanduser().resolve() for video in args.videos]
    assignments = Path(args.assignments).expanduser().resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        name = f"{videos[0].stem}_to_{videos[-1].stem}_identity_match"
        output_dir = project_dir / "output" / name

    tracking_root = output_dir / "tracking"
    matching_dir = output_dir / "matched"
    shared_checkpoint = matching_dir / "shared_model" / "contrastive_identity.pt"
    manifests = []

    for position, video in enumerate(videos, start=1):
        tracking_dir = tracking_root / video.stem
        tracking_dir.mkdir(parents=True, exist_ok=True)
        manifest = tracking_dir / f"{video.stem}_crop_manifest.csv"
        completion_marker = tracking_dir / ".tracking_complete"
        tracking_command = [
            sys.executable,
            SCRIPT_DIR / "simple_track.py",
            "--project-dir",
            project_dir,
            "--in_file",
            video,
            "--out_folder",
            tracking_dir,
            "--export-crop-manifest",
        ]
        reusable = (
            manifest.is_file()
            and manifest.stat().st_size > 0
            and (completion_marker.is_file() or shared_checkpoint.is_file())
            and not args.force_recompute
        )
        if reusable:
            print(
                f"\n=== Reusing tracking video {position}/{len(videos)} ===\n{manifest}",
                flush=True,
            )
            completion_marker.touch()
        else:
            run_stage(f"Tracking video {position}/{len(videos)}", tracking_command)
            completion_marker.touch()
        manifests.append(manifest)

    matching_command = [
        sys.executable,
        SCRIPT_DIR / "match_video_identities.py",
        "--manifests",
        *manifests,
        "--assignments",
        assignments,
        "--output-dir",
        matching_dir,
        "--input-color",
        args.input_color,
        "--device",
        args.device,
        "--min-location-fraction",
        args.min_location_fraction,
        "--max-training-crops",
        args.max_training_crops,
        "--max-embedding-crops",
        args.max_embedding_crops,
    ]
    if args.init_checkpoint:
        matching_command.extend(("--init-checkpoint", args.init_checkpoint))
    if args.epochs is not None:
        matching_command.extend(("--epochs", args.epochs))
    if args.background_subtraction:
        matching_command.append("--background-subtraction")
    for center in args.center or []:
        matching_command.extend(("--center", *center))
    if args.exclude_edges:
        matching_command.append("--exclude-edges")
    if args.inclusion_size:
        matching_command.extend(("--inclusion-size", *args.inclusion_size))
    if args.force_recompute:
        matching_command.append("--force-recompute")
    run_stage("Resolving cross-video identities", matching_command)

    if args.visualize:
        visualization_dir = output_dir / "visualizations"
        for position, video in enumerate(videos, start=1):
            predictions = matching_dir / video.stem / "identity_predictions.csv"
            overlay = visualization_dir / f"{video.stem}_identity_overlay_10min.mp4"
            reusable = (
                overlay.is_file()
                and predictions.is_file()
                and overlay.stat().st_mtime >= predictions.stat().st_mtime
                and not args.force_recompute
            )
            if reusable:
                print(
                    f"\n=== Reusing visualization {position}/{len(videos)} ===\n"
                    f"{overlay}",
                    flush=True,
                )
                continue
            overlay_command = [
                sys.executable,
                SCRIPT_DIR / "identity_overlay.py",
                "--video",
                video,
                "--predictions",
                predictions,
                "--output",
                overlay,
                "-ss",
                0,
                "-t",
                600,
            ]
            run_stage(
                f"Rendering visualization {position}/{len(videos)}",
                overlay_command,
            )

    print(f"\nCross-video identity pipeline complete: {output_dir}", flush=True)
    print(f"Matched predictions and mapping: {matching_dir}", flush=True)


if __name__ == "__main__":
    main()
