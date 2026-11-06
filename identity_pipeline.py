#!/usr/bin/env python3
"""Run tracking, contrastive identity assignment, and video overlay in sequence."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Track a video, train and assign contrastive identities, then create an "
            "identity overlay video."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        help="BehaveAI project directory containing BehaveAI_settings.ini",
    )
    parser.add_argument(
        "--in_file",
        "--in-file",
        dest="in_file",
        required=True,
        help="input video",
    )
    parser.add_argument(
        "--num-individuals",
        required=True,
        type=int,
        help="known number of individuals in the video",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "pipeline output directory; by default, uses "
            "PROJECT/output/VIDEO_NAME_identity_pipeline"
        ),
    )
    parser.add_argument(
        "--input-color",
        choices=("grayscale", "rgb"),
        default="grayscale",
        help="contrastive network input color",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="contrastive training device: auto, cpu, cuda, cuda:0, or mps",
    )
    parser.add_argument(
        "--init-checkpoint",
        help="optional contrastive checkpoint used to initialize training",
    )
    parser.add_argument(
        "--background-subtraction",
        action="store_true",
        help="use median-background masking during contrastive training",
    )
    parser.add_argument(
        "-ss",
        "--overlay-start",
        type=float,
        default=0.0,
        help="identity overlay start time in seconds",
    )
    parser.add_argument(
        "-t",
        "--overlay-duration",
        type=float,
        help="identity overlay duration in seconds; omit to render to the end",
    )
    parser.add_argument(
        "--show-track-ids",
        action="store_true",
        help="show BehaveAI track IDs alongside predicted identities",
    )
    return parser


def run_stage(label, command):
    print(f"\n=== {label} ===", flush=True)
    print(shlex.join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True)


def main():
    args = build_parser().parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    video = Path(args.in_file).expanduser().resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = project_dir / "output" / f"{video.stem}_identity_pipeline"

    tracking_dir = output_dir / "tracking"
    contrastive_dir = output_dir / "contrastive"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    contrastive_dir.mkdir(parents=True, exist_ok=True)

    manifest = tracking_dir / f"{video.stem}_crop_manifest.csv"
    predictions = contrastive_dir / "identity_predictions.csv"
    overlay = output_dir / f"{video.stem}_identity_overlay.mp4"

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
    run_stage("1/3 Tracking and crop manifest", tracking_command)

    contrastive_command = [
        sys.executable,
        SCRIPT_DIR / "contrastive_identity.py",
        "--mode",
        "train-predict",
        "--manifest",
        manifest,
        "--output-dir",
        contrastive_dir,
        "--num-individuals",
        args.num_individuals,
        "--input-color",
        args.input_color,
        "--device",
        args.device,
    ]
    if args.init_checkpoint:
        contrastive_command.extend(("--init-checkpoint", args.init_checkpoint))
    if args.background_subtraction:
        contrastive_command.extend(("--background-subtraction", "median"))
    run_stage("2/3 Contrastive identity training and assignment", contrastive_command)

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
        args.overlay_start,
    ]
    if args.overlay_duration is not None:
        overlay_command.extend(("-t", args.overlay_duration))
    if args.show_track_ids:
        overlay_command.append("--show-track-ids")
    run_stage("3/3 Identity overlay video", overlay_command)

    print(f"\nPipeline complete: {output_dir}", flush=True)
    print(f"Identity overlay: {overlay}", flush=True)


if __name__ == "__main__":
    main()
