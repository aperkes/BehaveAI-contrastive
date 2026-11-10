#!/usr/bin/env python3
"""Transfer a contrastive identity model from one video to another.

The source checkpoint is first used directly on the target video. If the
resulting assignment quality is poor, the same checkpoint initializes a short
training run on the target before identities are assigned again.
"""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Track a target video and transfer a contrastive identity model from "
            "a source video, fine-tuning only when direct inference is poor."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-dir", required=True, help="BehaveAI project")
    parser.add_argument(
        "--in_file",
        "--in-file",
        dest="in_file",
        required=True,
        help="target video receiving identity assignments",
    )
    parser.add_argument(
        "--source-video",
        required=True,
        help="previous video whose contrastive model should be transferred",
    )
    parser.add_argument("--num-individuals", required=True, type=int)
    parser.add_argument(
        "--checkpoint",
        "--model",
        dest="checkpoint",
        help="explicit source contrastive_identity.pt; overrides auto-discovery",
    )
    parser.add_argument(
        "--target-manifest",
        help="existing target crop manifest; skips target tracking",
    )
    parser.add_argument(
        "--source-manifest",
        help="existing source crop manifest, used only if the source model is absent",
    )
    parser.add_argument(
        "--output-dir",
        help="target output root (default: PROJECT/output/VIDEO_identity_pipeline)",
    )
    parser.add_argument(
        "--source-output-dir",
        help="source output root used for model discovery or source training",
    )
    parser.add_argument(
        "--input-color", choices=("grayscale", "rgb"), default="grayscale"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--epochs",
        type=int,
        help="maximum epochs when source training or target fine-tuning is required",
    )
    parser.add_argument(
        "--background-subtraction",
        action="store_true",
        help="use median-background masking in contrastive inference and training",
    )
    parser.add_argument(
        "--min-assigned-fraction",
        type=float,
        default=0.90,
        help="minimum assigned-detection fraction accepted without fine-tuning",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.80,
        help="minimum mean assignment confidence accepted without fine-tuning",
    )
    parser.add_argument(
        "--max-collision-fraction",
        type=float,
        default=0.01,
        help="maximum identity-collision frame fraction accepted",
    )
    return parser


def run_stage(label, command):
    command = [str(part) for part in command]
    print(f"\n=== {label} ===", flush=True)
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def standard_output_dir(project_dir, video):
    return project_dir / "output" / f"{video.stem}_identity_pipeline"


def track_video(project_dir, video, tracking_dir):
    tracking_dir.mkdir(parents=True, exist_ok=True)
    manifest = tracking_dir / f"{video.stem}_crop_manifest.csv"
    command = [
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
    run_stage(f"Tracking {video.name}", command)
    if not manifest.is_file():
        raise FileNotFoundError(f"Tracking did not create the expected manifest: {manifest}")
    return manifest


def contrastive_command(args, mode, manifest, output_dir, checkpoint_option=None):
    command = [
        sys.executable,
        SCRIPT_DIR / "contrastive_identity.py",
        "--mode",
        mode,
        "--manifest",
        manifest,
        "--output-dir",
        output_dir,
        "--num-individuals",
        args.num_individuals,
        "--input-color",
        args.input_color,
        "--device",
        args.device,
    ]
    if checkpoint_option:
        option, checkpoint = checkpoint_option
        command.extend((option, checkpoint))
    if args.epochs is not None and mode in {"train", "train-predict"}:
        command.extend(("--epochs", args.epochs))
    if args.background_subtraction:
        command.extend(("--background-subtraction", "median"))
    return command


def find_or_train_source_model(args, project_dir, source_video, source_output):
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Source checkpoint not found: {checkpoint}")
        print(f"Using explicitly supplied source checkpoint: {checkpoint}")
        return checkpoint

    checkpoint = source_output / "contrastive" / "contrastive_identity.pt"
    if checkpoint.is_file():
        print(f"Using existing source checkpoint: {checkpoint}")
        return checkpoint

    print(f"No existing source checkpoint found; training one from {source_video.name}.")
    if args.source_manifest:
        source_manifest = Path(args.source_manifest).expanduser().resolve()
        if not source_manifest.is_file():
            raise FileNotFoundError(f"Source manifest not found: {source_manifest}")
    else:
        expected = source_output / "tracking" / f"{source_video.stem}_crop_manifest.csv"
        source_manifest = (
            expected
            if expected.is_file()
            else track_video(project_dir, source_video, source_output / "tracking")
        )

    source_contrastive = source_output / "contrastive"
    source_contrastive.mkdir(parents=True, exist_ok=True)
    run_stage(
        "Training source contrastive model",
        contrastive_command(
            args, "train-predict", source_manifest, source_contrastive
        ),
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Source training did not create a checkpoint: {checkpoint}")
    return checkpoint


def read_quality(output_dir):
    path = output_dir / "identity_assignment_quality.json"
    if not path.is_file():
        raise FileNotFoundError(f"Contrastive inference did not create: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def quality_passes(args, quality):
    values = {
        "assigned_detection_fraction": float(
            quality.get("assigned_detection_fraction", 0.0)
        ),
        "mean_assigned_confidence": float(
            quality.get("mean_assigned_confidence", 0.0)
        ),
        "identity_collision_frame_fraction": float(
            quality.get("identity_collision_frame_fraction", 1.0)
        ),
    }
    checks = {
        "assigned_detection_fraction": (
            values["assigned_detection_fraction"] >= args.min_assigned_fraction
        ),
        "mean_assigned_confidence": (
            values["mean_assigned_confidence"] >= args.min_confidence
        ),
        "identity_collision_frame_fraction": (
            values["identity_collision_frame_fraction"]
            <= args.max_collision_fraction
        ),
    }
    print("\nTransfer quality:")
    print(
        f"  assigned detections: {values['assigned_detection_fraction']:.1%} "
        f"(required {args.min_assigned_fraction:.1%})"
    )
    print(
        f"  mean confidence: {values['mean_assigned_confidence']:.1%} "
        f"(required {args.min_confidence:.1%})"
    )
    print(
        f"  collision frames: {values['identity_collision_frame_fraction']:.1%} "
        f"(maximum {args.max_collision_fraction:.1%})"
    )
    return all(checks.values()), values, checks


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.num_individuals < 2:
        parser.error("--num-individuals must be at least 2")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1")
    for option in (
        "min_assigned_fraction",
        "min_confidence",
        "max_collision_fraction",
    ):
        if not 0 <= getattr(args, option) <= 1:
            parser.error(f"--{option.replace('_', '-')} must be between 0 and 1")

    project_dir = Path(args.project_dir).expanduser().resolve()
    target_video = Path(args.in_file).expanduser().resolve()
    source_video = Path(args.source_video).expanduser().resolve()
    if not target_video.is_file():
        raise FileNotFoundError(f"Target video not found: {target_video}")
    if not source_video.is_file():
        raise FileNotFoundError(f"Source video not found: {source_video}")

    target_output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else standard_output_dir(project_dir, target_video)
    )
    source_output = (
        Path(args.source_output_dir).expanduser().resolve()
        if args.source_output_dir
        else standard_output_dir(project_dir, source_video)
    )
    target_contrastive = target_output / "contrastive"
    target_contrastive.mkdir(parents=True, exist_ok=True)

    if args.target_manifest:
        target_manifest = Path(args.target_manifest).expanduser().resolve()
        if not target_manifest.is_file():
            raise FileNotFoundError(f"Target manifest not found: {target_manifest}")
    else:
        target_manifest = track_video(
            project_dir, target_video, target_output / "tracking"
        )

    source_checkpoint = find_or_train_source_model(
        args, project_dir, source_video, source_output
    )

    run_stage(
        "Trying direct identity transfer",
        contrastive_command(
            args,
            "predict",
            target_manifest,
            target_contrastive,
            ("--checkpoint", source_checkpoint),
        ),
    )
    initial_quality = read_quality(target_contrastive)
    passed, values, checks = quality_passes(args, initial_quality)
    initial_quality_path = target_output / "initial_transfer_quality.json"
    initial_quality_path.write_text(
        json.dumps(initial_quality, indent=2), encoding="utf-8"
    )

    action = "direct_inference"
    if passed:
        destination = target_contrastive / "contrastive_identity.pt"
        if source_checkpoint != destination:
            shutil.copy2(source_checkpoint, destination)
        print("Transfer quality passed; no target training is needed.")
    else:
        action = "fine_tuned"
        failed = ", ".join(name for name, result in checks.items() if not result)
        print(f"Transfer quality failed ({failed}); fine-tuning on the target video.")
        run_stage(
            "Fine-tuning transferred model on target video",
            contrastive_command(
                args,
                "train-predict",
                target_manifest,
                target_contrastive,
                ("--init-checkpoint", source_checkpoint),
            ),
        )

    final_quality = read_quality(target_contrastive)
    summary = {
        "source_video": str(source_video),
        "target_video": str(target_video),
        "source_checkpoint": str(source_checkpoint),
        "action": action,
        "initial_quality": values,
        "thresholds": {
            "min_assigned_fraction": args.min_assigned_fraction,
            "min_confidence": args.min_confidence,
            "max_collision_fraction": args.max_collision_fraction,
        },
        "final_quality": final_quality,
    }
    summary_path = target_output / "identity_transfer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nIdentity transfer complete: {target_output}")
    print(f"Predictions: {target_contrastive / 'identity_predictions.csv'}")
    print(f"Transfer summary: {summary_path}")


if __name__ == "__main__":
    main()
