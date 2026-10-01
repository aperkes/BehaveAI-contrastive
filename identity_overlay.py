#!/usr/bin/env python3
"""Render BehaveAI identity predictions over their source video.

Example::

    python identity_overlay.py --video input.mp4 --predictions identity_predictions.csv \
        --output identities.mp4 -ss 300 -t 60

Times supplied to ``-ss`` and ``-t`` are in seconds. When ``-t`` is omitted,
rendering continues from the start time to the end of the video.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import cv2


COLORS = (
    (56, 56, 255),
    (255, 157, 151),
    (60, 220, 60),
    (255, 180, 0),
    (210, 80, 220),
    (0, 215, 255),
    (255, 120, 40),
    (80, 220, 220),
    (180, 120, 255),
    (120, 255, 160),
    (255, 200, 120),
    (170, 170, 255),
)


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def load_predictions(path, video):
    """Load predictions for one source video, indexed by one-based frame number."""
    video = Path(video).resolve()
    rows_by_frame = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "source_video",
            "frame",
            "track_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "predicted_identity",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Prediction CSV is missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            if Path(row["source_video"]).resolve() != video:
                continue
            rows_by_frame[int(row["frame"])].append(
                {
                    "track_id": row["track_id"],
                    "identity": int(row["predicted_identity"]),
                    "bbox": tuple(
                        int(float(row[name])) for name in ("x1", "y1", "x2", "y2")
                    ),
                }
            )
    if not rows_by_frame:
        raise ValueError(f"No predictions in {path} match source video {video}")
    return rows_by_frame


def draw_label(frame, detection, show_track_ids=False):
    x1, y1, x2, y2 = detection["bbox"]
    identity = detection["identity"]
    if identity < 0:
        color = (150, 150, 150)
        label = "ID ?"
    else:
        color = COLORS[identity % len(COLORS)]
        label = f"ID {identity}"
    if show_track_ids:
        label += f"  track {detection['track_id']}"

    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    thickness = max(2, round(min(width, height) / 700))
    font_scale = max(0.5, min(width, height) / 1400)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    text_x = x1
    text_y = max(text_height + baseline + 2, y1)
    cv2.rectangle(
        frame,
        (text_x, text_y - text_height - baseline - 4),
        (min(width - 1, text_x + text_width + 6), text_y + 2),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (text_x + 3, text_y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def render_overlay(args):
    if args.ss < 0:
        raise ValueError("-ss cannot be negative")
    if args.duration is not None and args.duration <= 0:
        raise ValueError("-t must be greater than zero")

    video_path = Path(args.video).resolve()
    output_path = Path(args.output)
    if output_path.resolve() == video_path:
        raise ValueError("Output path must differ from the source video")
    predictions = load_predictions(args.predictions, video_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0 or total_frames <= 0:
        capture.release()
        raise RuntimeError("Source video reports invalid dimensions, frame rate, or length")

    start_frame = min(total_frames, int(round(args.ss * fps)))
    if args.duration is None:
        end_frame = total_frames
    else:
        end_frame = min(total_frames, start_frame + int(round(args.duration * fps)))
    frames_to_render = end_frame - start_frame
    if frames_to_render <= 0:
        capture.release()
        raise ValueError("The selected interval contains no video frames")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*args.codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open output video {output_path} with codec {args.codec}"
        )

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    started = perf_counter()
    last_report = started
    completed = 0
    try:
        for zero_based_frame in range(start_frame, end_frame):
            ok, frame = capture.read()
            if not ok:
                print(f"Warning: decoding stopped at frame {zero_based_frame + 1}")
                break
            for detection in predictions.get(zero_based_frame + 1, ()):
                draw_label(frame, detection, args.show_track_ids)
            writer.write(frame)
            completed += 1

            now = perf_counter()
            if completed == 1 or completed == frames_to_render or now - last_report >= 10:
                elapsed = now - started
                rate = completed / elapsed if elapsed else 0
                eta = (frames_to_render - completed) / rate if rate else 0
                print(
                    f"Rendered {completed}/{frames_to_render} frames "
                    f"({completed / frames_to_render:.1%}); "
                    f"elapsed {format_duration(elapsed)}, ETA {format_duration(eta)}",
                    flush=True,
                )
                last_report = now
    finally:
        capture.release()
        writer.release()

    print(f"Saved identity overlay: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Overlay predicted identities and bounding boxes on a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="source video")
    parser.add_argument(
        "--predictions",
        required=True,
        help="identity_predictions.csv produced by contrastive_identity.py",
    )
    parser.add_argument("--output", required=True, help="output overlay video")
    parser.add_argument("-ss", type=float, default=0.0, help="start time in seconds")
    parser.add_argument(
        "-t",
        dest="duration",
        type=float,
        help="duration in seconds; omit to render through the end",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="four-character OpenCV output codec",
    )
    parser.add_argument(
        "--show-track-ids",
        action="store_true",
        help="include source track IDs alongside predicted identities",
    )
    return parser


def main():
    render_overlay(build_parser().parse_args())


if __name__ == "__main__":
    main()
