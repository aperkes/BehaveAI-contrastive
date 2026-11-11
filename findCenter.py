#!/usr/bin/env python3
"""Find the arena center used by cropByCenter.py without cropping the video."""

import argparse
import re
from pathlib import Path

import cv2


PAD_TOP = 300
DEFAULT_FRAME = 3000
FIND_YOUR_CENTER_DIR = Path.home() / "Documents/Scripts/FindYourCenter"
WHITE_TEMPLATE = FIND_YOUR_CENTER_DIR / "templateWhiteSimple.png"
BLACK_TEMPLATE = FIND_YOUR_CENTER_DIR / "templateBlackSimple.png"
FILTER_TEMPLATE = FIND_YOUR_CENTER_DIR / "templateFilterSimple.png"
VIDEO_NAME = re.compile(r"^(?P<pi>pi\d+)\.(?P<year>\d{4})\.")


def default_template(video_path):
    """Select the same white/black center template as cropByCenter.py."""
    match = VIDEO_NAME.match(Path(video_path).name)
    if match and match.group("pi") in {"pi09", "pi11", "pi12"}:
        if match.group("year") == "2024":
            return BLACK_TEMPLATE
    return WHITE_TEMPLATE


def find_video_center(video_path, template_path=None, frame_number=DEFAULT_FRAME):
    """Return ``(x, y)`` in original-video coordinates plus the match score."""
    video_path = Path(video_path).expanduser().resolve()
    template_path = Path(template_path or default_template(video_path)).expanduser()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = min(frame_number, max(0, frame_count - 1))
    capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    success, image = capture.read()
    if not success and target_frame != 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, image = capture.read()
    capture.release()
    if not success or image is None:
        raise RuntimeError(f"Could not read a frame from video: {video_path}")

    template_rgba = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
    if template_rgba is None or template_rgba.ndim != 3 or template_rgba.shape[2] < 4:
        raise RuntimeError(f"Center template must be an RGBA image: {template_path}")

    padded = cv2.copyMakeBorder(
        image, PAD_TOP, PAD_TOP, 0, 0, cv2.BORDER_CONSTANT
    )
    search_image = cv2.bitwise_not(padded)
    template = cv2.bitwise_not(template_rgba[:, :, :3])
    alpha = template_rgba[:, :, 3]
    template_mask = cv2.merge((alpha, alpha, alpha))
    blurred = cv2.GaussianBlur(search_image, (11, 11), 0)
    blurred = cv2.GaussianBlur(blurred, (11, 11), 0)

    matched = cv2.matchTemplate(
        blurred, template, cv2.TM_CCORR_NORMED, mask=template_mask
    )
    _, score, _, top_left = cv2.minMaxLoc(matched)
    template_height, template_width = template.shape[:2]
    center_x = top_left[0] + template_width / 2
    center_y = top_left[1] + template_height / 2 - PAD_TOP
    return (center_x, center_y), float(score)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Find a video's arena center without cropping it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", help="input video")
    parser.add_argument("--template", help="override the automatically selected template")
    parser.add_argument(
        "--frame", type=int, default=DEFAULT_FRAME, help="frame used for template matching"
    )
    parser.add_argument(
        "--show-score", action="store_true", help="also print the template-match score"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.frame < 0:
        parser.error("--frame cannot be negative")
    center, score = find_video_center(args.video, args.template, args.frame)
    print(f"{center[0]:.1f},{center[1]:.1f}")
    if args.show_score:
        print(f"match_score={score:.6f}")


if __name__ == "__main__":
    main()
