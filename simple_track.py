import cv2
import numpy as np
import csv
import os
import glob
import configparser
import time
import shutil
import tkinter as tk
from tkinter import messagebox
import subprocess
import time
import sys
import argparse
import box_geometry as bg
from kalman_tracker import KalmanTracker
from motion_processor import MotionProcessor


YOLO = None


def load_runtime_dependencies():
    """Import model libraries only after argparse has handled help/errors."""
    global YOLO
    if YOLO is None:
        from ultralytics import YOLO as ultralytics_yolo

        YOLO = ultralytics_yolo


# --- NCNN helper utilities -----------------------


def ncnn_dir_for_weights(weights_path):
    """Return the expected NCNN export directory for a given .pt path."""
    base, ext = os.path.splitext(weights_path)
    # Ultralytics export typically creates a folder named like "<base>_ncnn_model"
    return base + "_ncnn_model"

def ncnn_files_exist(ncnn_dir):
    """Return True if NCNN param+bin appear to exist in the export dir."""
    if not os.path.isdir(ncnn_dir):
        return False
    # Look for .param and .bin files (ncnn export creates *.param and *.bin)
    has_param = any(f.endswith(".param") for f in os.listdir(ncnn_dir))
    has_bin = any(f.endswith(".bin") for f in os.listdir(ncnn_dir))
    return has_param and has_bin

def ensure_ncnn_export(weights_path, task="detect", timeout=300):
    """
    Ensure an NCNN conversion exists for weights_path.
    Returns the ncnn_dir on success, None on failure (falls back to .pt).
    This will skip conversion if the ncnn folder already exists.
    """
    ncnn_dir = ncnn_dir_for_weights(weights_path)
    if ncnn_files_exist(ncnn_dir):
        return ncnn_dir

    try:
        print(f"Exporting {weights_path} -> NCNN (this may take a while)...")
        model = YOLO(weights_path, task=task)
        # Use Ultralytics export API. This creates the folder "<base>_ncnn_model".
        # Some installs can be slow; we try and catch errors below.
        model.export(format="ncnn")
        # Wait a short time for files to appear (export is synchronous in most versions).
        start = time.time()
        while time.time() - start < timeout:
            if ncnn_files_exist(ncnn_dir):
                print(f"NCNN export complete: {ncnn_dir}")
                return ncnn_dir
            time.sleep(0.5)
        # timed out
        print(f"NCNN export timeout for {weights_path}")
        return None
    except Exception as e:
        # Don't crash — export can fail on some systems; print useful debugging info and return None
        print(f"Warning: NCNN export failed for {weights_path}: {e}")
        return None

def load_model_with_ncnn_preference(weights_path, task="detect"):
    """
    Attempt to use NCNN if available (or convert it). If conversion or loading fails,
    fall back to the original PyTorch .pt path.
    Returns a YOLO model instance (which may wrap NCNN or .pt).
    """
    # If a .pt was not provided (maybe already a folder), just try loading directly
    if not weights_path.endswith(".pt"):
        try:
            return YOLO(weights_path, task=task)
        except Exception as e:
            print(f"Error loading model {weights_path}: {e}")
            raise

    ncnn_dir = ncnn_dir_for_weights(weights_path)
    # prefer existing NCNN dir if present
    if ncnn_files_exist(ncnn_dir):
        try:
            print(f"Loading NCNN model from {ncnn_dir}")
            return YOLO(ncnn_dir, task=task)
        except Exception as e:
            print(f"Failed to load NCNN model at {ncnn_dir}: {e} (falling back to .pt)")

    # Otherwise attempt conversion (one-time). If it fails, fall back to .pt.
    exported = ensure_ncnn_export(weights_path, task=task)
    if exported:
        try:
            return YOLO(exported, task=task)
        except Exception as e:
            print(f"Failed to load NCNN-exported model {exported}: {e} (falling back to .pt)")

    # Finally, fallback to direct .pt load
    print(f"Using original weights (PyTorch) at {weights_path}")
    return YOLO(weights_path, task=task)
# --------------------------------------------------------------------




def resolve_runtime_paths(project_dir=None, config_path=None, model_path=None):
    """Resolve explicit paths before loading any project configuration."""
    project_dir = os.path.abspath(project_dir) if project_dir else None
    if config_path:
        config_path = os.path.abspath(config_path)
    elif project_dir:
        config_path = os.path.join(project_dir, 'BehaveAI_settings.ini')
    else:
        config_path = os.path.abspath('BehaveAI_settings.ini')

    if model_path:
        model_path = os.path.abspath(model_path)
    elif project_dir:
        model_path = os.path.join(
            project_dir, 'model_primary_static', 'train', 'weights', 'best.pt'
        )
    return config_path, model_path


def load_configuration(config_path='BehaveAI_settings.ini', model_override=None):
    """Load tracker settings only after command-line parsing has succeeded."""
    global config, project_base_dir
    global primary_motion_classes, primary_motion_colors, primary_motion_hotkeys
    global secondary_motion_classes, secondary_motion_colors, secondary_motion_hotkeys
    global primary_static_classes, primary_static_colors, primary_static_hotkeys
    global secondary_static_classes, secondary_static_colors, secondary_static_hotkeys
    global hierarchical_mode, motion_cropped_base_dir, static_cropped_base_dir
    global primary_classes, primary_colors, primary_hotkeys
    global secondary_classes, secondary_colors, secondary_hotkeys, secondary_class_ids
    global primary_static_project_path, primary_static_model_path, primary_static_yaml_path
    global primary_motion_project_path, primary_motion_model_path, primary_motion_yaml_path
    global secondary_static_project_path, secondary_static_data_path, secondary_static_model_path
    global secondary_motion_project_path, secondary_motion_data_path, secondary_motion_model_path
    global ignore_secondary, dominant_source
    global primary_classifier, primary_epochs, secondary_classifier, secondary_epochs
    global scale_factor, expA, expB, lum_weight, strategy
    global sliding_median_window, sliding_median_signed_color
    global motion_history_frames, motion_frame_gap
    global chromatic_tail_only, rgb_multipliers, use_ncnn, oriented
    global disambiguate_head_tail, static_ornt_base_dir, motion_ornt_base_dir
    global primary_conf_thresh, secondary_conf_thresh, match_distance_ratio
    global delete_after_missed, centroid_merge_ratio, iou_thresh, merge_aware
    global line_thickness, font_size, frame_skip
    global process_noise_pos_ratio, process_noise_vel_ratio
    global measurement_noise_ratio, motion_threshold

    config_path = os.path.abspath(config_path)
    project_base_dir = os.path.dirname(config_path)
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config.read(config_path)

    try:
        primary_motion_classes = [name.strip() for name in config['DEFAULT']['primary_motion_classes'].split(',')]
        cols = [c.strip() for c in config['DEFAULT'].get('primary_motion_colors', '').split(';') if c.strip()]
        primary_motion_colors = [tuple(map(int, c.split(',')))[::-1] for c in cols]
        primary_motion_hotkeys = [key.strip() for key in config['DEFAULT']['primary_motion_hotkeys'].split(',')]

        secondary_motion_classes = [name.strip() for name in config['DEFAULT']['secondary_motion_classes'].split(',')]
        cols = [c.strip() for c in config['DEFAULT'].get('secondary_motion_colors', '').split(';') if c.strip()]
        secondary_motion_colors = [tuple(map(int, c.split(',')))[::-1] for c in cols]
        secondary_motion_hotkeys = [key.strip() for key in config['DEFAULT']['secondary_motion_hotkeys'].split(',')]

        primary_static_classes = [name.strip() for name in config['DEFAULT']['primary_static_classes'].split(',')]
        cols = [c.strip() for c in config['DEFAULT'].get('primary_static_colors', '').split(';') if c.strip()]
        primary_static_colors = [tuple(map(int, c.split(',')))[::-1] for c in cols]
        primary_static_hotkeys = [key.strip() for key in config['DEFAULT']['primary_static_hotkeys'].split(',')]

        secondary_static_classes = [name.strip() for name in config['DEFAULT']['secondary_static_classes'].split(',')]
        cols = [c.strip() for c in config['DEFAULT'].get('secondary_static_colors', '').split(';') if c.strip()]
        secondary_static_colors = [tuple(map(int, c.split(',')))[::-1] for c in cols]
        secondary_static_hotkeys = [key.strip() for key in config['DEFAULT']['secondary_static_hotkeys'].split(',')]

        if len(secondary_motion_classes) >= 2 or len(secondary_static_classes) >= 2:
            hierarchical_mode = True
            motion_cropped_base_dir = 'annot_motion_crop'
            static_cropped_base_dir = 'annot_static_crop'

            if len(secondary_motion_classes) == 1:
                secondary_motion_classes = []
                secondary_motion_colors = []
                secondary_motion_hotkeys = []

            if len(secondary_static_classes) == 1:
                secondary_static_classes = []
                secondary_static_colors = []
                secondary_static_hotkeys = []
        else:
            hierarchical_mode = False

        primary_classes = primary_static_classes + primary_motion_classes
        primary_colors = primary_static_colors + primary_motion_colors
        primary_hotkeys = primary_static_hotkeys + primary_motion_hotkeys
        secondary_classes = secondary_static_classes + secondary_motion_classes
        secondary_colors = secondary_static_colors + secondary_motion_colors
        secondary_hotkeys = secondary_static_hotkeys + secondary_motion_hotkeys

        primary_static_project_path = os.path.join(project_base_dir, 'model_primary_static')
        primary_static_model_path = os.path.join(primary_static_project_path, "train", "weights", "best.pt")
        primary_static_yaml_path = os.path.join(project_base_dir, 'static_annotations.yaml')
        primary_motion_project_path = os.path.join(project_base_dir, 'model_primary_motion')
        primary_motion_model_path = os.path.join(primary_motion_project_path, "train", "weights", "best.pt")
        primary_motion_yaml_path = os.path.join(project_base_dir, 'motion_annotations.yaml')

        ignore_secondary = [name.strip() for name in config['DEFAULT']['ignore_secondary'].split(',')]
        dominant_source = config['DEFAULT']['dominant_source'].lower()
        primary_classifier = config['DEFAULT'].get('primary_classifier', 'yolo11s.pt')
        primary_epochs = int(config['DEFAULT'].get('primary_epochs', '50'))
        secondary_classifier = config['DEFAULT'].get('secondary_classifier', 'yolo11s-cls.pt')
        secondary_epochs = int(config['DEFAULT'].get('secondary_epochs', '50'))

        if hierarchical_mode:
            secondary_static_project_path = os.path.join(project_base_dir, 'model_secondary_static')
            secondary_static_data_path = os.path.join(project_base_dir, 'annot_static_crop')
            secondary_static_model_path = os.path.join(secondary_static_project_path, "train", "weights", "best.pt")
            secondary_motion_project_path = os.path.join(project_base_dir, 'model_secondary_motion')
            secondary_motion_data_path = os.path.join(project_base_dir, 'annot_motion_crop')
            secondary_motion_model_path = os.path.join(secondary_motion_project_path, "train", "weights", "best.pt")

            secondary_class_ids = list(range(len(secondary_classes)))
            paired = list(zip(secondary_classes, secondary_colors, secondary_class_ids, secondary_hotkeys))
            paired_sorted = sorted(paired, key=lambda x: x[0].lower())
            secondary_classes, secondary_colors, secondary_class_ids, secondary_hotkeys = zip(*paired_sorted)
            secondary_classes = list(secondary_classes)
            secondary_colors = list(secondary_colors)
            secondary_class_ids = list(secondary_class_ids)
            secondary_hotkeys = list(secondary_hotkeys)

        scale_factor = float(config['DEFAULT'].get('scale_factor', '1.0'))
        expA = float(config['DEFAULT'].get('expA', '0.5'))
        expB = float(config['DEFAULT'].get('expB', '0.8'))
        lum_weight = float(config['DEFAULT'].get('lum_weight', '0.7'))
        strategy = config['DEFAULT'].get('strategy', 'exponential')
        sliding_median_window = int(config['DEFAULT'].get('sliding_median_window', '50'))
        sliding_median_signed_color = config['DEFAULT'].get(
            'sliding_median_signed_color', 'false'
        ).lower() == 'true'
        chromatic_tail_only = config['DEFAULT']['chromatic_tail_only'].lower()
        rgb_multipliers = [float(x) for x in config['DEFAULT']['rgb_multipliers'].split(',')]
        use_ncnn = config['DEFAULT']['use_ncnn'].lower()
        oriented = config['DEFAULT'].get('box_shape', 'boxes').lower() == 'oriented'
        disambiguate_head_tail = oriented and config['DEFAULT'].get(
            'disambiguate_head_tail', 'false'
        ).lower() == 'true'
        static_ornt_base_dir = os.path.join(project_base_dir, 'annot_static_ornt')
        motion_ornt_base_dir = os.path.join(project_base_dir, 'annot_motion_ornt')
        primary_conf_thresh = float(config['DEFAULT'].get('primary_conf_thresh', '0.5'))
        secondary_conf_thresh = float(config['DEFAULT'].get('secondary_conf_thresh', '0.5'))
        match_distance_ratio = float(config['DEFAULT'].get('match_distance_ratio', '2.5'))
        delete_after_missed = float(config['DEFAULT'].get('delete_after_missed', '15'))
        centroid_merge_ratio = float(config['DEFAULT'].get('centroid_merge_ratio', '0.7'))
        iou_thresh = float(config['DEFAULT'].get('iou_thresh', '0.95'))
        merge_aware = config['DEFAULT'].get('merge_aware', 'true').lower() == 'true'
        line_thickness = int(config['DEFAULT'].get('line_thickness', '1'))
        font_size = float(config['DEFAULT'].get('font_size', '0.5'))
        frame_skip = int(config['DEFAULT'].get('frame_skip', '0'))
        motion_history_frames = int(config['DEFAULT'].get('motion_history_frames', '3'))
        motion_frame_gap = int(
            config['DEFAULT'].get('motion_frame_gap', str(frame_skip + 1))
        )

        process_noise_pos_ratio = float(config['kalman'].get('process_noise_pos_ratio', '0.05'))
        process_noise_vel_ratio = float(config['kalman'].get('process_noise_vel_ratio', '0.1'))
        measurement_noise_ratio = float(config['kalman'].get('measurement_noise_ratio', '0.1'))
        motion_threshold = -1 * int(config['DEFAULT'].get('motion_threshold', '0'))
    except KeyError as exc:
        raise KeyError(f"Missing configuration parameter: {exc}") from exc

    if len(primary_motion_classes) != len(primary_motion_colors) or len(primary_motion_classes) != len(primary_motion_hotkeys):
        raise ValueError("Primary motion classes, colors and hotkeys must match in configuration.")
    if len(secondary_motion_classes) != len(secondary_motion_colors) or len(secondary_motion_classes) != len(secondary_motion_hotkeys):
        raise ValueError("Secondary motion classes, colors and hotkeys must match in configuration.")
    if len(primary_static_classes) != len(primary_static_colors) or len(primary_static_classes) != len(primary_static_hotkeys):
        raise ValueError("Primary static classes, colors and hotkeys must match in configuration.")
    if len(secondary_static_classes) != len(secondary_static_colors) or len(secondary_static_classes) != len(secondary_static_hotkeys):
        raise ValueError("Secondary static classes, colors and hotkeys must match in configuration.")
    if dominant_source not in {'motion', 'static', 'confidence'}:
        raise ValueError("dominant_source must be motion, static, or confidence")

    if model_override:
        static_enabled = primary_static_classes[0] != '0'
        motion_enabled = primary_motion_classes[0] != '0'
        if static_enabled and motion_enabled:
            raise ValueError(
                "--model is ambiguous when both primary static and motion models "
                "are enabled; use their project-default paths instead."
            )
        if not static_enabled and not motion_enabled:
            raise ValueError("--model was provided, but no primary model is enabled.")
        if static_enabled:
            primary_static_model_path = model_override
        else:
            primary_motion_model_path = model_override

# This command-line tracker is inference-only. Annotation datasets and settings
# snapshots are not required when trained weights already exist.
settings_changed = False
# Globals for prompting/behaviour inside maybe_retrain
regen_prompt_shown = False
force_rebuild_motion = False


global_response = 0 # if 'yes' is selected for any model re-training, retraining should be perfoemd for all models

def count_images_in_dataset(path):
    ## Count images in a dataset, handling both YAML-based and directory-based datasets
    # If path is a YAML file (primary models)
    if path.endswith('.yaml'):
        try:
            import yaml
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Get the path to the training images
            train_path = data['train']
            base_dir = os.path.dirname(path)
            abs_train_path = os.path.join(base_dir, train_path)
            
            # Handle different dataset formats
            if abs_train_path.endswith('.txt'):
                # Text file with image paths
                with open(abs_train_path, 'r') as f:
                    return len(f.readlines())
            else:
                # Directory with images
                image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                return len([f for f in os.listdir(abs_train_path) 
                            if os.path.splitext(f)[1].lower() in image_exts])
        except Exception as e:
            print(f"Error counting images: {e}")
            return 0
    
    # If path is a directory (secondary models)
    elif os.path.isdir(path):
        total_count = 0
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(path):
            # Only count files in leaf directories (class directories)
            if not dirs:  # This is a leaf directory (no subdirectories)
                count = sum(1 for f in files 
                           if os.path.splitext(f)[1].lower() in image_exts)
                total_count += count
                
        return total_count
    
    else:
        print(f"Unsupported dataset format: {path}")
        return 0


def maybe_retrain(model_type, yaml_path, project_path, model_path, classifier, epochs, imgsz):
    """Validate inference weights without requiring annotations or retraining."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained weights for {model_type} were not found: {model_path}"
        )
    return False

secondary_static_models = None
secondary_motion_models = None
ornt_static_models = None
ornt_motion_models = None


def initialize_models():
    """Validate and load configured models after argument and config parsing."""
    global secondary_static_models, secondary_motion_models
    global ornt_static_models, ornt_motion_models
    global static_class_map, motion_class_map

    if hierarchical_mode:
        secondary_static_models = {}
        static_class_map = [[None] * len(secondary_classes) for _ in range(len(primary_classes))]
        if len(secondary_static_classes) >= 2:
            for primary_class in primary_classes:
                idx = primary_classes.index(primary_class)
                hotkey = primary_hotkeys[idx]
                if hotkey in secondary_hotkeys or primary_class in ignore_secondary:
                    continue

                data_dir = os.path.join(secondary_static_data_path, primary_class)
                if not os.path.isdir(data_dir):
                    continue

                model_dir = os.path.join(
                    project_base_dir, f"model_static_static_{primary_class}"
                )
                weights_path = os.path.join(model_dir, "train", "weights", "best.pt")
                maybe_retrain(
                    model_dir,
                    data_dir,
                    model_dir,
                    weights_path,
                    secondary_classifier,
                    secondary_epochs,
                    224,
                )
                if use_ncnn == 'true':
                    secondary_static_models[primary_class] = load_model_with_ncnn_preference(
                        weights_path, "classify"
                    )
                else:
                    secondary_static_models[primary_class] = YOLO(weights_path)

        secondary_motion_models = {}
        motion_class_map = [[None] * len(secondary_classes) for _ in range(len(primary_classes))]
        if len(secondary_motion_classes) >= 2:
            for primary_class in primary_classes:
                idx = primary_classes.index(primary_class)
                hotkey = primary_hotkeys[idx]
                if hotkey in secondary_hotkeys or primary_class in ignore_secondary:
                    continue

                data_dir = os.path.join(secondary_motion_data_path, primary_class)
                if not os.path.isdir(data_dir):
                    continue

                model_dir = os.path.join(
                    project_base_dir, f"model_secondary_motion_{primary_class}"
                )
                weights_path = os.path.join(model_dir, "train", "weights", "best.pt")
                maybe_retrain(
                    model_dir,
                    data_dir,
                    model_dir,
                    weights_path,
                    secondary_classifier,
                    secondary_epochs,
                    224,
                )
                if use_ncnn == 'true':
                    secondary_motion_models[primary_class] = load_model_with_ncnn_preference(
                        weights_path, "classify"
                    )
                else:
                    secondary_motion_models[primary_class] = YOLO(weights_path)

    if disambiguate_head_tail:
        ornt_static_models = {}
        ornt_motion_models = {}
        for primary_class in primary_classes:
            if primary_static_classes[0] != '0':
                weights_path = os.path.join(
                    project_base_dir,
                    f"model_ornt_static_{primary_class}",
                    "train",
                    "weights",
                    "best.pt",
                )
                if os.path.isfile(weights_path):
                    ornt_static_models[primary_class] = (
                        load_model_with_ncnn_preference(weights_path, "classify")
                        if use_ncnn == 'true'
                        else YOLO(weights_path)
                    )
            if primary_motion_classes[0] != '0':
                weights_path = os.path.join(
                    project_base_dir,
                    f"model_ornt_motion_{primary_class}",
                    "train",
                    "weights",
                    "best.pt",
                )
                if os.path.isfile(weights_path):
                    ornt_motion_models[primary_class] = (
                        load_model_with_ncnn_preference(weights_path, "classify")
                        if use_ncnn == 'true'
                        else YOLO(weights_path)
                    )

    if primary_static_classes[0] != '0':
        maybe_retrain(
            'primary static',
            primary_static_yaml_path,
            primary_static_project_path,
            primary_static_model_path,
            primary_classifier,
            primary_epochs,
            640,
        )

    if primary_motion_classes[0] != '0':
        maybe_retrain(
            'primary motion',
            primary_motion_yaml_path,
            primary_motion_project_path,
            primary_motion_model_path,
            primary_classifier,
            primary_epochs,
            640,
        )


def _orient_conf(model, image):
    return _orient_confs(model, [image])[0]


def _orient_confs(model, images):
    """Return 'correct' confidence for a batch of orientation crops."""
    correct_index = next(
        (index for index, name in model.names.items() if name == 'correct'), None
    )
    if correct_index is None:
        return [0.0] * len(images)
    valid_images = [image for image in images if image is not None and image.size]
    if len(valid_images) != len(images):
        return [0.0] * len(images)
    results = model.predict(valid_images, verbose=False)
    return [
        0.0 if result.probs is None else float(result.probs.data[correct_index])
        for result in results
    ]


def _disambiguate_head_tail(image, head, tail, width, primary_class, is_motion):
    if not disambiguate_head_tail:
        return head, tail, width
    models = ornt_motion_models if is_motion else ornt_static_models
    model = models.get(primary_class) if models else None
    if model is None:
        return head, tail, width
    return bg.best_head_tail_orientation(
        image,
        head,
        tail,
        width,
        lambda crop: _orient_conf(model, crop),
        lambda crops: _orient_confs(model, crops),
    )

# --- PARAMETERS -----------------------------------------------------------

input_folder = "./input/"
output_folder = "./output/"

progress_update = 10 # print progress every n frames
# --- MAIN PROCESSING -----------------------------------------------------
def bbox_in_source_coordinates(coords, processed_size, source_size):
    """Map a detector bounding box back to the source video's pixel space."""
    x1, y1, x2, y2 = coords
    processed_width, processed_height = processed_size
    source_width, source_height = source_size

    x_scale = source_width / processed_width
    y_scale = source_height / processed_height
    return (
        max(0, min(source_width, int(np.floor(x1 * x_scale)))),
        max(0, min(source_height, int(np.floor(y1 * y_scale)))),
        max(0, min(source_width, int(np.ceil(x2 * x_scale)))),
        max(0, min(source_height, int(np.ceil(y2 * y_scale)))),
    )


def oriented_box_in_source_coordinates(xywhr, processed_size, source_size):
    """Map an oriented ``(cx, cy, width, height, radians)`` box to source pixels."""
    cx, cy, box_width, box_height, angle = xywhr
    processed_width, processed_height = processed_size
    source_width, source_height = source_size
    x_scale = source_width / processed_width
    y_scale = source_height / processed_height
    return (
        cx * x_scale,
        cy * y_scale,
        box_width * x_scale,
        box_height * y_scale,
        angle,
    )


def iter_primary_detections(result, class_names, image, is_motion=False):
    """Yield enclosing rectangles from standard or oriented YOLO results."""
    boxes = result.obb if oriented and result.obb is not None else result.boxes
    # Be tolerant of an OBB model paired with an older config that omitted
    # box_shape=oriented. Ultralytics leaves result.boxes as None in that case.
    if boxes is None and result.obb is not None:
        boxes = result.obb
    if boxes is None:
        return
    using_oriented_boxes = result.obb is not None and boxes is result.obb
    for box in boxes:
        class_idx = int(box.cls[0])
        prediction = {
            'coords': tuple(map(int, box.xyxy[0].tolist())),
            'primary_class': class_names[class_idx],
            'primary_conf': float(box.conf[0]),
        }
        if using_oriented_boxes:
            cx, cy, box_width, box_height, angle = (
                float(value) for value in box.xywhr[0].tolist()
            )
            head, tail, width = bg.head_tail_width_from_xywhr_rad(
                cx, cy, box_width, box_height, angle
            )
            head, tail, width = _disambiguate_head_tail(
                image, head, tail, width, prediction['primary_class'], is_motion
            )
            prediction.update(head=head, tail=tail, width=width)
        else:
            prediction.update(head=None, tail=None, width=None)
        yield prediction


def oriented_box_from_detection(det):
    """Return canonical (center, length, width, angle) geometry for a merged OBB."""
    head, tail = det.get('head'), det.get('tail')
    if head is None or tail is None:
        return None
    dx, dy = head[0] - tail[0], head[1] - tail[1]
    return (
        0.5 * (head[0] + tail[0]),
        0.5 * (head[1] + tail[1]),
        float(np.hypot(dx, dy)),
        float(det['width']),
        float(np.arctan2(dy, dx)),
    )


def draw_detection_overlay(frame, det, track_id):
    """Draw the same axis-aligned or oriented annotation used by the GUI."""
    source = det.get('source', '')
    primary_class = (
        det.get('primary_static_class', '')
        if source == 'static'
        else det.get('primary_motion_class', '')
    )
    primary_color = primary_colors[primary_classes.index(primary_class)]
    secondary_class = primary_class
    secondary_color = (255, 255, 255)
    for candidate in (
        det.get('secondary_motion_class', ''),
        det.get('secondary_static_class', ''),
    ):
        if candidate and candidate != primary_class:
            secondary_class = candidate
            secondary_color = secondary_colors[secondary_classes.index(candidate)]

    if oriented and det.get('head') is not None:
        corners = bg.corners_from_head_tail_width(
            det['head'], det['tail'], det['width']
        )
        if not hierarchical_mode or primary_class in ignore_secondary:
            label = f"{track_id} {primary_class.upper()}"
            bg.draw_oriented_box(
                frame,
                corners,
                det['head'],
                primary_color,
                thickness=line_thickness,
                label=label,
                label_color=primary_color,
            )
        else:
            outer_thickness = line_thickness + 2
            outer_corners = bg.inset_corners(corners, -outer_thickness)
            cv2.polylines(
                frame,
                [np.array(outer_corners, dtype=np.int32)],
                True,
                primary_color,
                outer_thickness,
                cv2.LINE_AA,
            )
            label = [
                (f"{track_id} {primary_class.upper()} ", primary_color),
                (secondary_class, secondary_color),
            ]
            bg.draw_oriented_box(
                frame,
                bg.inset_corners(corners, -1),
                det['head'],
                secondary_color,
                thickness=line_thickness + 1,
                label=label,
            )
        return primary_color

    x1, y1, x2, y2 = det['coords']
    if not hierarchical_mode or primary_class in ignore_secondary:
        label = f"{track_id} {primary_class.upper()}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), primary_color, line_thickness)
        bg.draw_label_with_background(
            frame,
            [(label, primary_color)],
            x1,
            y1,
            font_size,
            line_thickness,
        )
    else:
        outer_thickness = line_thickness + 2
        cv2.rectangle(
            frame,
            (x1 - outer_thickness, y1 - outer_thickness),
            (x2 + outer_thickness, y2 + outer_thickness),
            primary_color,
            outer_thickness,
        )
        cv2.rectangle(
            frame,
            (x1 - 1, y1 - 1),
            (x2 + 1, y2 + 1),
            secondary_color,
            line_thickness + 1,
        )
        label = [
            (f"{track_id} {primary_class.upper()} ", primary_color),
            (secondary_class, secondary_color),
        ]
        bg.draw_label_with_background(
            frame, label, x1, y1, font_size, line_thickness
        )
    return primary_color


def process_video(file, viz=False, export_crop_manifest=False):
    """Track a video and optionally export source-referenced crop metadata."""
    os.makedirs(output_folder, exist_ok=True)
    base = os.path.splitext(os.path.basename(file))[0]
    cap = cv2.VideoCapture(file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened(): return
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(source_width*scale_factor)
    h = int(source_height*scale_factor)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if viz:
        writer = cv2.VideoWriter(
            os.path.join(output_folder, base + "_detected.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
        )

    detect_task = "obb" if oriented else "detect"
    if primary_static_classes[0] != '0':
        if use_ncnn == 'true':
            model_static = load_model_with_ncnn_preference(
                primary_static_model_path, detect_task
            )
        else:
            model_static = YOLO(primary_static_model_path)
         
    if primary_motion_classes[0] != '0':
        if use_ncnn == 'true':
            model_motion = load_model_with_ncnn_preference(
                primary_motion_model_path, detect_task
            )
        else:
            model_motion = YOLO(primary_motion_model_path)
        
        
    tracker = KalmanTracker(
        match_distance_ratio,
        delete_after_missed,
        process_noise_pos_ratio,
        process_noise_vel_ratio,
        measurement_noise_ratio,
        merge_aware=merge_aware,
        frames_per_step=frame_skip + 1,
    )
    motion_processor = MotionProcessor(
        history_frames=motion_history_frames,
        frame_gap=motion_frame_gap,
        strategy=strategy,
        exp_a=expA,
        exp_b=expB,
        lum_weight=lum_weight,
        rgb_multipliers=rgb_multipliers,
        motion_threshold=motion_threshold,
        chromatic_tail_only=chromatic_tail_only == 'true',
        median_window=sliding_median_window,
        median_signed_color=sliding_median_signed_color,
    )

    frame_idx = 0
    csv_file = open(os.path.join(output_folder, base + "_tracking.csv"), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    # Updated CSV header with four streams
    csv_header = [
        "frame", "id", "x", "y",
        "primary_static_class", "primary_static_conf",
        "primary_motion_class", "primary_motion_conf",
        "secondary_static_class", "secondary_static_conf",
        "secondary_motion_class", "secondary_motion_conf"
    ]
    if oriented:
        csv_header += ["angle_deg", "width", "length"]
    csv_writer.writerow(csv_header)

    crop_manifest_file = None
    crop_manifest_writer = None
    if export_crop_manifest:
        crop_manifest_path = os.path.join(output_folder, base + "_crop_manifest.csv")
        crop_manifest_file = open(crop_manifest_path, 'w', newline='')
        crop_manifest_writer = csv.writer(crop_manifest_file)
        crop_manifest_writer.writerow([
            "source_video", "frame", "track_id", "x1", "y1", "x2", "y2",
            "obb_cx", "obb_cy", "obb_width", "obb_height", "obb_angle_radians",
            "detection_source",
            "primary_static_class", "primary_static_conf",
            "primary_motion_class", "primary_motion_conf",
            "secondary_static_class", "secondary_static_conf",
            "secondary_motion_class", "secondary_motion_conf"
        ])

    print(f"Processing video: {file}")
    print('Initialising')
    current_frame = 0
    current_fps = 0.0
    print_tick = 0
    start_time = time.time()

    frame_count = 0
    
    while True:
        ret, raw_frame = cap.read()
        if not ret: break
        frame_idx += 1
        if scale_factor != 1.0:
            raw_frame = cv2.resize(raw_frame, None, fx=scale_factor, fy=scale_factor)
        motion_candidate = (
            motion_processor.update(raw_frame)
            if primary_motion_classes[0] != '0'
            else None
        )
        if frame_count == 0:
            frame = raw_frame.copy()
            if primary_motion_classes[0] != '0' and motion_candidate is None:
                frame_count += 1
                if frame_count > frame_skip:
                    frame_count = 0
                continue
            motion_image = motion_candidate
    
            # Collect all primary detections
            all_detections = []
            
            # Primary static detection
            if primary_static_classes[0] != '0':
                results_static = model_static.predict(frame, conf=primary_conf_thresh, verbose=False)
                for prediction in iter_primary_detections(
                    results_static[0], primary_static_classes, frame, is_motion=False
                ):
                    all_detections.append({
                        **prediction,
                        'source': 'static',
                        'primary_class_combined': '',
                        'primary_conf_combined': 0.0
                    })
                    
            # Primary motion detection
            if primary_motion_classes[0] != '0':
                results_motion = model_motion.predict(motion_image, conf=primary_conf_thresh, verbose=False)
                for prediction in iter_primary_detections(
                    results_motion[0], primary_motion_classes, motion_image, is_motion=True
                ):
                    all_detections.append({
                        **prediction,
                        'source': 'motion',
                        'primary_class_combined': '',
                        'primary_conf_combined': 0.0
                    })
    
    
            # Use the same canonical, scale-relative merge implementation as
            # GUI batch processing.
            merged_detections = bg.merge_source_detections(
                all_detections,
                centroid_merge_ratio,
                iou_thresh,
                dominant_source,
            )

            # Run secondary classification on each primary detection
            processed_detections = []
            for det in merged_detections:
                coords = det['coords']
                primary_class = det['primary_class']
                primary_conf = det['primary_conf']
                source = det['source']
                primary_class_combined = det['primary_class_combined']
                primary_conf_combined = det['primary_conf_combined']
    
                
                if source == 'static':
                    det['primary_static_class'] = primary_class
                    det['primary_static_conf'] = primary_conf
                    det['primary_motion_class'] = primary_class_combined
                    det['primary_motion_conf'] = primary_conf_combined
                else:
                    det['primary_motion_class'] = primary_class
                    det['primary_motion_conf'] = primary_conf
                    det['primary_static_class'] = primary_class_combined
                    det['primary_static_conf'] = primary_conf_combined
    
                if hierarchical_mode:
                    x1, y1, x2, y2 = coords
                    
                    # Determine which secondary model to use based on source and configuration
                    sec_model = None
                    sec_classes = []
                    crop_img = None
                    
                    if source == 'static':
                        # Use static secondary model if configured
                        if len(secondary_static_classes) >= 2:
                            sec_model = secondary_static_models.get(primary_class, None)
                            sec_classes = secondary_static_classes
                            crop_img = frame
                        # Fallback to motion secondary model if static not available
                        elif len(secondary_motion_classes) >= 2:
                            sec_model = secondary_motion_models.get(primary_class, None)
                            sec_classes = secondary_motion_classes
                            crop_img = motion_image if primary_motion_classes[0] != '0' else frame
                    else:  # motion source
                        # Use motion secondary model if configured
                        if len(secondary_motion_classes) >= 2:
                            sec_model = secondary_motion_models.get(primary_class, None)
                            sec_classes = secondary_motion_classes
                            crop_img = motion_image
                        # Fallback to static secondary model if motion not available
                        elif len(secondary_static_classes) >= 2:
                            sec_model = secondary_static_models.get(primary_class, None)
                            sec_classes = secondary_static_classes
                            crop_img = frame
                    
                    # Get the cropped region
                    crop = None
                    if crop_img is not None:
                        if oriented and det.get('head') is not None:
                            corners = bg.corners_from_head_tail_width(
                                det['head'], det['tail'], det['width']
                            )
                            crop = bg.crop_black_masked(crop_img, corners)
                        else:
                            crop = crop_img[y1:y2, x1:x2]
                    
                    secondary_class = primary_class
                    secondary_conf = 1.0
                    
                    # Run secondary classification if we have a model and valid crop
                    if sec_model and crop is not None and crop.size > 0:
                        sec_results = sec_model.predict(crop, verbose=False)
                        if sec_results[0].probs is not None:
                            secondary_class_idx = sec_results[0].probs.top1
                            secondary_conf = sec_results[0].probs.top1conf.item()
                            secondary_class = sec_model.names[secondary_class_idx]
    
                    # Add secondary results to detection
                    if source == 'static':
                        det['secondary_static_class'] = secondary_class
                        det['secondary_static_conf'] = secondary_conf
                    else:  # motion source
                        det['secondary_motion_class'] = secondary_class
                        det['secondary_motion_conf'] = secondary_conf
                    
                processed_detections.append(det)
    
    
            # The GUI tracker gates and scales Kalman noise relative to each
            # detection's own body size.
            detections_for_tracker = [
                (d['centroid'][0], d['centroid'][1], bg.detection_scale(d))
                for d in processed_detections
            ]
            assignment = tracker.update(detections_for_tracker)
    
            # ~ frame = motion_image ## enable this line ot save the motion video instead of static 
    
            # Process tracked objects
            for idx, det in enumerate(processed_detections):
                tid = assignment.get(idx, None)
                if tid is None or tid not in tracker.tracks:
                    continue
                    
                x1, y1, x2, y2 = det['coords']
                cx, cy = det['centroid']
    
                # Get all class info with default values
                ps_class = det.get('primary_static_class', '')
                ps_conf = det.get('primary_static_conf', 0)
                pm_class = det.get('primary_motion_class', '')
                pm_conf = det.get('primary_motion_conf', 0)
                ss_class = det.get('secondary_static_class', '')
                ss_conf = det.get('secondary_static_conf', 0)
                sm_class = det.get('secondary_motion_class', '')
                sm_conf = det.get('secondary_motion_conf', 0)
                p_source = det.get('source', '')
    
    
                if viz:
                    primary_col = draw_detection_overlay(frame, det, tid)
                    state_post = tracker.tracks[tid]['kf'].statePost
                    x, y = state_post[0, 0], state_post[1, 0]
                    vx, vy = state_post[2, 0], state_post[3, 0]
                    next_x = x + vx
                    next_y = y + vy
                    
                    light_color = tuple(int(0.8 * ch + 0.2 * 255) for ch in primary_col)
                    cv2.line(frame, (int(x), int(y)), (int(next_x), int(next_y)), primary_col, line_thickness)
                    cv2.circle(frame, (int(next_x), int(next_y)), 3, light_color, -line_thickness)
                    cv2.circle(frame, (int(cx), int(cy)), 3, primary_col, -line_thickness)
                
                # Write the same tracking columns as GUI batch processing.
                csv_row = [
                    frame_idx, tid, cx, cy,
                    ps_class, f"{ps_conf:.3f}",
                    pm_class, f"{pm_conf:.3f}",
                    ss_class, f"{ss_conf:.3f}",
                    sm_class, f"{sm_conf:.3f}"
                ]
                if oriented:
                    if det.get('head') is not None:
                        angle_deg, length = bg.angle_length_from_head_tail(
                            det['head'], det['tail']
                        )
                        csv_row += [
                            f"{angle_deg:.2f}",
                            f"{det['width']:.2f}",
                            f"{length:.2f}",
                        ]
                    else:
                        csv_row += ['', '', '']
                csv_writer.writerow(csv_row)

                if crop_manifest_writer is not None:
                    source_bbox = bbox_in_source_coordinates(
                        (x1, y1, x2, y2),
                        (frame.shape[1], frame.shape[0]),
                        (source_width, source_height),
                    )
                    oriented_bbox = oriented_box_from_detection(det)
                    if oriented_bbox is not None:
                        source_oriented_bbox = oriented_box_in_source_coordinates(
                            oriented_bbox,
                            (frame.shape[1], frame.shape[0]),
                            (source_width, source_height),
                        )
                    else:
                        source_oriented_bbox = ('', '', '', '', '')
                    crop_manifest_writer.writerow([
                        os.path.abspath(file), frame_idx, tid, *source_bbox,
                        *source_oriented_bbox, p_source,
                        ps_class, f"{ps_conf:.3f}",
                        pm_class, f"{pm_conf:.3f}",
                        ss_class, f"{ss_conf:.3f}",
                        sm_class, f"{sm_conf:.3f}"
                    ])


            if viz:
                label = str(current_frame)
                label_size, _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness
                )
                label_w, label_h = label_size
                cv2.rectangle(
                    frame,
                    (0, 0),
                    (label_w + line_thickness * 4, label_h + line_thickness * 4),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (line_thickness * 2, label_h + line_thickness * 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (255, 255, 255),
                    line_thickness,
                )
                writer.write(frame)
            
            if print_tick > progress_update:
                elapsed = time.time() - start_time
                current_fps = current_frame / elapsed if elapsed > 0 else 0
                pc_done = 100 * (frame_skip+1) * current_frame / total_frames
                print(f"Progress: {pc_done:.2f}% | {current_fps:.1f} FPS", end='\r', flush=True)
                print_tick = 0
            current_frame += 1
            print_tick += 1

        frame_count += 1
        
        if frame_count > frame_skip:
            frame_count = 0

    cap.release()
    if viz:
        writer.release()
    csv_file.close()
    if crop_manifest_file is not None:
        crop_manifest_file.close()
    print(f"Done processing {base} | {current_fps:.1f} FPS")

### Added by Ammon 
def build_parse():
    parser = argparse.ArgumentParser(description='Required and additional inputs') 
    parser.add_argument('--in_file','-i',required=True,help='Path to input .mp4')
    parser.add_argument('--out_folder','-o',required=False,default='./output/',help='Path to output .csv, if blank, defaults to output directory')
    parser.add_argument('--make_vid','-v',required=False,action='store_true',help='option to create output video')
    parser.add_argument(
        '--project-dir',
        help=(
            'project directory containing BehaveAI_settings.ini and the default '
            'model_primary_static/train/weights/best.pt model'
        ),
    )
    parser.add_argument(
        '--config',
        help='settings INI path; overrides the config inferred from --project-dir',
    )
    parser.add_argument(
        '--model',
        help='primary model path; overrides the model inferred from --project-dir',
    )
    parser.add_argument(
        '--export-crop-manifest', '--export_crop_manifest',
        dest='export_crop_manifest',
        action='store_true',
        help=(
            'write a compact CSV containing source-frame bounding boxes and track IDs; '
            'crops can be reconstructed from the source video for contrastive training'
        ),
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = build_parse()
    load_runtime_dependencies()
    config_path, model_path = resolve_runtime_paths(
        project_dir=args.project_dir,
        config_path=args.config,
        model_path=args.model,
    )
    load_configuration(config_path, model_override=model_path)
    initialize_models()
    vid = args.in_file
    if output_folder is not None:
        output_folder = args.out_folder
    process_video(
        vid,
        viz=args.make_vid,
        export_crop_manifest=args.export_crop_manifest,
    )
