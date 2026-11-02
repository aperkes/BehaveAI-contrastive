import cv2
import numpy as np
import csv
import os
import glob
from scipy.optimize import linear_sum_assignment
import configparser
import time
import shutil
import tkinter as tk
from tkinter import messagebox
import subprocess
import time
import sys
import argparse


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

def ensure_ncnn_export(weights_path, timeout=300):
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
        model = YOLO(weights_path)
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

def load_model_with_ncnn_preference(weights_path):
    """
    Attempt to use NCNN if available (or convert it). If conversion or loading fails,
    fall back to the original PyTorch .pt path.
    Returns a YOLO model instance (which may wrap NCNN or .pt).
    """
    # If a .pt was not provided (maybe already a folder), just try loading directly
    if not weights_path.endswith(".pt"):
        try:
            return YOLO(weights_path)
        except Exception as e:
            print(f"Error loading model {weights_path}: {e}")
            raise

    ncnn_dir = ncnn_dir_for_weights(weights_path)
    # prefer existing NCNN dir if present
    if ncnn_files_exist(ncnn_dir):
        try:
            print(f"Loading NCNN model from {ncnn_dir}")
            return YOLO(ncnn_dir)
        except Exception as e:
            print(f"Failed to load NCNN model at {ncnn_dir}: {e} (falling back to .pt)")

    # Otherwise attempt conversion (one-time). If it fails, fall back to .pt.
    exported = ensure_ncnn_export(weights_path)
    if exported:
        try:
            return YOLO(exported)
        except Exception as e:
            print(f"Failed to load NCNN-exported model {exported}: {e} (falling back to .pt)")

    # Finally, fallback to direct .pt load
    print(f"Using original weights (PyTorch) at {weights_path}")
    return YOLO(weights_path)
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
    global scale_factor, expA, expB, expA2, expB2, lum_weight, strategy
    global chromatic_tail_only, rgb_multipliers, use_ncnn, oriented
    global primary_conf_thresh, secondary_conf_thresh, match_distance_thresh
    global delete_after_missed, centroid_merge_thresh, iou_thresh
    global line_thickness, font_size, frame_skip
    global process_noise_pos, process_noise_vel, measurement_noise, motion_threshold

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
        expA2 = 1 - expA
        expB2 = 1 - expB
        lum_weight = float(config['DEFAULT'].get('lum_weight', '0.7'))
        strategy = config['DEFAULT'].get('strategy', 'exponential')
        chromatic_tail_only = config['DEFAULT']['chromatic_tail_only'].lower()
        rgb_multipliers = [float(x) for x in config['DEFAULT']['rgb_multipliers'].split(',')]
        use_ncnn = config['DEFAULT']['use_ncnn'].lower()
        oriented = config['DEFAULT'].get('box_shape', 'boxes').lower() == 'oriented'
        primary_conf_thresh = float(config['DEFAULT'].get('primary_conf_thresh', '0.5'))
        secondary_conf_thresh = float(config['DEFAULT'].get('secondary_conf_thresh', '0.5'))
        match_distance_thresh = float(config['DEFAULT'].get('match_distance_thresh', '200'))
        delete_after_missed = float(config['DEFAULT'].get('delete_after_missed', '5'))
        centroid_merge_thresh = float(config['DEFAULT'].get('centroid_merge_thresh', '50'))
        iou_thresh = float(config['DEFAULT'].get('iou_thresh', '0.95'))
        line_thickness = int(config['DEFAULT'].get('line_thickness', '1'))
        font_size = float(config['DEFAULT'].get('font_size', '0.5'))
        frame_skip = int(config['DEFAULT'].get('frame_skip', '0'))

        process_noise_pos = float(config['kalman'].get('process_noise_pos', '0.01'))
        process_noise_vel = float(config['kalman'].get('process_noise_vel', '0.1'))
        measurement_noise = float(config['kalman'].get('measurement_noise', '0.1'))
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


def initialize_models():
    """Validate and load configured models after argument and config parsing."""
    global secondary_static_models, secondary_motion_models
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
                    secondary_static_models[primary_class] = load_model_with_ncnn_preference(weights_path)
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
                    secondary_motion_models[primary_class] = load_model_with_ncnn_preference(weights_path)
                else:
                    secondary_motion_models[primary_class] = YOLO(weights_path)

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


# --- PARAMETERS -----------------------------------------------------------

input_folder = "./input/"
output_folder = "./output/"

progress_update = 10 # print progress every n frames

def iou(box1, box2):
    xa = max(box1[0], box2[0]); ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2]); yb = min(box1[3], box2[3])
    inter = max(0, xb-xa) * max(0, yb-ya)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    prop1 = inter/area1 
    prop2 = inter/area2
    # return the larger proportional overlap - e.g. if one box is entirely inside another, this will return a 1.0, whereas the previous wouldn't
    if prop1 > prop2:
        return prop1 if prop1 > 0 else 0
    else:
        return prop2 if prop2 > 0 else 0
    # ~ union = area1 + area2 - inter
    # ~ return inter/union if union > 0 else 0
    

# --- TRACKER CLASS -------------------------------------------------------
class KalmanTracker:
    def __init__(self, dist_thresh, max_missed):
        self.next_id = 1
        self.tracks = {}  # tid -> {'kf': KalmanFilter, 'missed': int}
        self.prev_positions = {}  # Track previous positions
        self.dist_thresh = dist_thresh
        self.max_missed = max_missed

    def _create_kf(self, initial_pt):

        #Create a 4D state (x, y, vx, vy) Kalman Filter measuring (x, y).
        kf = cv2.KalmanFilter(4, 2)
        # State transition: x' = x + vx, y' = y + vy
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], dtype=np.float32)
        # Measurement: we only observe x, y
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], dtype=np.float32)
        # Tune these covariances to your scene

        kf.processNoiseCov = np.diag([process_noise_pos, process_noise_pos, process_noise_vel, process_noise_vel]).astype(np.float32)   
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        # Initialize state
        kf.statePre  = np.array([[initial_pt[0]],
                                 [initial_pt[1]],
                                 [0.],
                                 [0.]], dtype=np.float32)
        kf.statePost = kf.statePre.copy()
        return kf
        
        self._prune_duplicate_tracks()

    def predict_all(self):
        """
        Predict the next position for every track.
        Returns list of (tid, predicted_pt).
        """
        preds = []
        for tid, tr in self.tracks.items():
            pred = tr['kf'].predict()
            preds.append((tid, (float(pred[0, 0]), float(pred[1, 0]))))
        return preds

        
        
    def _prune_duplicate_tracks(self):
        """
        Merge any two tracks whose current posteriors are very close.
        Call this at the end of update().
        """
        tids = list(self.tracks.keys())
        posts = {}
        for tid in tids:
            sp = self.tracks[tid]['kf'].statePost
            posts[tid] = (float(sp[0,0]), float(sp[1,0]))
        to_drop = set()
        for i, t1 in enumerate(tids):
            x1, y1 = posts[t1]
            for t2 in tids[i+1:]:
                x2, y2 = posts[t2]
                if np.hypot(x1-x2, y1-y2) < self.dist_thresh * 0.5:
                    # mark the higher ID for deletion
                    to_drop.add(max(t1, t2))
        for tid in to_drop:
            del self.tracks[tid]

    def update(self, detections):

        #detections: list of (x, y) centroids
        #Returns a dict: detection_index -> track_id

        # 1) Predict all tracks forward one step
        preds = self.predict_all()  # list of (tid, (px, py))
        track_ids = [t[0] for t in preds]
        pred_pts   = [t[1] for t in preds]

        # 2) Build cost matrix = Euclidean distance
        if pred_pts and detections:
            cost = np.zeros((len(pred_pts), len(detections)), dtype=np.float32)
            for i, p in enumerate(pred_pts):
                for j, d in enumerate(detections):
                    cost[i, j] = np.hypot(p[0] - d[0], p[1] - d[1])
            row_idx, col_idx = linear_sum_assignment(cost)
        else:
            row_idx = np.array([], dtype=int)
            col_idx = np.array([], dtype=int)

        assigned_detects = {}
        matched_tracks = set()
        matched_dets   = set()

        # 3) Associate tracks ↔ detections
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < self.dist_thresh:
                tid = track_ids[r]
                matched_tracks.add(tid)
                matched_dets.add(c)
                assigned_detects[c] = tid

                # Get the measurement point
                dpt = detections[c]
                meas = np.array([[np.float32(dpt[0])], [np.float32(dpt[1])]])
                
                # Correct KF with the detection measurement
                self.tracks[tid]['kf'].correct(meas)
                self.tracks[tid]['missed'] = 0
                
                # Update previous position
                self.prev_positions[tid] = (dpt[0], dpt[1])

        # 4) Process unassigned detections
        for i, dpt in enumerate(detections):
            if i in matched_dets:
                continue
                
            # try to find an existing track under the threshold
            best_tid, best_dist = None, float('inf')
            for tid, (px, py) in preds:
                if tid in matched_tracks:
                    continue
                d = np.hypot(dpt[0]-px, dpt[1]-py)
                if d < best_dist:
                    best_dist, best_tid = d, tid

            if best_dist < self.dist_thresh:
                assigned_detects[i] = best_tid
                self.tracks[best_tid]['missed'] = 0
                meas = np.array([[np.float32(dpt[0])], [np.float32(dpt[1])]])
                self.tracks[best_tid]['kf'].correct(meas)
                
                # Update previous position
                self.prev_positions[best_tid] = (dpt[0], dpt[1])
                matched_tracks.add(best_tid)  # Add to matched tracks

            else:
                # New track
                tid = self.next_id
                kf = self._create_kf(dpt)
                self.tracks[tid] = {'kf': kf, 'missed': 0}
                assigned_detects[i] = tid
                self.prev_positions[tid] = (dpt[0], dpt[1])  # Initialize position
                matched_tracks.add(tid)  # Add to matched tracks
                self.next_id += 1

        # 5) Handle unmatched tracks
        for tid in list(self.tracks.keys()):
            if tid not in matched_tracks:
                self.tracks[tid]['missed'] += 1
                # Increase uncertainty when missing detections
                noise_scale = min(2.0, 1.0 + self.tracks[tid]['missed'] * 0.2)
                
                # FIXED: Preserve matrix type and structure
                kf = self.tracks[tid]['kf']
                new_noise = kf.processNoiseCov.copy()
                new_noise *= noise_scale
                kf.processNoiseCov = new_noise
                    
                # Remove track if missed too many times
                if self.tracks[tid]['missed'] > self.max_missed:
                    del self.tracks[tid]
                    if tid in self.prev_positions:
                        del self.prev_positions[tid]

        return assigned_detects


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


def iter_primary_detections(result, class_names):
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
            prediction['oriented_bbox'] = tuple(
                float(value) for value in box.xywhr[0].tolist()
            )
        else:
            prediction['oriented_bbox'] = None
        yield prediction


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

    if primary_static_classes[0] != '0':
        if use_ncnn == 'true':
            model_static = load_model_with_ncnn_preference(primary_static_model_path)
        else:
            model_static = YOLO(primary_static_model_path)
         
    if primary_motion_classes[0] != '0':
        if use_ncnn == 'true':
            model_motion = load_model_with_ncnn_preference(primary_motion_model_path)
        else:
            model_motion = YOLO(primary_motion_model_path)
        
        
    tracker = KalmanTracker(match_distance_thresh, delete_after_missed)

    prev_frames, frame_idx = None, 0
    csv_file = open(os.path.join(output_folder, base + "_tracking.csv"), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    # Updated CSV header with four streams
    csv_writer.writerow([
        "frame", "id", "x", "y",
        "primary_static_class", "primary_static_conf",
        "primary_motion_class", "primary_motion_conf",
        "secondary_static_class", "secondary_static_conf",
        "secondary_motion_class", "secondary_motion_conf"
    ])

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
    print_tick = 0
    start_time = time.time()

    frame_count = 0
    
    while True:
        ret, raw_frame = cap.read()
        if not ret: break
        frame_idx += 1
        if frame_count == 0:
            if scale_factor != 1.0:
                raw_frame = cv2.resize(raw_frame, None, fx=scale_factor, fy=scale_factor)
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            frame = raw_frame.copy()
            if prev_frames is None:
                prev_frames = [gray.copy() for _ in range(3)]
                continue
            
            # only process motion information if necessary
            if primary_motion_classes[0] != '0':
    
                diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]
                
                if strategy == 'exponential':
                    prev_frames[0] = gray
                    prev_frames[1] = cv2.addWeighted(prev_frames[1], expA, gray, expA2, 0)
                    prev_frames[2] = cv2.addWeighted(prev_frames[2], expB, gray, expB2, 0)
                elif strategy == 'sequential':
                    prev_frames[2] = prev_frames[1]
                    prev_frames[1] = prev_frames[0]
                    prev_frames[0] = gray


                if chromatic_tail_only == 'true':
                    tb = cv2.subtract(diffs[0], diffs[1])   
                    tr = cv2.subtract(diffs[2], diffs[1])
                    tg = cv2.subtract(diffs[1], diffs[0])
                            
                    blue = cv2.addWeighted(gray, lum_weight, tb, rgb_multipliers[2], motion_threshold)
                    green = cv2.addWeighted(gray, lum_weight, tg, rgb_multipliers[1], motion_threshold)
                    red = cv2.addWeighted(gray, lum_weight, tr, rgb_multipliers[0], motion_threshold)
                else:
                    blue = cv2.addWeighted(gray, lum_weight, diffs[0], rgb_multipliers[2], motion_threshold)
                    green = cv2.addWeighted(gray, lum_weight, diffs[1], rgb_multipliers[1], motion_threshold)
                    red = cv2.addWeighted(gray, lum_weight, diffs[2], rgb_multipliers[0], motion_threshold)

                motion_image = cv2.merge((blue, green, red)).astype(np.uint8)
    
            # Collect all primary detections
            all_detections = []
            
            # Primary static detection
            if primary_static_classes[0] != '0':
                results_static = model_static.predict(frame, conf=primary_conf_thresh, verbose=False)
                for prediction in iter_primary_detections(
                    results_static[0], primary_static_classes
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
                    results_motion[0], primary_motion_classes
                ):
                    all_detections.append({
                        **prediction,
                        'source': 'motion',
                        'primary_class_combined': '',
                        'primary_conf_combined': 0.0
                    })
    
    
            # Merge detections based on proximity
            merged_detections = []
            for det in all_detections:
                x1, y1, x2, y2 = det['coords']
                cx, cy = (x1+x2)//2, (y1+y2)//2
                
                # Find matching existing detection
                matched = False
                for md in merged_detections:
                    md_cx, md_cy = md['centroid']
                    dist = np.hypot(cx - md_cx, cy - md_cy)
                    
                    # Calculate IOU
                    md_x1, md_y1, md_x2, md_y2 = md['coords']
                    overlap = iou((x1, y1, x2, y2), (md_x1, md_y1, md_x2, md_y2))
                    ms_source = md['source']
                    
                    if dist < centroid_merge_thresh or overlap > iou_thresh:
                        # Merge classes - keep highest confidence detection for each source
                        if det['source'] == ms_source or dominant_source == 'confidence': # mathcing sources so select best, or confidence strategy used
                            if det['source'] == 'static':
                                # Keep highest confidence static detection
                                if 'primary_conf' not in md or det['primary_conf'] > md['primary_conf']:
                                    md['primary_class_combined'] = md['primary_class'] # retain the combined primary
                                    md['primary_conf_combined'] = md['primary_conf']
                                    md['primary_class'] = det['primary_class']
                                    md['primary_conf'] = det['primary_conf']
                                    md['coords'] = det['coords']  # Update to higher conf box
                                    md['oriented_bbox'] = det.get('oriented_bbox')
                                    md['centroid'] = (cx, cy)
                                    md['source'] = det['source']
                            else:  # motion source
                                # Keep highest confidence motion detection
                                if 'primary_conf' not in md or det['primary_conf'] > md['primary_conf']:
                                    md['primary_class_combined'] = md['primary_class'] # retain the combined primary
                                    md['primary_conf_combined'] = md['primary_conf']
                                    md['primary_class'] = det['primary_class']
                                    md['primary_conf'] = det['primary_conf']
                                    md['coords'] = det['coords']  # Update to higher conf box
                                    md['oriented_bbox'] = det.get('oriented_bbox')
                                    md['centroid'] = (cx, cy)
                                    md['source'] = det['source']
                        elif det['source'] == 'static' and dominant_source == 'static':
                                # Keep static detection
                                md['primary_class_combined'] = md['primary_class'] # retain the combined primary
                                md['primary_conf_combined'] = md['primary_conf']
                                md['primary_class'] = det['primary_class']
                                md['primary_conf'] = det['primary_conf']
                                md['coords'] = det['coords']  # Update to higher conf box
                                md['oriented_bbox'] = det.get('oriented_bbox')
                                md['centroid'] = (cx, cy)
                                md['source'] = det['source']
                        elif det['source'] == 'motion' and dominant_source == 'motion':
                                # Keep motion detection
                                md['primary_class_combined'] = md['primary_class'] # retain the combined primary
                                md['primary_conf_combined'] = md['primary_conf']
                                md['primary_class'] = det['primary_class']
                                md['primary_conf'] = det['primary_conf']
                                md['coords'] = det['coords']  # Update to higher conf box
                                md['oriented_bbox'] = det.get('oriented_bbox')
                                md['centroid'] = (cx, cy)
                                md['source'] = det['source']
    
                            
                        matched = True
                        break
                        
                if not matched:
                    # Add as new detection
                    new_det = {
                        'coords': det['coords'],
                        'centroid': (cx, cy),
                        'source': det['source'],
                        'oriented_bbox': det.get('oriented_bbox'),
                        'primary_class_combined': '',
                        'primary_conf_combined': 0.0
                    }
                    if det['source'] == 'static':
                        new_det['primary_class'] = det['primary_class']
                        new_det['primary_conf'] = det['primary_conf']
                        # ~ if 'secondary_static_class' in det:
                            # ~ new_det['secondary_static_class'] = det['secondary_static_class']
                            # ~ new_det['secondary_static_conf'] = det['secondary_static_conf']
                    else:  # motion source
                        new_det['primary_class'] = det['primary_class']
                        new_det['primary_conf'] = det['primary_conf']
                        # ~ if 'secondary_motion_class' in det:
                            # ~ new_det['secondary_motion_class'] = det['secondary_motion_class']
                            # ~ new_det['secondary_motion_conf'] = det['secondary_motion_conf']
                    merged_detections.append(new_det)
    
    
    
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
    
    
            # Prepare for tracking
            cents = [d['centroid'] for d in processed_detections]
            assignment = tracker.update(cents)
    
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
    
    
                # Create display label
                label_parts = []
                # ~ if ps_class: 
                if p_source == 'static': 
                    label_parts.append(f"{ps_class.upper()}")
                    primary_cls = ps_class
                else:
                    label_parts.append(f"{pm_class.upper()}")
                    primary_cls = pm_class
                
                primary_col = primary_colors[primary_classes.index(primary_cls)]
                secondary_col = (255, 255, 255)
    
                if hierarchical_mode:
                    
                    if sm_class != '' and sm_class != primary_cls:
                        secondary_cls = sm_class
                        secondary_col = secondary_colors[secondary_classes.index(secondary_cls)]
                    if ss_class != '' and ss_class != primary_cls:
                        secondary_cls = ss_class
                        secondary_col = secondary_colors[secondary_classes.index(secondary_cls)]
                    
    
                    if primary_cls in ignore_secondary:
                        label = f"{tid} {primary_cls.upper()}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
                        label_w, label_h = label_size
                        cv2.rectangle(frame, (x1-line_thickness, y1 - label_h - line_thickness*4), (x1 + label_w + line_thickness*2, y1), (0, 0, 0), -1)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), primary_col, line_thickness)
                        cv2.putText(frame, label, (x1, y1 - line_thickness*2), cv2.FONT_HERSHEY_SIMPLEX, 
                                    font_size, primary_col, line_thickness, cv2.LINE_AA)
                    else:
                        # Draw outer static box (slightly larger)
                        outer_thickness = line_thickness + 2
                        cv2.rectangle(frame, (x1-outer_thickness, y1-outer_thickness), 
                                     (x2+outer_thickness, y2+outer_thickness), 
                                    primary_col, outer_thickness)
                        label = f"{tid} {primary_cls.upper()} {secondary_cls}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
                        label_w, label_h = label_size
                        cv2.rectangle(frame, (x1-line_thickness, y1 - label_h - line_thickness*4), (x1 + label_w + line_thickness*2, y1), (0, 0, 0), -1)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), secondary_col, line_thickness)
                        cv2.putText(frame, label, (x1, y1 - line_thickness*2), cv2.FONT_HERSHEY_SIMPLEX, 
                                    font_size, secondary_col, line_thickness, cv2.LINE_AA)
                else:
                    label = f"{tid} {primary_cls}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
                    label_w, label_h = label_size
                    cv2.rectangle(frame, (x1-line_thickness, y1 - label_h - line_thickness*4), (x1 + label_w + line_thickness*2, y1), (0, 0, 0), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), primary_col, line_thickness)
                    cv2.putText(frame, label, (x1, y1 - line_thickness*3), cv2.FONT_HERSHEY_SIMPLEX, 
                                font_size, primary_col, line_thickness, cv2.LINE_AA)
    
    
    
        
                # Draw motion vector (if tracking available)
                if tid in tracker.tracks:
                    state_post = tracker.tracks[tid]['kf'].statePost
                    x, y = state_post[0, 0], state_post[1, 0]
                    vx, vy = state_post[2, 0], state_post[3, 0]
                    next_x = x + vx
                    next_y = y + vy
                    
                    light_color = tuple(int(0.8 * ch + 0.2 * 255) for ch in primary_col)
                    cv2.line(frame, (int(x), int(y)), (int(next_x), int(next_y)), primary_col, line_thickness)
                    cv2.circle(frame, (int(next_x), int(next_y)), 3, light_color, -line_thickness)
                    cv2.circle(frame, (int(cx), int(cy)), 3, primary_col, -line_thickness)
                
                # Write to CSV
                csv_writer.writerow([
                    frame_idx, tid, cx, cy,
                    ps_class, f"{ps_conf:.3f}",
                    pm_class, f"{pm_conf:.3f}",
                    ss_class, f"{ss_conf:.3f}",
                    sm_class, f"{sm_conf:.3f}"
                ])

                if crop_manifest_writer is not None:
                    source_bbox = bbox_in_source_coordinates(
                        (x1, y1, x2, y2),
                        (frame.shape[1], frame.shape[0]),
                        (source_width, source_height),
                    )
                    oriented_bbox = det.get('oriented_bbox')
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


            # ~ # print frame number
            text_color = (255, 255, 255)  # white text
            label = str(current_frame)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
            label_w, label_h = label_size
            cv2.rectangle(frame, (0, 0), 
                         (label_w + line_thickness*4, label_h + line_thickness*4), (0, 0, 0), -1)
            cv2.putText(frame, label, (line_thickness*2, label_h + line_thickness*2), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, line_thickness)
                           
            if viz: ## obviously would be more optimal to comment out rectangles above
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
