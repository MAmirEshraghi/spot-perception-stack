#!/usr/bin/env python3
"""
Detection Processing Pipeline

Processing script for detection tasks:
  - vocab        : generate vocabulary JSON per entry using the VLM
  - yolo         : run YOLO-World detection using vocab JSON
  - rex_detection: run Rex-Omni detection using vocab JSON
  - rex_pointing : run Rex-Omni pointing using vocab JSON

Each task outputs a unified JSON file with metadata, entries, and summary.

Running Commands:

1) Vocabulary extraction:
   python rex_play.py --task vocab

2) YOLO detection:
   python rex_play.py --task yolo 
   (optional) --vocab-json output/vocabulary_results.json

3) Rex-Omni detection:
   python rex_play.py --task rex_detection 
   (optional)--vocab-json output/vocabulary_results.json

4) Rex-Omni pointing:
   python rex_play.py --task rex_pointing 
   (optional)--vocab-json output/vocabulary_results.json

Optional arguments:
   --pickle-path PATH          : Path to pickle file (default: data/obs_buffer_capture.pkl)
   --output-dir DIR            : Base output directory (default: output)
   --vocab-json PATH           : Vocabulary JSON path (default: output/vocabulary_results.json)
   --yolo-json PATH            : YOLO results JSON path (default: output/yolo_results.json)
   --rex-detection-json PATH  : Rex-Omni detection JSON path (default: output/rex_omni_detection_results.json)
   --rex-pointing-json PATH   : Rex-Omni pointing JSON path (default: output/rex_omni_pointing_results.json)

Robin Eshraghi 
update: 12/05/2025
"""

import os
from pathlib import Path

# Check if standard temp directories exist and are writable
temp_dirs = ['/tmp', '/var/tmp', '/usr/tmp']
temp_dir_found = False

for temp_dir in temp_dirs:
    if os.path.exists(temp_dir) and os.access(temp_dir, os.W_OK):
        temp_dir_found = True
        break

if not temp_dir_found:
    user_tmp = Path.home() / '.tmp'
    user_tmp.mkdir(exist_ok=True, mode=0o700)
    os.environ['TMPDIR'] = str(user_tmp)
    os.environ['TMP'] = str(user_tmp)
    os.environ['TEMP'] = str(user_tmp)

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import argparse
import json
import logging
import time
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import torch

from tiamat_agent.vision_grounding.obs_data_buffer import ObsDataEntry
from tiamat_agent.vision_grounding.intern_vlm_interface import create_vlm_detector
from ultralytics import YOLOWorld
import supervision as sv

from rex_omni import RexOmniWrapper

# ============================================================================
# Configuration
# ============================================================================

INPUT_PICKLE_PATH = SCRIPT_DIR / "data" / "obs_buffer_capture.pkl"
OUTPUT_BASE_DIR = SCRIPT_DIR / "output"
VLM_MODEL_NAME = "OpenGVLab/InternVL3_5-1B"
YOLO_MODEL_NAME = "yolov8x-worldv2.pt"
REX_OMNI_MODEL_PATH = "IDEA-Research/Rex-Omni"

DEFAULT_PICKLE_PATH = INPUT_PICKLE_PATH
DEFAULT_OUTPUT_DIR = OUTPUT_BASE_DIR
DEFAULT_VOCAB_JSON = DEFAULT_OUTPUT_DIR / "vocabulary_results.json"
DEFAULT_YOLO_JSON = DEFAULT_OUTPUT_DIR / "yolo_results.json"
DEFAULT_REX_DETECTION_JSON = DEFAULT_OUTPUT_DIR / "rex_omni_detection_results.json"
DEFAULT_REX_POINTING_JSON = DEFAULT_OUTPUT_DIR / "rex_omni_pointing_results.json"

# Constants
EMPTY_DETECTION = {"boxes": [], "labels": [], "confidences": []}
EMPTY_POINTING = {"points": [], "labels": [], "confidences": []}

# Processing limit: set to None to process all entries, or an integer to limit (e.g., 5)
MAX_ENTRIES_TO_PROCESS = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# 1. Image Processing & Utilities
# ============================================================================

CAMERA_ROTATIONS = {
    "head_rgb_left": "90_cw",
    "head_rgb_right": "90_cw",
    "right_rgb": "180",
}

def rotate_images_for_cameras(images_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Return a new dict with camera-specific rotations applied."""
    result = {}
    for cam, img in images_dict.items():
        mode = CAMERA_ROTATIONS.get(cam)
        if mode == "90_cw":
            result[cam] = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif mode == "180":
            result[cam] = cv2.rotate(img, cv2.ROTATE_180)
        else:
            result[cam] = img
    return result

# ============================================================================
# 2. GPU & Resource Management
# ============================================================================

def _setup_gpu_tracking(logger_obj):
    """Common GPU setup: check CUDA and prepare memory tracking."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Models require GPU.")
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    return mem_before

def _finish_gpu_tracking(logger_obj, model_name, mem_before):
    """Calculate and log GPU memory usage."""
    mem_after = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    gpu_memory_mb = mem_after - mem_before
    logger_obj.info(f"{model_name} initialized. GPU memory: %.2f MB", gpu_memory_mb)
    return gpu_memory_mb

def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    torch.cuda.empty_cache()

# ============================================================================
# 3. Model Initialization
# ============================================================================

def initialize_vlm(logger_obj):
    """Initialize only the VLM detector. Returns (vlm_detector, gpu_memory_mb)."""
    logger_obj.info("Initializing VLM...")
    mem_before = _setup_gpu_tracking(logger_obj)
    
    logger_obj.info(f"Loading VLM: {VLM_MODEL_NAME}")
    vlm_detector = create_vlm_detector(
        model_type="internvlm",
        model_name=VLM_MODEL_NAME,
        logger=logger_obj,
    )
    
    gpu_memory_mb = _finish_gpu_tracking(logger_obj, "VLM", mem_before)
    return vlm_detector, gpu_memory_mb

def initialize_yolo(logger_obj):
    """Initialize only the YOLO-World model. Returns (yolo_model, gpu_memory_mb)."""
    logger_obj.info("Initializing YOLO-World...")
    mem_before = _setup_gpu_tracking(logger_obj)
    
    logger_obj.info(f"Loading YOLO-World: {YOLO_MODEL_NAME}")
    yolo_model = YOLOWorld(YOLO_MODEL_NAME)
    yolo_model.model.to("cpu")
    
    gpu_memory_mb = _finish_gpu_tracking(logger_obj, "YOLO-World", mem_before)
    return yolo_model, gpu_memory_mb

def initialize_rex(logger_obj):
    """Initialize only the Rex-Omni wrapper. Returns (rex_omni_wrapper, gpu_memory_mb)."""
    logger_obj.info("Initializing Rex-Omni...")
    mem_before = _setup_gpu_tracking(logger_obj)
    
    logger_obj.info(f"Loading Rex-Omni with transformers backend: {REX_OMNI_MODEL_PATH}")
    # Note: top_p and top_k are omitted to avoid warnings when do_sample=False (temperature=0.0)
    rex_omni_wrapper = RexOmniWrapper(
        model_path=REX_OMNI_MODEL_PATH,
        backend="transformers",
        max_tokens=2048,
        temperature=0.0,  # Set to 0.0 to ensure do_sample=False (deterministic mode)
        # top_p and top_k removed - not used when do_sample=False
        repetition_penalty=1.05,
        trust_remote_code=True,
    )
    
    gpu_memory_mb = _finish_gpu_tracking(logger_obj, "Rex-Omni", mem_before)
    return rex_omni_wrapper, gpu_memory_mb

# ============================================================================
# 4. Data Loading
# ============================================================================

def load_captured_buffer(pickle_path: Path) -> OrderedDict:
    """Load captured buffer from pickle file and return OrderedDict."""
    import pickle

    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    logger.info("Loading pickle file: %s", pickle_path)
    with open(pickle_path, "rb") as f:
        buffer_data = pickle.load(f)

    if isinstance(buffer_data, dict):
        if "entries" in buffer_data:
            entries = buffer_data["entries"]
        else:
            entries = buffer_data
    elif hasattr(buffer_data, "entries"):
        entries = buffer_data.entries
    else:
        entries = buffer_data

    if not isinstance(entries, OrderedDict):
        entries = OrderedDict(sorted(entries.items()))

    logger.info("Loaded %d entries from pickle file", len(entries))
    return entries

def iter_entries(pickle_path: Path):
    """Yield (entry_id, entry) pairs from the captured buffer.
    
    Respects MAX_ENTRIES_TO_PROCESS limit if set.
    """
    entries = load_captured_buffer(pickle_path)
    count = 0
    for entry_id, entry in entries.items():
        count += 1
        if MAX_ENTRIES_TO_PROCESS is not None and count > MAX_ENTRIES_TO_PROCESS:
            break
        yield entry_id, entry

def save_json(path: Path, data: Any) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved JSON: %s", path)

def load_json(path: Path) -> Any:
    """Load data from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================================
# 5. Model Execution Functions
# ============================================================================

def get_all_vocabulary(vocabulary_by_camera: Dict[str, List[str]]) -> List[str]:
    """Get all unique vocabulary items from vocabulary_by_camera."""
    all_objects = set()
    for camera_objects in vocabulary_by_camera.values():
        all_objects.update(camera_objects)
    return sorted(list(all_objects))

# 5.1 Vocabulary Extraction
def extract_vocabulary(vlm_detector, entry: ObsDataEntry) -> Dict[str, List[str]]:
    """Extract vocabulary from a single ObsDataEntry using VLM."""
    images_dict = entry.rgb_images
    if not images_dict:
        logger.warning("No RGB images found in entry")
        return {}

    images_dict = rotate_images_for_cameras(images_dict)

    logger.info("Running VLM on %d images...", len(images_dict))
    objects_by_camera = vlm_detector.detect_objects(images_dict)
    torch.cuda.empty_cache()

    vocabulary_by_camera: Dict[str, List[str]] = {}

    for camera_name, object_list in objects_by_camera.items():
        camera_objects: List[str] = []
        for obj in object_list:
            obj_name = obj.get("object_name", "").strip()
            if obj_name:
                camera_objects.append(obj_name)
        vocabulary_by_camera[camera_name] = camera_objects

    all_vocab = get_all_vocabulary(vocabulary_by_camera)
    logger.info("Extracted vocabulary: %d unique objects", len(all_vocab))
    
    return vocabulary_by_camera

# 5.2 YOLO Detection
def run_yolo_detection(yolo_model, images_dict: Dict[str, np.ndarray], vocabulary_by_camera: Dict[str, List[str]]) -> Dict[str, Dict]:
    """Run YOLO detection on images using per-camera vocabulary."""
    if not vocabulary_by_camera:
        logger.warning("Empty vocabulary_by_camera, skipping YOLO detection")
        return {cam: EMPTY_DETECTION for cam in images_dict.keys()}

    images_dict = rotate_images_for_cameras(images_dict)

    yolo_model.model.to("cuda")
    torch.cuda.empty_cache()

    detection_results: Dict[str, Dict[str, Any]] = {}
    
    for camera_name, image in images_dict.items():
        vocabulary = vocabulary_by_camera.get(camera_name, [])
        if not vocabulary:
            logger.debug("No vocabulary for %s, skipping", camera_name)
            detection_results[camera_name] = EMPTY_DETECTION
            continue

        logger.debug("Running YOLO on %s with %d categories...", camera_name, len(vocabulary))
        
        start_time = time.perf_counter()
        yolo_model.set_classes(vocabulary)
        yolo_result = yolo_model.predict(
            [image],
            conf=0.1,
            iou=0.80,
            verbose=False,
        )[0]
        elapsed = time.perf_counter() - start_time

        detections = sv.Detections.from_ultralytics(yolo_result)

        boxes: List[List[float]] = []
        labels: List[str] = []
        confidences: List[float] = []

        for i in range(len(detections.xyxy)):
            box = detections.xyxy[i].tolist()
            class_id = int(detections.class_id[i])
            confidence = float(detections.confidence[i])

            if class_id < len(vocabulary):
                boxes.append(box)
                labels.append(vocabulary[class_id])
                confidences.append(confidence)

        detection_results[camera_name] = {
            "boxes": boxes,
            "labels": labels,
            "confidences": confidences,
            "timing_seconds": elapsed,
        }

    torch.cuda.empty_cache()
    total_detections = sum(len(r["boxes"]) for r in detection_results.values())
    logger.info("YOLO detected %d objects across %d cameras", total_detections, len(images_dict))
    return detection_results

# 5.3 Rex-Omni Batch (Unified)
def run_rex_omni_batch(
    rex_omni_wrapper,
    images_dict: Dict[str, np.ndarray],
    vocabulary_by_camera: Dict[str, List[str]],
    task: str = "detection",
) -> Dict[str, Dict]:
    """
    Run Rex-Omni batch inference (detection or pointing).
    
    Args:
        rex_omni_wrapper: Rex-Omni wrapper instance
        images_dict: Dictionary of camera_name -> image array
        vocabulary_by_camera: Dictionary of camera_name -> vocabulary list
        task: "detection" (returns boxes) or "pointing" (returns points)
    
    Returns:
        Dictionary of camera_name -> results dict with boxes/points, labels, timing
    """
    if task not in ("detection", "pointing"):
        raise ValueError(f"task must be 'detection' or 'pointing', got '{task}'")
    
    TASK_CONFIG = {
        "detection": {
            "pred_type": "box",
            "coords_len": 4,
            "result_key": "boxes",
            "empty_result": EMPTY_DETECTION,
        },
        "pointing": {
            "pred_type": "point",
            "coords_len": 2,
            "result_key": "points",
            "empty_result": EMPTY_POINTING,
        },
    }
    config = TASK_CONFIG[task]
    
    if not vocabulary_by_camera:
        logger.warning(f"Empty vocabulary_by_camera, skipping Rex-Omni {task}")
        return {cam: config["empty_result"] for cam in images_dict.keys()}

    images_dict = rotate_images_for_cameras(images_dict)

    images_list = []
    vocabularies_list = []
    camera_names = []
    
    for camera_name, image in images_dict.items():
        vocabulary = vocabulary_by_camera.get(camera_name, [])
        if not vocabulary:
            logger.debug("No vocabulary for %s, skipping", camera_name)
            continue
        
        pil_image = Image.fromarray(image.astype(np.uint8) if image.dtype != np.uint8 else image)
        images_list.append(pil_image)
        vocabularies_list.append(vocabulary)
        camera_names.append(camera_name)
    
    if not images_list:
        return {cam: config["empty_result"] for cam in images_dict.keys()}
    
    logger.info("Running Rex-Omni batch %s on %d images...", task, len(images_list))
    results = rex_omni_wrapper.inference(
        images=images_list,
        task=task,
        categories=vocabularies_list,
    )
    
    task_results = {}
    for i, camera_name in enumerate(camera_names):
        result = results[i]
        if not result.get("success", False):
            logger.warning("Rex-Omni %s failed for %s", task, camera_name)
            task_results[camera_name] = config["empty_result"]
            continue
        
        inference_time = result.get("inference_time", 0.0)
        extracted_predictions = result.get("extracted_predictions", {})
        
        coords_list: List[List[float]] = []
        labels: List[str] = []
        confidences: List[Any] = []
        
        for category, predictions in extracted_predictions.items():
            for pred in predictions:
                if pred.get("type") == config["pred_type"]:
                    coords = pred.get("coords", [])
                    if len(coords) == config["coords_len"]:
                        coords_list.append(coords)
                        labels.append(category)
                        confidences.append(None)
        
        task_results[camera_name] = {
            config["result_key"]: coords_list,
            "labels": labels,
            "confidences": confidences,
            "timing": {
                "inference_seconds": inference_time,
            },
        }
    
    for camera_name in images_dict.keys():
        if camera_name not in task_results:
            task_results[camera_name] = config["empty_result"]
    
    total_count = sum(len(r[config["result_key"]]) for r in task_results.values())
    logger.info("Rex-Omni %s: %d %s across %d cameras", task, total_count, config["result_key"], len(images_dict))
    return task_results

# ============================================================================
# 6. Task Functions
# ============================================================================

def task_vocab(args) -> None:
    """Generate vocabulary JSON for each entry using VLM."""
    pickle_path = Path(args.pickle_path)
    output_json_path = Path(args.vocab_json)

    logger.info("Task: vocab")
    logger.info("Pickle path: %s", pickle_path)
    logger.info("Output JSON: %s", output_json_path)

    vlm_detector, gpu_memory_mb = initialize_vlm(logger)

    entries_data = {}
    timings = {}
    total_vocab = 0
    total = 0

    for total, (entry_id, entry) in enumerate(iter_entries(pickle_path), start=1):
        entry_key = str(entry_id)
        logger.info("Processing entry %s (%d)", entry_key, total)
        
        start_time = time.perf_counter()
        vocabulary_by_camera = extract_vocabulary(vlm_detector, entry)
        elapsed = time.perf_counter() - start_time
        timings[entry_key] = elapsed
        
        entries_data[entry_key] = {
            "vocabulary_by_camera": vocabulary_by_camera,
            "timing_seconds": elapsed,
        }

        v_count = len(get_all_vocabulary(vocabulary_by_camera))
        total_vocab += v_count
        print(f"Entry {entry_key}: vocab={v_count}, time={elapsed:.3f}s ({total})", flush=True)

    total_time = sum(timings.values())
    avg_time = (total_time / len(timings)) if timings else 0.0
    avg_vocab = (total_vocab / total) if total else 0.0

    unified_json = {
        "metadata": {
            "task": "vocab",
            "model": VLM_MODEL_NAME,
            "gpu_memory_mb": gpu_memory_mb,
            "pickle_path": str(pickle_path),
            "timestamp": datetime.now().isoformat(),
            "total_entries": total,
        },
        "entries": entries_data,
        "summary": {
            "total_entries": total,
            "total_vocab": total_vocab,
            "avg_vocab_per_entry": avg_vocab,
            "total_time_seconds": total_time,
            "avg_time_per_entry": avg_time,
            "gpu_memory_mb": gpu_memory_mb,
        },
    }
    save_json(output_json_path, unified_json)
    logger.info("=" * 60)
    logger.info("Vocabulary Task Summary:")
    logger.info("  Entries processed: %d", total)
    logger.info("  Total vocabulary: %d", total_vocab)
    logger.info("  Avg vocab per entry: %.2f", avg_vocab)
    logger.info("  Total time: %.2f seconds", total_time)
    logger.info("  Avg time per entry: %.3f seconds", avg_time)
    logger.info("  GPU memory: %.2f MB", gpu_memory_mb)
    logger.info("=" * 60)

def task_yolo(args) -> None:
    """Run YOLO-World detection using per-entry vocabulary."""
    pickle_path = Path(args.pickle_path)
    vocab_json_path = Path(args.vocab_json)
    output_json_path = Path(args.yolo_json)

    logger.info("Task: yolo")
    logger.info("Pickle path: %s", pickle_path)
    logger.info("Vocab JSON: %s", vocab_json_path)
    logger.info("Output JSON: %s", output_json_path)

    vocab_data = load_json(vocab_json_path)
    yolo_model, gpu_memory_mb = initialize_yolo(logger)

    entries_data = {}
    timings = {}
    total_detections = 0
    total = 0

    for total, (entry_id, entry) in enumerate(iter_entries(pickle_path), start=1):
        entry_key = str(entry_id)
        vocab_entry = vocab_data.get("entries", {}).get(entry_key)
        if not vocab_entry:
            logger.warning("No vocabulary for entry %s, skipping", entry_key)
            continue

        vocabulary_by_camera = vocab_entry.get("vocabulary_by_camera", {})
        images_dict = getattr(entry, "rgb_images", None)

        logger.info("Running YOLO for entry %s (%d)", entry_key, total)
        start_time = time.perf_counter()
        detections = run_yolo_detection(yolo_model, images_dict, vocabulary_by_camera)
        elapsed = time.perf_counter() - start_time
        timings[entry_key] = elapsed

        per_camera_timing = {
            cam: res.get("timing_seconds", 0.0)
            for cam, res in detections.items()
        }

        entries_data[entry_key] = {
            "detections": detections,
            "timing_seconds": elapsed,
            "per_camera_timing": per_camera_timing,
        }

        entry_dets = sum(len(cam_res.get("boxes", [])) for cam_res in detections.values())
        total_detections += entry_dets
        print(f"Entry {entry_key}: detections={entry_dets}, time={elapsed:.3f}s", flush=True)

    total_time = sum(timings.values())
    avg_time = (total_time / len(timings)) if timings else 0.0

    unified_json = {
        "metadata": {
            "task": "yolo",
            "model": YOLO_MODEL_NAME,
            "gpu_memory_mb": gpu_memory_mb,
            "pickle_path": str(pickle_path),
            "vocab_json_path": str(vocab_json_path),
            "timestamp": datetime.now().isoformat(),
            "total_entries": total,
        },
        "entries": entries_data,
        "summary": {
            "total_entries": total,
            "total_detections": total_detections,
            "avg_detections_per_entry": (total_detections / total) if total else 0.0,
            "total_time_seconds": total_time,
            "avg_time_per_entry": avg_time,
            "gpu_memory_mb": gpu_memory_mb,
        },
    }

    save_json(output_json_path, unified_json)

    logger.info("=" * 60)
    logger.info("YOLO Task Summary:")
    logger.info("  Entries processed: %d", total)
    logger.info("  Total detections: %d", total_detections)
    logger.info("  Avg detections per entry: %.2f", (total_detections / total) if total else 0.0)
    logger.info("  Total time: %.2f seconds", total_time)
    logger.info("  Avg time per entry: %.3f seconds", avg_time)
    logger.info("  GPU memory: %.2f MB", gpu_memory_mb)
    logger.info("=" * 60)

def task_rex_detection(args) -> None:
    """Run Rex-Omni detection using per-entry vocabulary."""
    pickle_path = Path(args.pickle_path)
    vocab_json_path = Path(args.vocab_json)
    output_json_path = Path(args.rex_detection_json)

    logger.info("Task: rex_detection")
    logger.info("Pickle path: %s", pickle_path)
    logger.info("Vocab JSON: %s", vocab_json_path)
    logger.info("Output JSON: %s", output_json_path)

    vocab_data = load_json(vocab_json_path)
    rex_omni_wrapper, gpu_memory_mb = initialize_rex(logger)

    entries_data = {}
    timings = {}
    total_detections = 0
    total = 0

    for total, (entry_id, entry) in enumerate(iter_entries(pickle_path), start=1):
        entry_key = str(entry_id)
        vocab_entry = vocab_data.get("entries", {}).get(entry_key)
        if not vocab_entry:
            logger.warning("No vocabulary for entry %s, skipping", entry_key)
            continue

        vocabulary_by_camera = vocab_entry.get("vocabulary_by_camera", {})
        images_dict = getattr(entry, "rgb_images", None)

        logger.info("Running Rex-Omni detection for entry %s (%d)", entry_key, total)
        start_time = time.perf_counter()
        detections = run_rex_omni_batch(rex_omni_wrapper, images_dict, vocabulary_by_camera, task="detection")
        elapsed = time.perf_counter() - start_time
        timings[entry_key] = elapsed

        per_camera_timing = {
            cam: res.get("timing", {}).get("inference_seconds", 0.0)
            for cam, res in detections.items()
            if "timing" in res
        }

        entries_data[entry_key] = {
            "detections": detections,
            "timing_seconds": elapsed,
            "per_camera_timing": per_camera_timing,
        }

        entry_dets = sum(len(cam_res.get("boxes", [])) for cam_res in detections.values())
        total_detections += entry_dets
        print(f"Entry {entry_key}: detections={entry_dets}, time={elapsed:.3f}s", flush=True)

    total_time = sum(timings.values())
    avg_time = (total_time / len(timings)) if timings else 0.0

    unified_json = {
        "metadata": {
            "task": "rex_detection",
            "model": REX_OMNI_MODEL_PATH,
            "gpu_memory_mb": gpu_memory_mb,
            "pickle_path": str(pickle_path),
            "vocab_json_path": str(vocab_json_path),
            "timestamp": datetime.now().isoformat(),
            "total_entries": total,
        },
        "entries": entries_data,
        "summary": {
            "total_entries": total,
            "total_detections": total_detections,
            "avg_detections_per_entry": (total_detections / total) if total else 0.0,
            "total_time_seconds": total_time,
            "avg_time_per_entry": avg_time,
            "gpu_memory_mb": gpu_memory_mb,
        },
    }

    save_json(output_json_path, unified_json)

    logger.info("=" * 60)
    logger.info("Rex-Omni Detection Task Summary:")
    logger.info("  Entries processed: %d", total)
    logger.info("  Total detections: %d", total_detections)
    logger.info("  Avg detections per entry: %.2f", (total_detections / total) if total else 0.0)
    logger.info("  Total time: %.2f seconds", total_time)
    logger.info("  Avg time per entry: %.3f seconds", avg_time)
    logger.info("  GPU memory: %.2f MB", gpu_memory_mb)
    logger.info("=" * 60)

def task_rex_pointing(args) -> None:
    """Run Rex-Omni pointing using per-entry vocabulary."""
    pickle_path = Path(args.pickle_path)
    vocab_json_path = Path(args.vocab_json)
    output_json_path = Path(args.rex_pointing_json)

    logger.info("Task: rex_pointing")
    logger.info("Pickle path: %s", pickle_path)
    logger.info("Vocab JSON: %s", vocab_json_path)
    logger.info("Output JSON: %s", output_json_path)

    vocab_data = load_json(vocab_json_path)
    rex_omni_wrapper, gpu_memory_mb = initialize_rex(logger)

    entries_data = {}
    timings = {}
    total_points = 0
    total = 0

    for total, (entry_id, entry) in enumerate(iter_entries(pickle_path), start=1):
        entry_key = str(entry_id)
        vocab_entry = vocab_data.get("entries", {}).get(entry_key)
        if not vocab_entry:
            logger.warning("No vocabulary for entry %s, skipping", entry_key)
            continue

        vocabulary_by_camera = vocab_entry.get("vocabulary_by_camera", {})
        images_dict = getattr(entry, "rgb_images", None)

        logger.info("Running Rex-Omni pointing for entry %s (%d)", entry_key, total)
        start_time = time.perf_counter()
        pointing_results = run_rex_omni_batch(rex_omni_wrapper, images_dict, vocabulary_by_camera, task="pointing")
        elapsed = time.perf_counter() - start_time
        timings[entry_key] = elapsed

        per_camera_timing = {
            cam: res.get("timing", {}).get("inference_seconds", 0.0)
            for cam, res in pointing_results.items()
            if "timing" in res
        }

        entries_data[entry_key] = {
            "points": pointing_results,
            "timing_seconds": elapsed,
            "per_camera_timing": per_camera_timing,
        }

        entry_points = sum(len(cam_res.get("points", [])) for cam_res in pointing_results.values())
        total_points += entry_points
        print(f"Entry {entry_key}: points={entry_points}, time={elapsed:.3f}s", flush=True)

    total_time = sum(timings.values())
    avg_time = (total_time / len(timings)) if timings else 0.0

    unified_json = {
        "metadata": {
            "task": "rex_pointing",
            "model": REX_OMNI_MODEL_PATH,
            "gpu_memory_mb": gpu_memory_mb,
            "pickle_path": str(pickle_path),
            "vocab_json_path": str(vocab_json_path),
            "timestamp": datetime.now().isoformat(),
            "total_entries": total,
        },
        "entries": entries_data,
        "summary": {
            "total_entries": total,
            "total_points": total_points,
            "avg_points_per_entry": (total_points / total) if total else 0.0,
            "total_time_seconds": total_time,
            "avg_time_per_entry": avg_time,
            "gpu_memory_mb": gpu_memory_mb,
        },
    }

    save_json(output_json_path, unified_json)

    logger.info("=" * 60)
    logger.info("Rex-Omni Pointing Task Summary:")
    logger.info("  Entries processed: %d", total)
    logger.info("  Total points: %d", total_points)
    logger.info("  Avg points per entry: %.2f", (total_points / total) if total else 0.0)
    logger.info("  Total time: %.2f seconds", total_time)
    logger.info("  Avg time per entry: %.3f seconds", avg_time)
    logger.info("  GPU memory: %.2f MB", gpu_memory_mb)
    logger.info("=" * 60)

# ============================================================================
# 7. CLI & Main
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detection Processing Pipeline")
    parser.add_argument(
        "--task",
        required=True,
        choices=["vocab", "yolo", "rex_detection", "rex_pointing"],
        help="Which task to run.",
    )
    parser.add_argument(
        "--pickle-path",
        type=str,
        default=str(DEFAULT_PICKLE_PATH),
        help=f"Path to obs buffer pickle file (default: {DEFAULT_PICKLE_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--vocab-json",
        type=str,
        default=str(DEFAULT_VOCAB_JSON),
        help="Path to vocabulary JSON output/input.",
    )
    parser.add_argument(
        "--yolo-json",
        type=str,
        default=str(DEFAULT_YOLO_JSON),
        help="Path to YOLO detection JSON output.",
    )
    parser.add_argument(
        "--rex-detection-json",
        type=str,
        default=str(DEFAULT_REX_DETECTION_JSON),
        help="Path to Rex-Omni detection JSON output.",
    )
    parser.add_argument(
        "--rex-pointing-json",
        type=str,
        default=str(DEFAULT_REX_POINTING_JSON),
        help="Path to Rex-Omni pointing JSON output.",
    )

    return parser

def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    output_base = Path(args.output_dir)
    if Path(args.vocab_json) == DEFAULT_VOCAB_JSON:
        args.vocab_json = str(output_base / DEFAULT_VOCAB_JSON.name)
    if Path(args.yolo_json) == DEFAULT_YOLO_JSON:
        args.yolo_json = str(output_base / DEFAULT_YOLO_JSON.name)
    if Path(args.rex_detection_json) == DEFAULT_REX_DETECTION_JSON:
        args.rex_detection_json = str(output_base / DEFAULT_REX_DETECTION_JSON.name)
    if Path(args.rex_pointing_json) == DEFAULT_REX_POINTING_JSON:
        args.rex_pointing_json = str(output_base / DEFAULT_REX_POINTING_JSON.name)

    # Ensure all JSON output directories exist (for both default and custom paths)
    for json_path in [args.vocab_json, args.yolo_json, args.rex_detection_json, args.rex_pointing_json]:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)

    if args.task == "vocab":
        task_vocab(args)
    elif args.task == "yolo":
        task_yolo(args)
    elif args.task == "rex_detection":
        task_rex_detection(args)
    elif args.task == "rex_pointing":
        task_rex_pointing(args)
    else:
        parser.error(f"Unknown task: {args.task}")

if __name__ == "__main__":
    main()

