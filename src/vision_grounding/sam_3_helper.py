#!/usr/bin/env python3
"""
SAM 3 Segmentation Helper
Simple box prompt mode: one SAM 3 call per image with multiple box prompts.

Author: Robin Eshraghi 
Based on fast_sam_helper2.py
Created: 12/19/25

Usage: "python sam_3_helper.py --real --viz --profile"
                    --real: use real data
                    --viz: visualize the results
                    --profile: profile the code            

Input data path:            /logs/current_run_outputs/offline_outputs/image_data_dumps
Output data path:           /logs/current_run_outputs/offline_outputs/sam_3_helper_benchmark
Output visualization path:  /logs/current_run_outputs/offline_outputs/sam_3_helper_benchmark/visualizations
"""

import numpy as np
import cv2
import time
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
from contextlib import redirect_stdout
from io import StringIO

try:
    from ultralytics import SAM
    from PIL import Image
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    SAM = None


# ============================================================================
# CONFIGURATION
# ============================================================================

SAM3_CONFIG = {
    "conf": 0.25,
    "iou": 0.7,
    "imgsz": 1036,
    "device": "cuda"
}

MATCHING_CONFIG = {
    "iou_threshold": 0.7,
    "duplicate_threshold": 0.9
}

# Benchmark flag (only True when running as main)
_BENCHMARK_ENABLED = False

# Profiling flag (enable for timing analysis)
_PROFILE_ENABLED = False


# ============================================================================
# PROFILING
# ============================================================================

def _profile(func):
    """Minimal profiling decorator (zero overhead when disabled)."""
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _PROFILE_ENABLED:
            return func(*args, **kwargs)
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        print(f"[PROFILE] {func.__name__}: {elapsed:.3f}ms")
        return result
    return wrapper


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _create_empty_masks(num: int, shape: Tuple[int, int]) -> List[np.ndarray]:
    """Create list of empty boolean masks."""
    return [np.zeros(shape, dtype=bool) for _ in range(num)]

def _get_mask_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
    """Get bounding box [x1, y1, x2, y2] from mask. Returns None if empty."""
    mask_y, mask_x = np.where(mask)
    if len(mask_y) == 0:
        return None
    return np.array([mask_x.min(), mask_y.min(), mask_x.max(), mask_y.max()])

def _calculate_bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Calculate IoU between two bboxes [x1, y1, x2, y2]. Returns 0.0 if no overlap."""
    x1_i = max(bbox1[0], bbox2[0])
    y1_i = max(bbox1[1], bbox2[1])
    x2_i = min(bbox1[2], bbox2[2])
    y2_i = min(bbox1[3], bbox2[3])
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    inter = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0

@_profile
def match_masks_to_bboxes(bboxes: List[np.ndarray], masks: List[np.ndarray], 
                          iou_threshold: float = 0.3,
                          duplicate_threshold: float = 0.9,
                          config: Optional[dict] = None) -> Tuple[List[Tuple[int, int, float]], List[int]]:
    """
    Match masks to bboxes using IoU-based greedy matching (one-to-one).
    For unmatched bboxes, check if they're duplicates of matched bboxes and copy masks.
    
    Args:
        bboxes: List of [x1, y1, x2, y2] bounding boxes
        masks: List of boolean masks (H, W)
        iou_threshold: Minimum IoU for a valid match (default: 0.3)
        duplicate_threshold: IoU threshold to consider bboxes as duplicates (default: 0.9)
        config: Optional config dict with 'iou_threshold' and 'duplicate_threshold' (overrides defaults if provided)
    
    Returns:
        Tuple of (matches, unmatched_bbox_indices):
        - matches: List of (bbox_idx, mask_idx, iou) tuples for matched pairs
        - unmatched_bbox_indices: List of bbox indices with no match (after Phase 2)
    """
    # Use config if provided, otherwise use function defaults
    if config is not None:
        iou_threshold = config.get("iou_threshold", iou_threshold)
        duplicate_threshold = config.get("duplicate_threshold", duplicate_threshold)
    matches = []
    used_mask_indices = set()
    
    # Pre-convert bboxes to arrays
    bbox_arrays = [np.array(bbox, dtype=float) for bbox in bboxes]
    
    # Phase 1: Normal matching (bbox -> mask)
    for bbox_idx, bbox_array in enumerate(bbox_arrays):
        best_mask_idx = None
        best_iou = 0.0
        
        for mask_idx, mask in enumerate(masks):
            if mask_idx in used_mask_indices:
                continue
            
            mask_bbox = _get_mask_bbox(mask)
            if mask_bbox is None:
                continue
            
            iou = _calculate_bbox_iou(bbox_array, mask_bbox)
            if iou > best_iou:
                best_iou = iou
                best_mask_idx = mask_idx
        
        if best_mask_idx is not None and best_iou > iou_threshold:
            matches.append((bbox_idx, best_mask_idx, best_iou))
            used_mask_indices.add(best_mask_idx)
    
    # Phase 2: Handle duplicate bboxes (unmatched -> matched)
    matched_bbox_indices = {bbox_idx for bbox_idx, _, _ in matches}
    matched_bbox_to_mask = {bbox_idx: mask_idx for bbox_idx, mask_idx, _ in matches}
    
    # Only run Phase 2 if there are unmatched bboxes
    if len(matched_bbox_indices) < len(bboxes):
        for bbox_idx in range(len(bboxes)):
            if bbox_idx in matched_bbox_indices:
                continue
            
            bbox_array = bbox_arrays[bbox_idx]
            best_matched_bbox_idx = None
            best_duplicate_iou = 0.0
            
            # Check IoU with matched bboxes
            for matched_bbox_idx in matched_bbox_indices:
                duplicate_iou = _calculate_bbox_iou(bbox_array, bbox_arrays[matched_bbox_idx])
                
                if duplicate_iou > best_duplicate_iou:
                    best_duplicate_iou = duplicate_iou
                    best_matched_bbox_idx = matched_bbox_idx
            
            # If duplicate found, copy the mask
            if best_matched_bbox_idx is not None and best_duplicate_iou > duplicate_threshold:
                mask_idx = matched_bbox_to_mask[best_matched_bbox_idx]
                matches.append((bbox_idx, mask_idx, best_duplicate_iou))
                matched_bbox_indices.add(bbox_idx)  # Mark as matched
    
    # Find unmatched bboxes (after both phases)
    all_bbox_indices = set(range(len(bboxes)))
    matched_bbox_indices_after_phase2 = {bbox_idx for bbox_idx, _, _ in matches}
    unmatched_bbox_indices = sorted(list(all_bbox_indices - matched_bbox_indices_after_phase2))
    
    return matches, unmatched_bbox_indices

def _visualize_benchmark_result(image: np.ndarray, bboxes: List[np.ndarray], 
                                masks: List[np.ndarray], output_path: Path,
                                total_objects: int = 0,
                                matched_objects: int = 0,
                                failed_objects: int = 0,
                                processing_time: float = 0.0,
                                current_objects: int = 0,
                                current_matched: int = 0,
                                current_failed: int = 0,
                                image_number: int = 0):
    """Minimal visualization: bboxes + mask overlays with IoU-based matching."""
    vis = image.copy()
    
    # Color palette for masks (RGB)
    mask_colors = [
        (0, 150, 0),    # Green
        #(255, 255, 0),  # Yellow
        #(255, 0, 255),  # Magenta
        #(0, 255, 255),  # Cyan
        #(255, 128, 0),  # Orange
        #(128, 0, 255),  # Purple
        #(0, 255, 128),  # Teal
        #(255, 0, 128),  # Pink
    ]
    
    # Get matches using matching function
    matches, unmatched_bbox_indices = match_masks_to_bboxes(bboxes, masks, config=MATCHING_CONFIG)
    matched_mask_indices = {mask_idx for _, mask_idx, _ in matches}
    
    # Draw matched pairs
    for bbox_idx, mask_idx, _ in matches:
        bbox = bboxes[bbox_idx]
        mask = masks[mask_idx]
        x1, y1, x2, y2 = map(int, bbox)
        color = mask_colors[mask_idx % len(mask_colors)]
        
        if np.sum(mask) > 0:
            vis[mask] = (vis[mask] * 0.6 + np.array(color) * 0.4).astype(np.uint8)
        bbox_color = tuple(c // 2 for c in color)
        cv2.rectangle(vis, (x1, y1), (x2, y2), bbox_color, 1)
    
    # Draw unmatched bboxes (red)
    for bbox_idx in unmatched_bbox_indices:
        bbox = bboxes[bbox_idx]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Red
    
    # Draw unmatched masks (red) and their bounding boxes (blue)
    for mask_idx, mask in enumerate(masks):
        if mask_idx not in matched_mask_indices and np.sum(mask) > 0:
            color = (0, 0, 200)  # blue for mask overlay
            vis[mask] = (vis[mask] * 0.6 + np.array(color) * 0.4).astype(np.uint8)
            
            # Draw mask bounding box in blue
            mask_bbox = _get_mask_bbox(mask)
            if mask_bbox is not None:
                x1, y1, x2, y2 = map(int, mask_bbox)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Blue (RGB)
    
    # Add title at top and legend at bottom
    H, W = vis.shape[:2]
    title_height = 30
    legend_height = 150
    vis_with_legend = np.ones((H + title_height + legend_height, W, 3), dtype=np.uint8) * 255
    vis_with_legend[title_height:title_height + H, :] = vis
    
    # Draw title
    font_title = cv2.FONT_HERSHEY_SIMPLEX
    title_text = "Matching: YOLO bbox and SAM3 mask"
    title_scale = 0.7
    title_thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(title_text, font_title, title_scale, title_thickness)
    title_x = (W - text_width) // 2
    title_y = title_height - 5
    cv2.putText(vis_with_legend, title_text, (title_x, title_y), font_title, title_scale, (0, 0, 0), title_thickness)
    
    # Draw legend text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    y_offset = title_height + H + 20
    
    # Red: unmatched YOLO bbox
    cv2.rectangle(vis_with_legend, (10, y_offset - 10), (30, y_offset + 5), (255, 0, 0), -1)
    cv2.putText(vis_with_legend, "Red: unmatched YOLO bbox", (35, y_offset + 5), font, font_scale, (0, 0, 0), thickness)
    
    # Blue: unmatched SAM3 mask and box
    cv2.rectangle(vis_with_legend, (10, y_offset + 15), (30, y_offset + 30), (0, 0, 255), -1)
    cv2.putText(vis_with_legend, "Blue: unmatched SAM3 mask and its box", (35, y_offset + 30), font, font_scale, (0, 0, 0), thickness)
    
    # Green: matched ones
    cv2.rectangle(vis_with_legend, (10, y_offset + 40), (30, y_offset + 55), (0, 150, 0), -1)
    cv2.putText(vis_with_legend, "Green: matched YOLO bbox and SAM3 mask box", (35, y_offset + 55), font, font_scale, (0, 0, 0), thickness)
    
    # IoU threshold
    iou_thresh = MATCHING_CONFIG.get("iou_threshold", 0.3)
    cv2.putText(vis_with_legend, f"Matching IoU: {iou_thresh}", (W - 200, y_offset + 30), font, font_scale, (0, 0, 0), thickness)
    
    # Image number
    if image_number > 0:
        image_num_text = f"Image #{image_number}"
        cv2.putText(vis_with_legend, image_num_text, (10, y_offset + 70), font, font_scale, (0, 0, 0), thickness)
    
    # Statistics line (current image)
    if current_objects > 0:
        current_stats_text = f"Current: {current_objects} | Matched: {current_matched} | Failed: {current_failed}"
        cv2.putText(vis_with_legend, current_stats_text, (10, y_offset + 85), font, font_scale, (0, 0, 0), thickness)
    
    # Statistics line (accumulated)
    if total_objects > 0:
        accum_stats_text = f"Accumulated: {total_objects} | Matched: {matched_objects} | Failed: {failed_objects}"
        cv2.putText(vis_with_legend, accum_stats_text, (10, y_offset + 100), font, font_scale, (0, 0, 0), thickness)
    
    # Processing time for this image
    if processing_time > 0:
        time_text = f"Processing time: {processing_time*1000:.2f}ms"
        cv2.putText(vis_with_legend, time_text, (10, y_offset + 115), font, font_scale, (0, 0, 0), thickness)
    
    # Save
    cv2.imwrite(str(output_path), cv2.cvtColor(vis_with_legend, cv2.COLOR_RGB2BGR))


# ============================================================================
# MAIN FUNCTION
# ============================================================================

@_profile
def get_sam3_masks_from_boxes(
    sam3_model: SAM,
    image: np.ndarray,
    bboxes: List[np.ndarray],
    device: str = "cuda",
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 1024,
    config: Optional[dict] = None
) -> List[np.ndarray]:
    """
    SAM 3 segmentation using box prompts: one model call for all bboxes.
    
    Args:
        sam3_model: Initialized SAM 3 model
        image: Full RGB image (H, W, 3)
        bboxes: List of [x1, y1, x2, y2] bounding boxes
        device: 'cuda' or 'cpu' (default: 'cuda')
        conf: Confidence threshold (default: 0.25)
        iou: IoU threshold for NMS (default: 0.7)
        imgsz: Image size for inference (default: 1024)
        config: Optional config dict with 'conf', 'iou', 'imgsz', 'device' (overrides defaults if provided)
    
    Returns:
        List of boolean masks (H, W), one per bbox
    """
    # Use config if provided, otherwise use function defaults
    if config is not None:
        conf = config.get("conf", conf)
        iou = config.get("iou", iou)
        imgsz = config.get("imgsz", imgsz)
        device = config.get("device", device)
    
    H, W = image.shape[:2]
    
    if not bboxes:
        return []
    
    # Convert image to PIL if needed
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Convert bboxes to list format for SAM 3
    # SAM 3 expects bboxes as list of [x1, y1, x2, y2] or list of lists
    box_prompts = []
    for bbox in bboxes:
        if isinstance(bbox, np.ndarray):
            box_prompts.append(bbox.tolist())
        else:
            box_prompts.append(list(bbox))
    
    masks = []
    result = None
    try:
        with redirect_stdout(StringIO()):
            # SAM 3 API: model.predict() with bboxes parameter
            # For visual prompts (boxes), SAM 3 segments the specific object in each box
            # SAM 3 supports both single box [x1, y1, x2, y2] and multiple boxes [[x1, y1, x2, y2], ...]
            results = sam3_model.predict(
                source=pil_image,
                bboxes=box_prompts,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                verbose=False
            )
        
        result = results[0] if len(results) > 0 else None
        
        # Extract masks from result (same structure as FastSAM)
        if result is not None and result.masks is not None and result.masks.data is not None:
            masks_tensor = result.masks.data.cpu().numpy()
            
            for mask_data in masks_tensor:
                mask_uint8 = mask_data.astype(np.uint8)
                if mask_uint8.shape != (H, W):
                    mask = cv2.resize(mask_uint8, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    mask = mask_uint8.astype(bool)
                masks.append(mask)
        else:
            masks.extend(_create_empty_masks(len(bboxes), (H, W)))
    
    except Exception as e:
        print(f"Warning: SAM 3 inference failed: {e}")
        masks.extend(_create_empty_masks(len(bboxes), (H, W)))
    
    # Ensure we return exactly one mask per bbox (same as reference code)
    while len(masks) < len(bboxes):
        masks.append(np.zeros((H, W), dtype=bool))
    
    return masks[:len(bboxes)]  # Trim if we got more masks than bboxes


# ============================================================================
# BENCHMARKING (only runs when script executed directly)
# ============================================================================

def load_real_test_cases(data_dir: Path, max_cases: int = 10) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
    """Load real test cases from image_data_dumps."""
    image_data_dir = data_dir / "image_data_dumps"
    if not image_data_dir.exists():
        return []
    
    test_cases = []
    for pkl_path in sorted(image_data_dir.glob("image_data_*.pkl"))[:max_cases]:
        with open(pkl_path, 'rb') as f:
            viz_data = pickle.load(f)
        for image_data in viz_data.values():
            image = image_data.get('rotated_rgb_image')
            if image is None:
                continue
            bboxes = [np.array(obj['rotated_bbox_xyxy']) 
                     for obj in image_data.get('yolo_object_dict', {}).values() 
                     if obj.get('rotated_bbox_xyxy')]
            if bboxes:
                test_cases.append((image, bboxes))
    return test_cases

def run_benchmark(model_path: str = "sam3.pt", num_iterations: int = 10, 
                  use_real_data: bool = False, real_data_path: Optional[str] = None,
                  enable_viz: bool = False):
    """Run benchmark when script is executed directly."""
    global _BENCHMARK_ENABLED
    
    if not SAM3_AVAILABLE:
        print("ERROR: SAM 3 not available. Please install ultralytics: pip install -U ultralytics")
        return
    
    print(f"\n{'='*70}")
    print(f"SAM 3 Benchmark")
    print(f"{'='*70}")
    
    # Setup viz output if enabled
    viz_dir = None
    if enable_viz:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        viz_dir = project_root / "logs" / "current_run_outputs" / "offline_outputs" / "sam_3_helper_benchmark"
        viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"Visualization enabled: saving to {viz_dir}\n")
    
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SAM(model_path)
        print(f"Model: {model_path} | Device: {device}\n")
    except Exception as e:
        print(f"ERROR: Failed to load SAM 3 model: {e}")
        print("Note: SAM 3 weights (sam3.pt) must be downloaded separately from Hugging Face.")
        return
    
    # Load test cases
    if use_real_data and real_data_path:
        print(f"Loading real data from: {real_data_path}")
        test_cases = load_real_test_cases(Path(real_data_path), max_cases=num_iterations)
        if not test_cases:
            print("ERROR: No test cases found")
            return
        print(f"Loaded {len(test_cases)} test cases\n")
    else:
        # Synthetic data
        H, W = 480, 640
        test_cases = [(
            np.random.randint(0, 255, (H, W, 3), dtype=np.uint8),
            [np.array([50, 50, 150, 150]), np.array([200, 100, 300, 200]), np.array([400, 200, 500, 300])]
        ) for _ in range(num_iterations)]
    
    config = SAM3_CONFIG.copy()
    config["device"] = device
    
    _BENCHMARK_ENABLED = True
    
    # Warmup
    test_image, test_bboxes = test_cases[0]
    _ = get_sam3_masks_from_boxes(model, test_image, test_bboxes, device=device, config=config)

    
    # Benchmark
    print(f"Running {len(test_cases)} iterations...\n")
    times = []
    total_objects = 0
    total_matched = 0
    total_failed = 0
    
    for idx, (test_image, test_bboxes) in enumerate(test_cases):
        start = time.perf_counter()
        masks = get_sam3_masks_from_boxes(model, test_image, test_bboxes, device=device, config=config)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        # Calculate stats for this image
        matches, unmatched_bbox_indices = match_masks_to_bboxes(test_bboxes, masks, config=MATCHING_CONFIG)
        num_bboxes = len(test_bboxes)
        num_matched = len(matches)
        num_failed = len(unmatched_bbox_indices)
        
        # Accumulate
        total_objects += num_bboxes
        total_matched += num_matched
        total_failed += num_failed
        
        # Save visualization if enabled
        if enable_viz and viz_dir:
            output_path = viz_dir / f"benchmark_{idx:03d}.png"
            _visualize_benchmark_result(
                test_image, test_bboxes, masks, output_path,
                total_objects=total_objects,
                matched_objects=total_matched,
                failed_objects=total_failed,
                processing_time=elapsed,
                current_objects=num_bboxes,
                current_matched=num_matched,
                current_failed=num_failed,
                image_number=idx + 1
            )
    
    _BENCHMARK_ENABLED = False
    
    # Results
    times_ms = [t * 1000 for t in times]
    print(f"{'='*70}")
    print(f"Results: {np.mean(times_ms):.2f}ms avg | {np.min(times_ms):.2f}ms min | {np.max(times_ms):.2f}ms max")
    print(f"FPS: {1000/np.mean(times_ms):.2f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    enable_viz = "--viz" in sys.argv
    _PROFILE_ENABLED = "--profile" in sys.argv  # Enable profiling if flag present
    
    if _PROFILE_ENABLED:
        print("[PROFILE] Profiling enabled\n")
    
    # Default: synthetic data
    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        # Real data mode
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent  # Go up from vision_grounding -> tiamat_agent -> tiamatl_eval_mvp
        data_path = project_root / "logs" / "current_run_outputs" / "offline_outputs"
        run_benchmark(
            use_real_data=True,
            real_data_path=str(data_path),
            num_iterations=55,
            enable_viz=enable_viz
        )
    else:
        # Synthetic data mode (default)
        run_benchmark(num_iterations=55, enable_viz=enable_viz)

