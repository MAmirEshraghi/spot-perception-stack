#!/usr/bin/env python3
"""
Object Detection & 3D Localization Pipeline - REFACTORED VERSION

Processes robot observation buffer entry-by-entry using:
- VLM (Vision-Language Model) for semantic understanding + descriptions
- YOLO-World for precise 2D bounding box detection
- Fast SAM for geometric segmentation
- Depth data for 3D world coordinate localization

Outputs:
- all_objects.json: Comprehensive object database with rich metadata
- all_points.ply: Full reconstructed scene point cloud

Author: Robin Eshraghi 
initial 10/30/25

MAJOR CHANGES:
(update: 2025-11-13)
- Unified pair-based processing: prepare_pairs() + parse_rgbd_image_dicts_for_objects()
- Removed redundant camera mapping functions (use obs_data_buffer)
- Batch 3D extraction using color-encoded bboxes
- Deferred image resizing (VLM/YOLO use original resolution, resize only for 3D processing)
- Modular single-responsibility functions
- Configuration structure for function parameters
(Update: 2025-11-21)
- Robot pose extraction and caching for each frame
(Update: 2025-12-16)
- Added Fast SAM segmentation
- Added color index management
- Added mask saving in image dumps
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard libraries
import pickle
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
from datetime import datetime
from collections import defaultdict
import sys
import cv2

# Deep learning libraries
import torch
from ultralytics import YOLOWorld
import supervision as sv

# Fast SAM for segmentation
try:
    from ultralytics import FastSAM
    FASTSAM_AVAILABLE = True
except ImportError:
    FASTSAM_AVAILABLE = False
    FastSAM = None

# self.session_logger = SessionLogger(mapping_config.session_id, "sensor_object_mapper")

# Import from existing perception modules
from src.vision_grounding.obs_data_buffer import (
    ObsDataBuffer, 
    ObsDataEntry,   
    compose_transforms_optimized,
    # create_segmented_point_cloud,  # new: I added in obs_data_buffer
)
from src.vision_grounding.vlm_interface import VLMInterface,create_vlm_detector

from tiamat_agent.mapping.occupancy_grid import get_robot_world_coords as _get_robot_world_coords
from src.utils.ros_utils import quaternion_to_yaw
from src.utils.func_utils import v_time_fn
from src.utils.bbox_utils import (
    scale_bboxes,
    scale_bbox,
    unrotate_bboxes,
    create_mask_from_bbox_center,
    create_mask_from_bbox,
    bbox_area,
    bbox_center_pixel,
    bbox_width,
    bbox_height,
    generate_distinct_colors,
    generate_distinct_color_for_index,
)

# Import Fast SAM helpers
from src.vision_grounding.fast_sam_helper2 import (
    get_fastsam_masks_from_boxes,
    match_masks_to_bboxes,
    FASTSAM_AVAILABLE as FASTSAM_HELPER_AVAILABLE
)


# ============================================================================
# GLOBAL COLOR INDEX MANAGEMENT (Module-level state)
# ============================================================================
_global_color_index = 0  # Internal state, not exposed as parameter

def reset_color_index(initial_value: int = 0) -> None:
    """Reset the global color index to a starting value."""
    global _global_color_index
    _global_color_index = initial_value

def get_color_index() -> int:
    """Get current global color index."""
    return _global_color_index

def increment_color_index(count: int) -> None:
    """Increment global color index by count."""
    global _global_color_index
    _global_color_index += count


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
HAS_WORLD_COORDS = True
SCRIPT_DIR = Path(__file__).resolve().parent
# Project root directory (tiamatl_eval_mvp/)
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # vision_grounding -> tiamat_agent -> tiamatl_eval_mvp
# Input/Output Paths
INPUT_BUFFER_PATH = SCRIPT_DIR / "data" / "obs_buffer.pkl"
OUTPUT_OBJECTS_PATH = SCRIPT_DIR / "data" / "all_objects.json"
OUTPUT_POINTCLOUD_PATH = SCRIPT_DIR / "data" / "all_points.ply"

# Model Configuration
VLM_MODEL_TYPE = "internvlm"
VLM_MODEL_NAME = "OpenGVLab/InternVL3_5-1B"
VLM_MODEL_TYPE = "qwen2.5"
VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
YOLO_MODEL_NAME = "yolov8x-worldv2.pt"

# Processing Limits
MAX_IMAGES_TO_PROCESS = 400  # Set to integer (e.g., 200) to limit total RGB-Depth pairs processed
                               # Works consistently across all modes (prioritized, frame-by-frame, cross-frame)
BATCH_SIZE = 20 # None = frame-by-frame (~5 pairs per frame), Integer = cross-frame batching (e.g., 20)

# Prioritized Processing Mode
PRIORITIZED_JSON_PATH = SCRIPT_DIR /  "prioritized_order.json"      #"data" /

# structured function configuration
FUNCTION_CONFIGS = {
    "extract_3d_position": {
        "mask_pixel_radius": 6,       # Creates NxN grid around bbox center
        "depth_scale": 1000.0,        # Conversion factor (mm to meters)
        "filter_max_height": 20.0,    # Filter ceiling points (meters)
        "filter_min_height": 0.06,    # Filter floor/robot body points (meters)
        "min_points": 10,             # Minimum number of points to consider a valid object
    },
    "pointcloud_generation": {
        "voxel_size": 0.01,           # Downsampling voxel size (meters)
        "clip_height": 2.5,           # Height limit for scene reconstruction
        "depth_scale": 1000.0,        # Conversion factor
        "depth_trunc": 25.0,           # Maximum depth to consider (meters)
        "min_points": 10,             # Minimum number of points to consider a valid object
    },
    "yolo_detection": {
        "conf_threshold": 0.3,        # Confidence threshold
        "iou_threshold": 0.70         # IoU threshold for NMS
    },
    "fastsam_segmentation": {
        "enabled": True,              # Enable/disable Fast SAM segmentation
        "mode": "box_prompt",         # FastSAM mode (box_prompt: single call with all bboxes)
        "model_path": "FastSAM-x.pt", # Model path (will auto-download if not present)
        "conf": 0.5,                  # Confidence threshold
        "iou": 0.9,                   # IoU threshold
        "imgsz": 1024,                # Image size for inference
        "retina_masks": True,         # Use high-quality retina masks
        "device": "cuda",             # Device: 'cuda' or 'cpu'
        "matching_iou_threshold": 0.7,      # IoU threshold for matching masks to bboxes
        "matching_duplicate_threshold": 0.9 # IoU threshold for duplicate bbox detection
    }
}

# ============================================================================
# INITIALIZATION & LOADING FUNCTIONS
# ============================================================================

def load_observation_buffer(filepath: Path) -> Optional[ObsDataBuffer]:
    """
    Load the observation buffer from pickle file.
    Args:
        filepath: Path to obs_buffer.pkl file
    Returns:
        ObsDataBuffer object or None if loading fails
    """
    print(f"\n{'='*70}")
    print(f"  Loading Observation Buffer")
    print(f"{'='*70}")
    print(f"Path: {filepath}")
    
    if not filepath.exists():
        print(f"  ERROR: File not found at {filepath}")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            buffer = pickle.load(f)
        
        if not hasattr(buffer, 'entries') or not hasattr(buffer, 'static_transforms'):
            print(f"  ERROR: Invalid buffer format")
            return None
        
        print(f"  ✓ Successfully loaded buffer")
        print(f"    Total entries: {len(buffer.entries)}")
        print(f"    TF static ready: {buffer.is_tf_static_ready()}")
        
        if not buffer.is_tf_static_ready():
            print(f"  ERROR: Static transforms not complete. Cannot proceed.")
            return None
            
        # Count complete frames
        complete_frames = sum(1 for entry in buffer.entries.values() if entry.is_frame_full())
        print(f"    Complete frames: {complete_frames}/{len(buffer.entries)}")
        
        return buffer
        
    except Exception as e:
        print(f"  ERROR loading buffer: {e}")
        return None

def initialize_models(logger) :
    """
    Initialize VLM, YOLO-World, and Fast SAM models.
    Returns:
        Tuple of (VLM detector, YOLO model, FastSAM model) or (None, None, None) if initialization fails
    """
    assert torch.cuda.is_available()

    print(f"\n{'='*70} Initializing Models {'='*70}")
    print(f"\n1. Loading VLM: {VLM_MODEL_NAME}")
    vlm_detector = create_vlm_detector(
        model_type=VLM_MODEL_TYPE,
        model_name=VLM_MODEL_NAME,
        logger=logger
    )
    
    # Initialize YOLO-World
    print(f"\n2. Loading YOLO-World: {YOLO_MODEL_NAME}")
    yolo_model = YOLOWorld(YOLO_MODEL_NAME)
    yolo_model.model.to('cuda')
    
    # Initialize Fast SAM
    fastsam_model = None
    fastsam_config = FUNCTION_CONFIGS["fastsam_segmentation"]
    if fastsam_config["enabled"] and FASTSAM_AVAILABLE:
        print(f"\n3. Loading Fast SAM: {fastsam_config['model_path']}")
        try:
            fastsam_model = FastSAM(fastsam_config['model_path'])
            fastsam_model.to(fastsam_config['device'] if torch.cuda.is_available() else 'cpu')
            print(f"   ✓ Fast SAM loaded successfully (mode: {fastsam_config['mode']})")
        except Exception as e:
            print(f"   ⚠ Fast SAM failed to load: {e}, continuing without Fast SAM")
            fastsam_model = None
    else:
        if not fastsam_config["enabled"]:
            print(f"\n3. Fast SAM disabled in configuration")
        elif not FASTSAM_AVAILABLE:
            print(f"\n3. Fast SAM not available (package not installed)")
    
    return vlm_detector, yolo_model, fastsam_model

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Bbox utility functions are now imported from bbox_utils

# Mask creation functions are now imported from bbox_utils


# ============================================================================
# ROBOT POSE EXTRACTION UTILITIES
# ============================================================================

def extract_robot_pose(odometry: Dict, static_transforms: Optional[Dict] = None) -> Dict:
    """
    Extract robot position, orientation, and yaw from odometry.
    
    Args:
        odometry: Dictionary with keys "position" and "orientation"
        static_transforms: Optional static transforms for world coordinate conversion
    
    Returns:
        Dictionary with keys:
        - "robot_position": [x, y, z] in world coordinates
        - "robot_orientation": [x, y, z, w] quaternion
        - "robot_yaw": float yaw in radians
    """
    robot_orientation = odometry.get("orientation")
    robot_world_pos, robot_world_yaw = _get_robot_world_coords(odometry, static_transforms, return_yaw=True)
    robot_position = robot_world_pos.tolist()
    robot_yaw = float(robot_world_yaw)

    return {
        "robot_position": robot_position,
        "robot_orientation": [
            robot_orientation["x"],
            robot_orientation["y"],
            robot_orientation["z"],
            robot_orientation["w"]
        ],
        "robot_yaw": robot_yaw
    }


from tiamat_agent.vision_grounding.obs_data_buffer import depth_to_pointcloud

BASE_SEGMENTATION_CONFIG = {
    "depth_scale": 1000.0,
    "filter_max_height": 20.0,
    "filter_min_height": 0.06,
    "min_points": 10,
    "depth_trunc": 25.0,
    "mask_pixel_radius": 10
}

# Bbox helper functions are now imported from bbox_utils
class DummyLogger:
    def info(self, message):
        pass
    def warning(self, message):
        pass
    def error(self, message):
        pass
    def debug(self, message):
        pass
    def critical(self, message):
        pass
    def fatal(self, message):
        pass
    def trace(self, message):
        pass
    def exception(self, message):
        pass

def add_segmented_point_cloud_to_object_data(
    image_data_dict: Dict,
    config: Optional[Dict] = BASE_SEGMENTATION_CONFIG, 
    logger = DummyLogger()
) -> Dict:
    """
    Add segmented point cloud to object data dictionary.
    
    Args:
        object_data_dict: Dictionary containing:
            - 'object_pcd_segment': o3d.geometry.PointCloud
        config: Configuration dict with keys:
            - 'depth_scale': Depth conversion factor (default: 1000.0)
            - 'filter_max_height': Max height filter (default: 20.0)
            - 'filter_min_height': Min height filter (default: 0.06)
            - 'min_points': Minimum points for valid segment (default: 10)
            - 'depth_trunc': Maximum depth to consider (default: 25.0)
            - 'mask_pixel_radius': Radius for mask creation (not used here, already created)
    
    Returns:
        Dictionary keyed by object keys (e.g., "object_0", "object_1") with results:
        {
            "object_0": {
                "center_3d": [x, y, z] or None,
                "num_points": int or None,
                "avg_depth": float or None,
                "is_valid": bool,
                "extracted_segments": o3d.geometry.PointCloud or None
            },
            ...
        }
    """

    assert "depth_image" in image_data_dict, f"Depth image not found in image_data_dict: {image_data_dict.keys()}"
    assert "camera_position" in image_data_dict, f"Camera position not found in image_data_dict: {image_data_dict.keys()}"
    assert "camera_orientation" in image_data_dict, f"Camera orientation not found in image_data_dict: {image_data_dict.keys()}"
    assert "yolo_object_dict" in image_data_dict, f"yolo_object_dict not found in image_data_dict: {image_data_dict.keys()}"
    
    yolo_object_dict = image_data_dict['yolo_object_dict']
    num_objects = len(yolo_object_dict)
    if num_objects == 0: return image_data_dict# return {}

    # Generate globally unique colors using the global color index
    # Each object gets a distinct color based on its global index (current_index + local index)
    current_index = get_color_index()
    for i, (object_id, object_data) in enumerate(yolo_object_dict.items()):
        object_data["distinct_color"] = generate_distinct_color_for_index(current_index + i)
    
    # Increment color index after assigning colors to all objects in this image
    increment_color_index(num_objects)
    
    # 2. Create color-encoded mask image (only for objects with valid masks)
    H, W = image_data_dict['depth_image'].shape[:2]
    color_mask_image = np.zeros((H, W, 3), dtype=np.uint8)
    for object_id, object_data in yolo_object_dict.items():
        object_mask = object_data.get('scaled_bbox_masks')
        if object_mask is not None:
            object_mask = object_mask.astype(bool)
            color_mask_image[object_mask] = object_data['distinct_color']
    
    # # Debug: Log image sizes (rotated dimensions)
    # logger.info(f"[SIZE] {image_data_dict['image_id']}: "
    #             f"RotatedRGB={image_data_dict.get('rotated_rgb_image', np.array([])).shape[:2]}, "
    #             f"RotatedDepth={image_data_dict['depth_image'].shape[:2]}, "
    #             f"ColorMask={color_mask_image.shape[:2]}")
    
    # # Save color mask image for debugging
    # from pathlib import Path
    # mask_output_dir = Path(__file__).parent.parent.parent / "logs" / "current_run_outputs" / "offline_outputs" / "color_masks"
    # mask_output_dir.mkdir(parents=True, exist_ok=True)
    # safe_image_id = image_data_dict['image_id'].replace('/', '_')
    # mask_path = mask_output_dir / f"color_mask_{safe_image_id}.png"
    # color_mask_bgr = cv2.cvtColor(color_mask_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(str(mask_path), color_mask_bgr)
    # logger.debug(f"Saved color mask: {mask_path.name}")

    # 3. Create single point cloud using color-encoded mask
    full_pcd, _ = depth_to_pointcloud(
        depth=image_data_dict['depth_image'],
        rgb=color_mask_image,
        position=image_data_dict['camera_position'],
        quaternion_xyzw=image_data_dict['camera_orientation'],
        depth_scale=config['depth_scale'],
        clip_height=config['filter_max_height'],
        depth_trunc=config['depth_trunc']
    )
        
    # 4. Extract point cloud data
    points = np.asarray(full_pcd.points)
    colors = (np.asarray(full_pcd.colors) * 255).astype(np.uint8)
    camera_pos = np.array(image_data_dict['camera_position'])

    image_data_dict['full_pcd'] = full_pcd
    for object_id, object_data in yolo_object_dict.items():
        #object_point_mask = (colors == object_data['distinct_color']).all(axis=1)
        # Use tolerant matching to account for float precision loss (±1 RGB value)
        object_point_mask = np.all(np.abs(colors.astype(int) - np.array(object_data['distinct_color'])) <= 1, axis=1)
        object_points = points[object_point_mask]
        object_colors = colors[object_point_mask]

        logger.info(f"processing image_id: {image_data_dict['image_id']} object_id: {object_id} object name: {object_data['label']}")
        logger.info(f"      Area of Non Rotated Bounding Box: {bbox_area(np.array(object_data['bbox_xyxy'])):.2f} pixels, ")
        logger.info(f"      Area of     Rotated Bounding Box: {bbox_area(np.array(object_data['rotated_bbox_xyxy'])):.2f} pixels")
        logger.info(f"       Total image points: {len(points)} num object points: {len(object_points)}")
        logger.info(f"      Size of object mask in non-rotated bbox ratio: {np.sum(object_point_mask)} ")
        mask_size = np.sum(object_data['scaled_bbox_masks'].astype(bool)) if object_data.get('scaled_bbox_masks') is not None else 0
        logger.info(f"      Size of object mask in rotated bbox ratio: {mask_size} ")


        object_data['center_3d'] = np.mean(object_points, axis=0).tolist() if len(object_points) > 0 else None
        object_data['num_points'] = len(object_points) if len(object_points) > 0 else None
        object_data['avg_depth'] = np.mean(np.linalg.norm(object_points - camera_pos, axis=1)) if len(object_points) > 0 else None
        object_data['is_valid'] = len(object_points) >= config['min_points']
        if len(object_points) > 0:
            object_data['object_pcd_segment'] = o3d.geometry.PointCloud()
            object_data['object_pcd_segment'].points = o3d.utility.Vector3dVector(object_points)
            object_data['object_pcd_segment'].colors = o3d.utility.Vector3dVector(object_colors / 255.0)  # Normalize colors

    return image_data_dict


def _process_batch(
    image_data_dicts: List[Dict],
    vlm_detector: VLMInterface,
    yolo_model: YOLOWorld,
    fastsam_model: Optional[FastSAM],
    config_3d: Dict,
    logger = DummyLogger()
) -> Tuple[Dict, List]:
    """
    Parse a batch of RGB-Depth pairs and extract all detected objects.
    
    This function enables cross-frame batch processing where pairs can come from
    different timestamps (e.g., reordered by priority queue).
    
    Args:
        pairs_list: List of pair dictionaries, each containing:
                   - rgb_name, rgb_image, depth_name, depth_image
                   - timestamp, frame_id, odometry, static_transforms
        vlm_detector: VLM interface for semantic understanding
        yolo_model: YOLO-World model for detection
        global_object_id_counter: Starting object ID counter
        
    Returns:
        Tuple of (detected objects list, updated object_id_counter)
    """
    batch_start_time = time.time()

    if not image_data_dicts: return {}, []
    
    # Start batch timing
    logger.info(f"Processing batch of {len(image_data_dicts)} RGB-Depth pairs...")
    # ========================================================================
    # STEP 1: Prepare batch data structures (optimized: no RGB duplication)
    # ========================================================================
    image_data_by_id = {d['image_id']: d for d in image_data_dicts}
    image_by_id = {d['image_id']: d['rotated_rgb_image'] for d in image_data_dicts}
    # ========================================================================
    # STEP 2: Run VLM on all RGB images
    # ========================================================================
    logger.info(f"Running VLM on {len(image_data_by_id)} images (batch mode)...")
    object_list_by_id = v_time_fn("VLM", vlm_detector.detect_objects, image_by_id)

    # get global vocabulary and description from all image data
    total_vlm_detections = 0
    global_vocabulary , global_vocabulary_desc = [], {}
    for i_key, object_list in object_list_by_id.items():
        image_data_by_id[i_key]["vlm_obj_data_list"] = object_list
        image_data_by_id[i_key]["vlm_object_name_list"] = [obj.get("object_name", "") for obj in object_list]
        total_vlm_detections += len(object_list)
        for obj in object_list:
            global_vocabulary.append(obj.get("object_name", ""))
            global_vocabulary_desc[obj.get("object_name", "")] = obj.get("description", "")
    global_vocabulary = sorted(set(global_vocabulary))
    global_vocabulary_desc = {obj: global_vocabulary_desc[obj] for obj in global_vocabulary}
    if not global_vocabulary:
        logger.warning(f"        ⚠ No vocabulary extracted from VLM, skipping batch")
        return {}, []
    logger.info(f"         Vocabulary: {len(global_vocabulary)} unique objects")
    logger.info(f"           {', '.join(global_vocabulary[:8])}" + (f", ... (+{len(global_vocabulary)-8} more)" if len(global_vocabulary) > 8 else ""))
    print(f"[DETECTION] VLM - detected {total_vlm_detections} objects")


    
    # ========================================================================
    # STEP 4: Run YOLO Detection on all RGB images
    # ========================================================================
    print(f"         Running YOLO-World detection (batch mode)...")
    yolo_model.set_classes(global_vocabulary)
    yolo_config = FUNCTION_CONFIGS["yolo_detection"]
    yolo_result_batch = v_time_fn("YOLO", yolo_model.predict,
        [d['rotated_rgb_image'] for d in image_data_dicts],
        conf=yolo_config["conf_threshold"],
        iou=yolo_config["iou_threshold"],
        verbose=False
    )

    # ========================================================================
    # STEP 4.5: Extract rotated bboxes from YOLO results (needed for FastSAM)
    # ========================================================================
    for image_id, yolo_result in zip(image_data_by_id.keys(), yolo_result_batch):
        detections = sv.Detections.from_ultralytics(yolo_result)
        image_data_by_id[image_id]["yolo_rotated_detections_xyxy"] = detections.xyxy

    # ========================================================================
    # STEP 5: Run Fast SAM Segmentation & Match to YOLO Boxes
    # ========================================================================
    fastsam_total_start_time = time.time()
    fastsam_config = FUNCTION_CONFIGS["fastsam_segmentation"]
    total_fastsam_masks = 0

    if fastsam_model is not None and fastsam_config["enabled"]:
        print(f"         Running Fast SAM segmentation...")
        
        for image_id, image_data in image_data_by_id.items():
            rotated_bboxes = image_data.get("yolo_rotated_detections_xyxy", [])
            
            # Check #TODO: remove this after testing
            if len(rotated_bboxes) == 0:
                image_data["raw_sam_rotated_detections"] = None
                image_data["raw_sam_unrotated_detections"] = None
                image_data["fastsam_matches"] = []
                image_data["fastsam_unmatched_bbox_indices"] = []
                continue
            
            # Convert to numpy arrays
            rotated_bboxes_array = [np.array(bbox) for bbox in rotated_bboxes]
            
            # Single call with all bboxes (box_prompt mode)
            fastsam_masks = get_fastsam_masks_from_boxes(
                fastsam_model=fastsam_model,
                image=image_data["rotated_rgb_image"],
                bboxes=rotated_bboxes_array,
                device=fastsam_config["device"],
                config=fastsam_config
            )
            
            # Store rotated masks
            image_data["raw_sam_rotated_detections"] = fastsam_masks
            total_fastsam_masks += len(fastsam_masks)
            
            # Unrotate masks for final use
            backward_rot = image_data.get("rotation_map", {}).get("backward", None)
            #TODO: remove "if" this after testing
            if backward_rot is not None:
                unrotated_masks = [cv2.rotate(m.astype(np.uint8), backward_rot).astype(bool) for m in fastsam_masks]
            else:
                unrotated_masks = [m.astype(bool) for m in fastsam_masks]
            image_data["raw_sam_unrotated_detections"] = unrotated_masks
            
            # Match masks to bboxes using helper function (rotated coordinates for matching)
            matching_config = {
                "iou_threshold": fastsam_config.get("matching_iou_threshold", 0.3),
                "duplicate_threshold": fastsam_config.get("matching_duplicate_threshold", 0.9)
            }
            matches, unmatched_indices = match_masks_to_bboxes(
                bboxes=rotated_bboxes_array,
                masks=fastsam_masks,  # Use rotated masks for matching
                config=matching_config
            )
            
            # Store matching results
            image_data["fastsam_matches"] = matches
            image_data["fastsam_unmatched_bbox_indices"] = unmatched_indices
            
            # Summary logging per image
            num_matched = len(matches)
            num_unmatched = len(unmatched_indices)
            total_bboxes = len(rotated_bboxes_array)
            
            if num_unmatched > 0:
                logger.warning(
                    f"[{image_id}] FastSAM matching: {num_matched}/{total_bboxes} matched, "
                    f"{num_unmatched} unmatched (bbox indices: {unmatched_indices})"
                )
            else:
                logger.info(
                    f"[{image_id}] FastSAM matching: {num_matched}/{total_bboxes} matched ✓"
                )
                

    # ========================================================================
    # DEBUG: Save FastSAM results for visualization
    # ========================================================================
    if fastsam_model is not None and FUNCTION_CONFIGS["fastsam_segmentation"]["enabled"]:
        debug_output_dir = PROJECT_ROOT / "logs" / "current_run_outputs" / "offline_outputs" / "debug_sam_output"
        _debug_dump_fastsam_results(image_data_by_id, debug_output_dir, logger)
        

    total_yolo_detections = 0
    total_segmented_pcd_objects = 0
    
    for i_key, yolo_result in zip(image_data_by_id.keys(), yolo_result_batch):
        curr_image_data = image_data_by_id[i_key]
        
        # ------------------------------------------------------------
        # Get Camera and Robot Pose
        # ------------------------------------------------------------
        assert 'w2c_transform' in curr_image_data, f"w2c_transform not found in curr_image_data: {curr_image_data.keys()}"
        assert 'camera_position' in curr_image_data, f"camera_position not found in curr_image_data: {curr_image_data.keys()}"
        assert 'camera_orientation' in curr_image_data, f"camera_orientation not found in curr_image_data: {curr_image_data.keys()}"
        assert 'robot_position' in curr_image_data, f"robot_position not found in curr_image_data: {curr_image_data.keys()}"
        assert 'robot_orientation' in curr_image_data, f"robot_orientation not found in curr_image_data: {curr_image_data.keys()}"
        assert 'robot_yaw' in curr_image_data, f"robot_yaw not found in curr_image_data: {curr_image_data.keys()}"
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # Get YOLO Detections
        # Get backward rotation from rotation_map
        # ------------------------------------------------------------
        curr_image_data["raw_yolo_rotated_result"] = yolo_result  # Store full result for later use
        curr_image_data["raw_yolo_rotated_detections"] = sv.Detections.from_ultralytics(yolo_result)
        
        # ------------------------------------------------------------------------
        # Process YOLO Detections
        # ------------------------------------------------------------------------
        #curr_image_data["yolo_rotated_detections_xyxy"] = curr_image_data["raw_yolo_rotated_detections"].xyxy
        curr_image_data["yolo_detections_xyxy"] = unrotate_bboxes(
            curr_image_data["yolo_rotated_detections_xyxy"], 
            curr_image_data['rotated_rgb_image'].shape[:2],  # Get (H, W) shape
            curr_image_data.get('rotation_map', {'backward': None})['backward']
        )
        curr_image_data["yolo_detections_class_ids"] = list(map(int, curr_image_data["raw_yolo_rotated_detections"].class_id))
        curr_image_data["yolo_detections_confidences"] = list(map(float, curr_image_data["raw_yolo_rotated_detections"].confidence))
        
        # Count YOLO detections for this image
        num_detections = len(curr_image_data["yolo_detections_xyxy"])
        total_yolo_detections += num_detections
        
       
        curr_image_data["yolo_object_dict"] = {}
        # Usage
        # image_data_by_id[image_key]["yolo_object_dict"][object_key]["bbox_xyxy"] = [x1, y1, x2, y2]
        for i in range(len(curr_image_data["yolo_detections_xyxy"])):
            curr_image_data["yolo_object_dict"][f"object_{i}"] = {}
            object_data_dict = curr_image_data["yolo_object_dict"][f"object_{i}"]
            object_data_dict["class_id"] = curr_image_data["yolo_detections_class_ids"][i]
            object_data_dict["label"] = global_vocabulary[object_data_dict["class_id"]]
            object_data_dict["description"] = global_vocabulary_desc[object_data_dict["label"]]
            object_data_dict["bbox_xyxy"] = list(map(float, curr_image_data["yolo_detections_xyxy"][i]))
            object_data_dict["rotated_bbox_xyxy"] = list(map(float, curr_image_data["yolo_rotated_detections_xyxy"][i]))
            object_data_dict["confidence"] = curr_image_data["yolo_detections_confidences"][i]

            # Scale bbox from RGB resolution to depth resolution
            object_data_dict["scaled_bbox_xyxy"]  = scale_bbox(
                object_data_dict["bbox_xyxy"],
                original_shape=curr_image_data["rgb_image"].shape[:2], 
                target_shape=curr_image_data["depth_image"].shape[:2]
            ).tolist()
            
            
            # Create mask - use Fast SAM if matched, otherwise None (unmatched boxes are ignored)
            sam_matches = curr_image_data.get("fastsam_matches", [])
            sam_unmatched = set(curr_image_data.get("fastsam_unmatched_bbox_indices", []))
            sam_masks_unrotated = curr_image_data.get("raw_sam_unrotated_detections")
            
            # Check if this bbox is matched
            matched_mask_idx = None
            if i not in sam_unmatched and sam_matches:
                # Find the match for this bbox index
                for bbox_idx, mask_idx, iou in sam_matches:
                    if bbox_idx == i:
                        matched_mask_idx = mask_idx
                        break
            
            if matched_mask_idx is not None and sam_masks_unrotated is not None:
                # Use matched FastSAM mask (unrotated for final use)
                fastsam_mask = sam_masks_unrotated[matched_mask_idx]
                depth_shape = curr_image_data["depth_image"].shape[:2]
                
                # Resize if needed
                if fastsam_mask.shape != depth_shape:
                    fastsam_mask = cv2.resize(
                        fastsam_mask.astype(np.uint8),
                        (depth_shape[1], depth_shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                
                object_data_dict["scaled_bbox_masks"] = fastsam_mask
            else:
                # # Fallback to center mask
                # object_data_dict["scaled_bbox_masks"] = create_mask_from_bbox_center(
                #     np.array(object_data_dict["scaled_bbox_xyxy"]),
                #     curr_image_data["depth_image"].shape[:2], 
                #     config_3d["mask_pixel_radius"]
                # )
                # if i == 0:
                #     logger.info(f"Using fallback center masks (FastSAM not available)")

                # Unmatched: set to None (will be ignored in point cloud generation)
                object_data_dict["scaled_bbox_masks"] = None

            # # Create full bbox mask
            # object_data_dict["scaled_bbox_masks_full"] = create_mask_from_bbox(
            #     np.array(object_data_dict["scaled_bbox_xyxy"]),
            #     curr_image_data["depth_image"].shape[:2]
            # )
        
        # Process all objects at once with create_segmented_point_cloud
        curr_image_data = add_segmented_point_cloud_to_object_data(
            curr_image_data, 
            config=config_3d, 
            logger=logger
        )


        #Print real-time summary
        # Count valid segmented point cloud objects for this image
        if "yolo_object_dict" in curr_image_data:
            for object_id, object_data in curr_image_data["yolo_object_dict"].items():
                if object_data.get('is_valid', False):
                    total_segmented_pcd_objects += 1

    # Print YOLO detection summary
    print(f"[DETECTION] YOLO - detected {total_yolo_detections} objects")
    
    # Print FastSAM timing and detection summary
    if fastsam_model is not None and FUNCTION_CONFIGS["fastsam_segmentation"]["enabled"]:
        fastsam_total_elapsed = time.time() - fastsam_total_start_time
        print(f"[TIMING] FastSAM - detect_objects took {fastsam_total_elapsed:.5f}s")
        print(f"[DETECTION] FastSAM - detected {total_fastsam_masks} objects")
    
    # Print segmented point cloud detection summary
    print(f"[DETECTION] segmented-PCD - detected {total_segmented_pcd_objects} objects")
    
    # Debug Helper Run this command here in debug mode. #####
    # from tiamat_agent.vision_grounding.debug_scripts import *
    # visualize_all_points(image_data_by_id)
    # visualize_yolo_detections_in_depth(image_data_by_id)
    # visualize_all_images_yolo_detections_in_rotated(image_data_by_id)
    # visualize_all_images_yolo_detections_in_non_rotated(image_data_by_id)
    # ------------------------------------------------------------------------

    
    object_records = []
    for image_id, image_data in image_data_by_id.items():
        for object_id, object_data in image_data["yolo_object_dict"].items():
            if not object_data['is_valid']:
                continue
            # print(f"         Processing object {object_id} in image {image_id}")
            obj_record = {
                "object_id": f"{image_id}_bbox_{object_id}",
                
                # Frame metadata
                "frame_metadata": {
                    "frame_id": image_id,
                    "timestamp": image_data['timestamp'],
                    "camera_name": image_data['rgb_name'],
                    "camera_position": image_data['camera_position'],
                    "camera_orientation": image_data['camera_orientation'],
                    # Robot base information (from cache - calculated once per frame)
                    "robot_position": image_data['robot_position'],
                    "robot_orientation": image_data['robot_orientation'],
                    "robot_yaw": image_data['robot_yaw']
                },
                
                # Semantic metadata
                "semantic_metadata": {
                    "label": object_data['label'],
                    "description": object_data['description'],
                    "scene_description": None,
                    "room_description": None,
                    "category": object_data['label'],
                    "vlm_confidence": object_data['confidence']
                },
                
                # Detection metadata
                "detection_metadata": {
                    "bbox_2d": object_data['bbox_xyxy'],
                    "bbox_center_pixel": bbox_center_pixel(np.array(object_data['bbox_xyxy'])),
                    "bbox_width": bbox_width(np.array(object_data['bbox_xyxy'])),
                    "bbox_height": bbox_height(np.array(object_data['bbox_xyxy'])),
                    "bbox_area": bbox_area(np.array(object_data['bbox_xyxy'])),
                    "yolo_score": object_data['confidence'],
                    "detection_method": "yolo_world_vlm_batch",
                    "distinct_color_rgb": list(object_data['distinct_color']) if 'distinct_color' in object_data else None
                },

                "rotated_detection_metadata": {
                    "bbox_2d": object_data['rotated_bbox_xyxy'],
                    "bbox_center_pixel": bbox_center_pixel(np.array(object_data['rotated_bbox_xyxy'])),
                    "bbox_width": bbox_width(np.array(object_data['rotated_bbox_xyxy'])),
                    "bbox_height": bbox_height(np.array(object_data['rotated_bbox_xyxy'])),
                    "bbox_area": bbox_area(np.array(object_data['rotated_bbox_xyxy'])),
                    "yolo_score": object_data['confidence'],
                    "detection_method": "yolo_world_vlm_batch"
                },
                
                # Spatial metadata
                "spatial_metadata": {
                    "position_3d": object_data['center_3d'],
                    "depth_value": float(object_data['avg_depth'] * config_3d["depth_scale"]),
                    "depth_meters": float(object_data['avg_depth']),
                    "is_valid_depth": object_data['is_valid'],
                    "height_from_ground": float(object_data['center_3d'][2]),
                    "num_points_used": object_data['num_points']
                },
                
                # Placeholders for future features
                "embeddings": {
                    "text_embedding": None,
                    "visual_embedding": None
                },
                
                "relationships": {
                    "nearby_objects": [],
                    "on_surface": None,
                    "in_container": None
                }
            }
            
            object_records.append(obj_record)
    
    # Debug Helper Run this command here in debug mode. #####
    # visualize_object_records_top_down(object_records)
    # ##############################################################

    # Calculate batch timing
    batch_total_time = time.time() - batch_start_time
    num_pairs = len(image_data_by_id)
    avg_time_per_image = batch_total_time / num_pairs if num_pairs > 0 else 0.0
    
    print(f"      Batch summary: {len(object_records)} total detections")
    print(f"      Batch timing: {batch_total_time:.2f}s total, {avg_time_per_image:.3f}s per image")
    logger.info(f"      Batch summary: {len(object_records)} total detections")
    logger.info(f"      Batch timing: {batch_total_time:.2f}s total, {avg_time_per_image:.3f}s per image")
    
    return image_data_by_id, object_records



def _debug_dump_fastsam_results(image_data_by_id: Dict, output_dir: Path, logger=None):
    """
    Debug helper: Save FastSAM segmentation visualization with matching results.
    Uses _visualize_benchmark_result from fast_sam_helper2.py for better visualization.
    
    Args:
        image_data_by_id: Dictionary with image data including rotated_rgb_image and sam masks
        output_dir: Path to save debug outputs
        logger: Optional logger for info messages
    """
    from tiamat_agent.vision_grounding.fast_sam_helper2 import _visualize_benchmark_result
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for image_id, image_data in image_data_by_id.items():
        # Skip if no FastSAM results
        if image_data.get("raw_sam_rotated_detections") is None:
            continue
        
        masks = image_data["raw_sam_rotated_detections"]
        bboxes = image_data.get("yolo_rotated_detections_xyxy", [])
        
        if len(masks) == 0 or len(bboxes) == 0:
            continue
        
        # Create safe filename
        safe_image_id = image_id.replace('/', '_').replace('.', '_')
        output_path = output_dir / f"{safe_image_id}_matching.png"
        
        # Call the visualization function
        _visualize_benchmark_result(
            image=image_data["rotated_rgb_image"],
            bboxes=bboxes,
            masks=masks,
            output_path=output_path
        )
        
        if logger:
            logger.info(f"[DEBUG SAM] Saved visualization for {image_id}")




def split_input_to_max_batch_size(
    items: List[Any],
    max_batch_size: int
) -> List[List[Any]]:
    """
    Split a list of items into batches of at most max_batch_size.
    
    Args:
        items: List of items to split into batches
        max_batch_size: Maximum size of each batch
        
    Returns:
        List of batches, each containing at most max_batch_size items
    """
    if not items:
        return []
    
    if len(items) <= max_batch_size:
        return [items]
    
    batches = []
    for i in range(0, len(items), max_batch_size):
        batches.append(items[i:i + max_batch_size])
    
    return batches


def parse_rgbd_image_dicts_for_objects(
    rgbd_image_dicts: List[Dict],
    vlm_detector: VLMInterface,
    yolo_model: YOLOWorld,
    fastsam_model: Optional[FastSAM] = None,
    max_batch_size: int = 20,
    config_3d: Dict = BASE_SEGMENTATION_CONFIG,
    logger = DummyLogger()
) -> Tuple[Dict, List]:
    """
    Parse pairs and extract all detected objects with internal batching.
    
    This is the unified processing function that handles batching internally.
    Works with pairs from any source (frame mode, priority mode, or real-time).
    
    Args:
        rgbd_image_dicts: List of pair dictionaries, each containing:
                   - rgb_name, rgb_image, depth_name, depth_image
                   - timestamp, frame_id, odometry, static_transforms
        vlm_detector: VLM interface for semantic understanding
        yolo_model: YOLO-World model for detection
        fastsam_model: Optional Fast SAM model for segmentation
        max_batch_size: Number of pairs to process per batch (default: 20)
        config_3d: Configuration dictionary for 3D processing
        
    Returns:
        Tuple of (image_data_by_id dict, object list)
    """
    if not rgbd_image_dicts:
        return {}, []
    
    batches = split_input_to_max_batch_size(rgbd_image_dicts, max_batch_size)
    
    if len(batches) > 1:
        logger.info(f"\n  Processing {len(rgbd_image_dicts)} pairs in {len(batches)} batches (batch size: {max_batch_size})")
    
    all_objects = []
    image_data_by_id = {}
    
    for batch_idx, batch_rgbd_image_dicts in enumerate(batches):
        if len(batches) > 1:
            start_idx = batch_idx * max_batch_size
            end_idx = min(start_idx + max_batch_size, len(rgbd_image_dicts))
            logger.info(f"  Batch {batch_idx + 1}/{len(batches)} | Pairs: {start_idx} - {end_idx - 1} ({len(batch_rgbd_image_dicts)} pairs)")
            logger.info(f"    {'-'*66}")
        
        tmp_image_data_by_id, batch_objects = _process_batch(
            image_data_dicts=batch_rgbd_image_dicts,
            vlm_detector=vlm_detector,
            yolo_model=yolo_model,
            fastsam_model=fastsam_model,
            config_3d=config_3d,
            logger=logger
        )
        
        image_data_by_id.update(tmp_image_data_by_id)
        all_objects.extend(batch_objects)
        
    
    return image_data_by_id, all_objects


# ============================================================================
# POINT CLOUD (SCENE RECONSTRUCTION) & OUTPUT FUNCTIONS
# ============================================================================

def dump_object_list(
    object_list: List[Dict],
    output_path: Path,
    metadata: Dict
):
    """
    Save object list to JSON file with metadata.
    
    Args:
        object_list: List of detected object dictionaries
        output_path: Path to save JSON file
        metadata: Processing metadata to include
    """
    statistics = calculate_statistics(object_list)
    
    output_data = {
        "metadata": metadata,
        "objects": object_list,
        "statistics": statistics
    }
    
    print(f"\n  Saving objects database...")
    print(f"    Path: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    ✓ Saved {len(object_list)} objects ({file_size_mb:.2f} MB)")
    
    return statistics

# uses get_pointcloud_for_entry form obs_data_buffer.py
def build_scene_pointcloud(
    obs_buffer: ObsDataBuffer,
    complete_entries: List[Tuple[str, ObsDataEntry]],
    voxel_size: float
) -> o3d.geometry.PointCloud:
    """
    Build complete scene point cloud from all entries.
    
    Args:
        obs_buffer: ObsDataBuffer with static transforms
        complete_entries: List of (timestamp, entry) tuples
        voxel_size: Voxel size for downsampling
        
    Returns:
        Combined and downsampled point cloud
    """
    print(f"\n{'='*70}")
    print(f"  Building Scene Point Cloud")
    print(f"{'='*70}")
    
    all_pcds = []
    
    for timestamp, entry in complete_entries:
        try:
            # Use obs_buffer's built-in method
            # Returns tuple: (List[PointCloud], List[camera_positions])
            entry_pcds, _ = obs_buffer.get_pointcloud_for_entry(timestamp)
            all_pcds.extend(entry_pcds)
        except Exception as e:
            print(f"  ⚠ Error getting point cloud for {timestamp}: {e}")
            continue
    
    if not all_pcds:
        print(f"  ⚠ No point clouds generated")
        return o3d.geometry.PointCloud()
    
    print(f"  Combining {len(all_pcds)} point clouds...")
    
    # Combine all point clouds
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in all_pcds:
        combined_pcd += pcd
    
    print(f"    Total points before downsampling: {len(combined_pcd.points):,}")
    
    # Downsample
    if len(combined_pcd.points) > 0:
        downsampled = combined_pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"    Points after downsampling: {len(downsampled.points):,}")
        print(f"    Voxel size: {voxel_size}m")
        return downsampled
    
    return combined_pcd

def save_segmented_pointcloud(
    image_data_by_id: Dict,
    output_path: Optional[Path] = None
) -> None:
    """
    Save color-encoded segmented point cloud from all processed images.
    
    Args:
        image_data_by_id: Dictionary mapping image_id to image_data (with 'full_pcd')
        output_path: Optional path to save the file (defaults to logs/current_run_outputs/offline_outputs/)
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "logs" / "current_run_outputs" / "offline_outputs" / "objects_segmented_pc.ply"
    
    print(f"\n{'='*70}")
    print(f"  Saving Color-Encoded Segmented Point Cloud")
    print(f"{'='*70}")
    
    all_segmented_pcds = []
    for image_id, image_data in image_data_by_id.items():
        if 'full_pcd' in image_data and image_data['full_pcd'] is not None:
            all_segmented_pcds.append(image_data['full_pcd'])
    
    if not all_segmented_pcds:
        print(f"  ⚠ No segmented point clouds to save")
        return
    
    print(f"  Combining {len(all_segmented_pcds)} image point clouds...")
    
    combined_segmented = o3d.geometry.PointCloud()
    for pcd in all_segmented_pcds:
        combined_segmented += pcd
    
    print(f"  Total segmented points: {len(combined_segmented.points):,}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), combined_segmented)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved to {output_path} ({file_size_mb:.2f} MB)")
    print(f"{'='*70}")


def save_individual_object_pointclouds(
    image_data_by_id: Dict,
    output_dir: Optional[Path] = None
) -> None:
    """
    Save individual object point clouds as separate PLY files.
    Uses the same object_id pattern as JSON (image_id + bbox id),
    so filenames can be matched directly to JSON records.
    
    Only saves objects that have:
    - FastSAM mask match (scaled_bbox_masks != None)
    - Valid point cloud (object_pcd_segment exists with points)
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "logs" / "current_run_outputs" / "offline_outputs" / "individual_objects"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  Saving Individual Object Point Clouds")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    
    saved_count = 0
    skipped_count = 0
    
    # Iterate through image_data to access object_pcd_segment
    for image_id, image_data in image_data_by_id.items():
        yolo_object_dict = image_data.get("yolo_object_dict", {})
        
        for local_object_id, object_data in yolo_object_dict.items():
            # Construct the full object_id (same format as JSON)
            full_object_id = f"{image_id}_bbox_{local_object_id}"
            
            # Get point cloud segment (only exists if mask matched and points found)
            object_pcd = object_data.get('object_pcd_segment')
            
            if object_pcd is None or len(object_pcd.points) == 0:
                skipped_count += 1
                continue
            
            # Sanitize filename (remove special characters)
            safe_object_id = full_object_id.replace('/', '_').replace(':', '_').replace('\\', '_')
            filename = f"{safe_object_id}.ply"
            filepath = output_dir / filename
            
            # Save point cloud (includes unique color from distinct_color)
            o3d.io.write_point_cloud(str(filepath), object_pcd)
            saved_count += 1
    
    print(f"\n  ✓ Saved {saved_count} object point clouds")
    if skipped_count > 0:
        print(f"  ⚠ Skipped {skipped_count} objects (no point cloud or unmatched)")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*70}")

def calculate_statistics(all_objects: List[Dict]) -> Dict:
    """Calculate statistics from detected objects."""
    if not all_objects:
        return {}
    
    # Count by label
    label_counts = defaultdict(int)
    for obj in all_objects:
        label = obj["semantic_metadata"]["label"]
        label_counts[label] += 1
    
    # Sort by count
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Count by camera
    camera_counts = defaultdict(int)
    for obj in all_objects:
        camera = obj["frame_metadata"]["camera_name"]
        camera_counts[camera] += 1
    
    # Depth statistics
    depths = [obj["spatial_metadata"]["depth_meters"] for obj in all_objects]
    yolo_scores = [obj["detection_metadata"]["yolo_score"] for obj in all_objects]
    
    return {
        "total_unique_labels": len(label_counts),
        "objects_by_label": dict(sorted_labels),
        "objects_by_camera": dict(camera_counts),
        "average_yolo_score": float(np.mean(yolo_scores)) if yolo_scores else 0,
        "average_depth": float(np.mean(depths)) if depths else 0,
        "min_depth": float(np.min(depths)) if depths else 0,
        "max_depth": float(np.max(depths)) if depths else 0
    }

def save_pointcloud(scene_pcd: o3d.geometry.PointCloud, output_path: Path):
    """
    Save point cloud to PLY file.
    
    Args:
        scene_pcd: Point cloud to save
        output_path: Path to save PLY file
    """
    print(f"\n  Saving scene point cloud...")
    print(f"    Path: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    o3d.io.write_point_cloud(str(output_path), scene_pcd)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    ✓ Saved {len(scene_pcd.points):,} points ({file_size_mb:.2f} MB)")

def print_final_summary(
    total_frames: int,
    processed_frames: int,
    total_objects: int,
    total_time: float,
    statistics: Dict
):
    """Print comprehensive final summary of processing run."""
    print(f"\n{'='*70}")
    print(f"✨ Processing Complete!")
    print(f"{'='*70}")
    
    print(f"\n  Processing Summary:")
    print(f"    Total frames in buffer:    {total_frames}")
    print(f"    Frames processed:          {processed_frames}")
    print(f"    Total objects detected:    {total_objects}")
    print(f"    Valid 3D localizations:    {total_objects}")
    print(f"    Total processing time:     {total_time:.2f}s ({total_time/60:.2f} min)")
    
    if processed_frames > 0:
        print(f"    Average time per frame:    {total_time/processed_frames:.2f}s")
    if total_objects > 0:
        print(f"    Average objects per frame: {total_objects/processed_frames:.2f}")
    
    if statistics:
        print(f"\n  Detection Statistics:")
        print(f"    Unique object labels:      {statistics.get('total_unique_labels', 0)}")
        print(f"    Average YOLO confidence:   {statistics.get('average_yolo_score', 0):.3f}")
        print(f"    Average depth:             {statistics.get('average_depth', 0):.2f}m")
        print(f"    Depth range:               {statistics.get('min_depth', 0):.2f}m - {statistics.get('max_depth', 0):.2f}m")
        
        print(f"\n  Top 10 Detected Objects:")
        for i, (label, count) in enumerate(list(statistics.get('objects_by_label', {}).items())[:10], 1):
            print(f"       {i:2d}. {label:25s} {count:4d} instances")
        
        print(f"\n  Detections by Camera:")
        for camera, count in statistics.get('objects_by_camera', {}).items():
            print(f"       {camera:25s} {count:4d} detections")
    
    print(f"\n{'='*70}")

def save_log_json(statistics: Dict, output_path: Path):
    """
    Save statistics to log.json file.
    
    Args:
        statistics: Statistics dictionary from calculate_statistics()
        output_path: Path to save log.json file
    """
    if not statistics:
        return
    
    # Format statistics for log file
    log_data = {
        "objects_by_camera": statistics.get("objects_by_camera", {}),
        "average_yolo_score": statistics.get("average_yolo_score", 0),
        "average_depth": statistics.get("average_depth", 0),
        "min_depth": statistics.get("min_depth", 0),
        "max_depth": statistics.get("max_depth", 0),
        "detection_statistics": {
            "total_unique_labels": statistics.get("total_unique_labels", 0),
            "objects_by_label": statistics.get("objects_by_label", {})
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"    ✓ Saved log to {output_path}")

# ============================================================================
# ------------------------------ MAIN EXECUTION ----------------------------
# Current architecture: Pair-based processing
#      1. prepare_pairs() - Extract RGB-depth pairs from buffer
#      2. parse_rgbd_image_dicts_for_objects() - Process pairs in batches
#      3. dump_object_list() - Save results
# ============================================================================

def main():
    """
    Main execution pipeline - UNIFIED PAIR-BASED VERSION.
    
    NEW STRUCTURE:
    1. Load observation buffer
    2. Initialize models
    3. Prepare pairs (frame mode or priority mode)
    4. Process pairs in batches (unified function with internal batching)
    5. Build scene point cloud
    6. Save outputs (objects JSON + point cloud PLY)
    7. Print summary
    """
    print(f"\n{'#'*70}")
    print(f"#{'OBJECT DETECTION & 3D LOCALIZATION PIPELINE - REFACTORED'.center(68)}#")
    print(f"#{'VLM + YOLO-World + Depth Fusion (Batch Processing)'.center(68)}#")
    print(f"{'#'*70}\n")
    
    start_time = time.time()
    
    # ========================================================================
    # STAGE 1: Load Observation Buffer
    # ========================================================================
    buffer = load_observation_buffer(INPUT_BUFFER_PATH)
    if buffer is None:
        print("\n  Failed to load observation buffer. Exiting.")
        return
    
    # ========================================================================
    # STAGE 2: Initialize Models
    # ========================================================================
    logger = DummyLogger()  # Create logger for initialization
    vlm_detector, yolo_model, fastsam_model = initialize_models(logger)
    if vlm_detector is None or yolo_model is None:
        print("\n  Failed to initialize models. Exiting.")
        return
    
    # Reset color index for offline processing
    reset_color_index(0)
    
    # ========================================================================
    # STAGE 3: Prepare Pairs - SUPPORTS BOTH MODES
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"  Preparing Pairs for Processing")
    print(f"{'='*70}")
    
    # Get complete entries
    complete_entries = [
        (stamp, entry) for stamp, entry in buffer.entries.items()
        if entry.is_frame_full()
    ]
    
    total_frames = len(complete_entries)
    
    # Calculate processing limits based on MAX_IMAGES_TO_PROCESS
    if MAX_IMAGES_TO_PROCESS is not None:
        # Estimate frames needed to get desired number of images
        avg_images_per_frame = 5  # Typical: 5 cameras per frame
        estimated_frames_needed = (MAX_IMAGES_TO_PROCESS + avg_images_per_frame - 1) // avg_images_per_frame
        frames_to_process = min(estimated_frames_needed, total_frames)
        max_images_limit = MAX_IMAGES_TO_PROCESS
    else:
        frames_to_process = total_frames
        max_images_limit = None
    
    print(f"Total complete frames: {total_frames}")
    print(f"Frames to process: {frames_to_process}")
    if MAX_IMAGES_TO_PROCESS is not None:
        print(f"Target images to process: {MAX_IMAGES_TO_PROCESS}")
    
    # ========================================================================
    # STAGE 3: Prepare Pairs Based on Mode
    # ========================================================================

    # MODE 1: Frame-based processing (from PKL buffer)
    if BATCH_SIZE is None:
        print(f"Processing mode: Frame-by-frame")
    else:
        print(f"Processing mode: Cross-frame batching (batch size = {BATCH_SIZE} pairs)")
    if max_images_limit is not None:
        print(f"Limiting to {max_images_limit} images")
    print()
    
    pairs_list, processed_entries = prepare_pairs_frame_mode(
        buffer=buffer,
        complete_entries=complete_entries[:frames_to_process],
        max_images=max_images_limit
    )

    # Mark processed entries
    for timestamp, entry in processed_entries:
        buffer.mark_processed(timestamp)
    
    if not pairs_list:
        print("  ⚠ No pairs to process. Exiting.")
        return
    
    # ========================================================================
    # STAGE 4: Process Pairs in Batches
    # ========================================================================
    batch_size = BATCH_SIZE if BATCH_SIZE is not None else 20
    image_data_by_id, object_list = parse_rgbd_image_dicts_for_objects(
        rgbd_image_dicts=pairs_list,
        vlm_detector=vlm_detector,
        yolo_model=yolo_model,
        fastsam_model=fastsam_model,
        max_batch_size=batch_size
    )
    
    # ========================================================================
    # STAGE 5: Build Scene Point Cloud
    # ========================================================================
    pcd_config = FUNCTION_CONFIGS["pointcloud_generation"]
    scene_pcd = build_scene_pointcloud(
        buffer, 
        processed_entries,  # Use only entries that were actually processed
        pcd_config["voxel_size"]
    )
    
    # ========================================================================
    # STAGE 6: Prepare Metadata
    # ========================================================================
    # Calculate total pairs processed and determine processing mode
    processing_mode = "frame_based"
    batch_size_used = BATCH_SIZE if BATCH_SIZE is not None else 20
    total_pairs_processed = len(pairs_list)
    prioritized_source = None
    
    processing_metadata = {
        "total_frames_processed": frames_to_process,
        "total_objects_detected": len(object_list),
        "vlm_model": VLM_MODEL_NAME,
        "yolo_model": YOLO_MODEL_NAME,
        "processing_timestamp": datetime.now().isoformat(),
        "depth_scale": pcd_config["depth_scale"],
        "filter_settings": FUNCTION_CONFIGS["extract_3d_position"],
        "scene_settings": pcd_config,
        "batch_processing": {
            "enabled": True,
            "mode": processing_mode,
            "batch_size": batch_size_used,
            "total_pairs_processed": total_pairs_processed,
            "prioritized_order_enabled": False,
            "prioritized_json_source": prioritized_source
        },
        "color_encoding_method": True
    }
    
    # ========================================================================
    # STAGE 7: Save Outputs - Using Clean Functions
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"  Saving Outputs")
    print(f"{'='*70}")
    
    statistics = dump_object_list(object_list, OUTPUT_OBJECTS_PATH, processing_metadata)
    save_pointcloud(scene_pcd, OUTPUT_POINTCLOUD_PATH)
    
    # ========================================================================
    # STAGE 8: Print Final Summary
    # ========================================================================
    total_time = time.time() - start_time
    
    print_final_summary(
        total_frames=total_frames,
        processed_frames=frames_to_process,
        total_objects=len(object_list),
        total_time=total_time,
        statistics=statistics
    )
    
    # Save log.json
    log_path = SCRIPT_DIR / "data" / "log.json"
    save_log_json(statistics, log_path)
    
    print(f"\n  Pipeline completed successfully!")
    print(f"\n  Output files:")
    print(f"    1. {OUTPUT_OBJECTS_PATH}")
    print(f"    2. {OUTPUT_POINTCLOUD_PATH}")
    print(f"    3. {PROJECT_ROOT / 'logs' / 'current_run_outputs' / 'offline_outputs' / 'objects_segmented_pc.ply'}")
    print(f"    4. {log_path}")
    print(f"\nReady for semantic search and scene understanding! 🚀\n")


if __name__ == "__main__":
    main()