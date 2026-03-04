#!/usr/bin/env python3
"""
Visualize images with bounding boxes and labels from offline_outputs (OpenCV version).
Highlights selected objects from object_selection_results.json with different colors.

Steps to prepare data:
             1. set SEED_OBJECT_LIBRARY = False / dump_entry = True (in z_sensor_object_map_node.py) / Then, run: `main.py` to generate the fixed raw rntry data
             2. Run: `python z_sensor_object_map_node.py --entry-dir logs/current_run_outputs/entry_dumps` 
                to generate the object_list.json and object_selection_results.json files in offline mode.
             3.  Run: `python object_candidate_selection.py --object-library-path logs/current_run_outputs/offline_outputs/object_list.json` 
                to generate the object_selection_results.json file.
             4. Run: `python visualize_offline_detections.py` to visualize the data with all Bboxes and selected ones highlighted in red.

Usage:
    python visualize_offline_detections.py \
        --dumps-dir logs/current_run_outputs/offline_outputs/image_data_dumps \
        --object-list logs/current_run_outputs/offline_outputs/object_list.json \
        --selection-results logs/current_run_outputs/offline_outputs/object_selection_results.json \
        --output-dir logs/current_run_outputs/offline_outputs/visualizations

Author: Robin Eshraghi
Date: 12/10/2025
"""
import pickle as pk
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, List

# Visualization constants (matching rex_play_vis.py style)
GREEN_COLOR = (0, 255, 0)  # BGR for green boxes
RED_COLOR = (0, 0, 255)  # BGR for red candidate boxes
BLUE_COLOR = (255, 0, 0)  # BGR for blue
ORANGE_COLOR = (0, 165, 255)  # BGR for orange
TEXT_COLOR = (255, 255, 255)  # White text
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LABEL = 0.4
FONT_SCALE_CAMERA = 0.6
THICKNESS_BOX = 2
THICKNESS_TEXT = 1
THICKNESS_CAMERA = 2

def load_object_mapping(object_list_path: Path) -> Dict:
    """Load object_list.json and create mapping from object_id to frame_id and bbox.
    
    Returns:
        Dict mapping object_id (can be string or int) to object info dict
    """
    with open(object_list_path, 'r') as f:
        data = json.load(f)
    
    mapping = {}
    for obj in data.get('objects', []):
        obj_id = obj.get('object_id')
        if obj_id is not None:
            mapping[obj_id] = {
                'frame_id': obj.get('frame_metadata', {}).get('frame_id'),
                'bbox_2d': obj.get('detection_metadata', {}).get('bbox_2d'),
                'label': obj.get('semantic_metadata', {}).get('label', ''),
                'yolo_score': obj.get('detection_metadata', {}).get('yolo_score', 0.0)
            }
    
    return mapping

def load_selected_objects(selection_results_path: Path) -> Tuple[Set, Dict]:
    """Load object_selection_results.json and extract selected object_ids with task info.
    
    Returns:
        Tuple of (selected_ids_set, object_id_to_task_info_dict)
        where task_info contains: task_id, rank, task_prompt, relevance_score
        Note: object_id can be string or int depending on the JSON format
    """
    with open(selection_results_path, 'r') as f:
        data = json.load(f)
    
    selected_ids = set()
    object_task_info = {}  # object_id -> {task_id, rank, task_prompt, relevance_score}
    
    for task in data.get('tasks', []):
        task_id = task.get('task_id', '')
        task_prompt = task.get('task_prompt', '')
        selected_objects = task.get('selected_objects', [])
        
        for rank, obj in enumerate(selected_objects, start=1):
            obj_id = obj.get('object_id')
            if obj_id is not None:
                # Keep object_id as-is (string or int) - don't convert
                selected_ids.add(obj_id)
                object_task_info[obj_id] = {
                    'task_id': task_id,
                    'rank': rank,
                    'task_prompt': task_prompt,
                    'relevance_score': obj.get('relevance_score', 0.0)
                }
    
    return selected_ids, object_task_info

def bbox_match(bbox1: list, bbox2: list, threshold: float = 10.0) -> bool:
    """Check if two bboxes match (within threshold pixels)."""
    if len(bbox1) != 4 or len(bbox2) != 4:
        return False
    # Check if all coordinates are within threshold, or use IoU-like matching
    # More lenient: check if center and size are similar
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate centers and sizes
    center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
    center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
    size1 = (x2_1 - x1_1, y2_1 - y1_1)
    size2 = (x2_2 - x1_2, y2_2 - y1_2)
    
    # Check if centers are close and sizes are similar
    center_diff = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
    size_diff = max(abs(size1[0] - size2[0]), abs(size1[1] - size2[1]))
    
    return center_diff < threshold and size_diff < threshold * 2

def find_matching_object_in_image_data(
    image_data: Dict, 
    target_frame_id: str, 
    target_bbox: list,
    target_label: str = None,
    target_camera: str = None,
    camera_name: str = None,
    is_rotated: bool = False
) -> Optional[str]:
    """Find the object_id in yolo_object_dict that matches the target bbox + label + camera.
    
    Matches by bbox + label + camera, NOT by frame_id/timestamp.
    This allows matching across different sessions.
    
    Args:
        image_data: Image data dictionary
        target_frame_id: Frame ID (kept for reference but not required for match)
        target_bbox: Bounding box coordinates to match
        target_label: Label to match (optional but recommended)
        target_camera: Camera name to match (optional but recommended)
        camera_name: Current image camera name (for validation)
        is_rotated: Whether the image is rotated (affects which bbox to check)
    """
    image_id = image_data.get('image_id', '')
    image_camera = image_data.get('rgb_name', '')
    
    # Match by camera name first (if provided)
    if target_camera and image_camera != target_camera:
        return None
    
    # Search through yolo_object_dict for matching bbox + label
    yolo_objects = image_data.get('yolo_object_dict', {})
    for obj_id, obj_data in yolo_objects.items():
        # Try both bbox coordinate systems for rotated cameras
        bboxes_to_check = []
        
        # For rotated cameras, check rotated_bbox_xyxy first
        if is_rotated:
            rotated_bbox = obj_data.get('rotated_bbox_xyxy')
            if rotated_bbox:
                bboxes_to_check.append(rotated_bbox)
        
        # Always check unrotated bbox as well
        unrotated_bbox = obj_data.get('bbox_xyxy')
        if unrotated_bbox:
            bboxes_to_check.append(unrotated_bbox)
        
        # Check if any bbox matches
        for bbox in bboxes_to_check:
            if bbox_match(bbox, target_bbox):
                # If label is provided, also check label match
                if target_label:
                    obj_label = obj_data.get('label', '').lower().strip()
                    if obj_label != target_label.lower().strip():
                        continue  # Bbox matches but label doesn't
                return obj_id
    
    return None

def draw_bboxes_on_image_opencv(
    image: np.ndarray,
    yolo_objects: Dict,
    selected_in_image: Dict,
    camera_name: str,
    is_rotated: bool
) -> np.ndarray:
    """Draw bounding boxes on image using OpenCV (matching rex_play_vis.py style).
    
    Args:
        image: RGB image array
        yolo_objects: Dictionary of detected objects
        selected_in_image: Dictionary of selected candidates {obj_id: task_info}
        camera_name: Name of the camera
        is_rotated: Whether image is rotated
    
    Returns:
        Annotated image (BGR format for OpenCV)
    """
    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        annotated = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    else:
        annotated = image.copy()
    
    h, w = annotated.shape[:2]
    
    # Draw camera name header (matching rex_play_vis.py style)
    (text_w, text_h), baseline = cv2.getTextSize(camera_name, FONT, FONT_SCALE_CAMERA, THICKNESS_CAMERA)
    cv2.rectangle(annotated, (5, 5), (text_w + 15, text_h + baseline + 15), (0, 0, 0), -1)
    cv2.putText(annotated, camera_name, (10, text_h + 10), FONT, FONT_SCALE_CAMERA, TEXT_COLOR, THICKNESS_CAMERA)
    
    # First pass: draw non-candidate boxes (green) - these go behind
    for obj_id, obj_data in yolo_objects.items():
        is_candidate = obj_id in selected_in_image
        if is_candidate:
            continue  # Skip candidates for now
        
        # Get appropriate bbox
        if is_rotated:
            bbox = obj_data.get('rotated_bbox_xyxy')
        else:
            bbox = obj_data.get('bbox_xyxy')
        
        if bbox is None:
            continue
        
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        
        if x1 >= x2 or y1 >= y2:
            continue
        
        label = obj_data.get('label', 'unknown')
        confidence = obj_data.get('confidence', 0.0)
        label_text = f"{label} {confidence:.2f}"
        
        # Draw green bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), GREEN_COLOR, THICKNESS_BOX)
        
        # Draw label with background (matching rex_play_vis.py style)
        (text_w_label, text_h_label), baseline_label = cv2.getTextSize(
            label_text, FONT, FONT_SCALE_LABEL, THICKNESS_TEXT
        )
        
        # Position label above box, or below if not enough space
        if y1 - text_h_label - baseline_label - 5 >= 0:
            label_y1 = y1 - text_h_label - baseline_label - 5
            label_y2 = y1
            text_y = y1 - baseline_label - 2
        else:
            label_y1 = y1
            label_y2 = y1 + text_h_label + baseline_label + 5
            text_y = y1 + text_h_label + 2
        
        label_x2 = min(x1 + text_w_label + 10, w - 1)
        text_bg_color = (GREEN_COLOR[0] // 2, GREEN_COLOR[1] // 2, GREEN_COLOR[2] // 2)
        
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), text_bg_color, -1)
        cv2.putText(annotated, label_text, (x1 + 3, text_y), FONT, FONT_SCALE_LABEL, TEXT_COLOR, THICKNESS_TEXT)
    
    # Second pass: draw candidate boxes (red) on top
    for obj_id, obj_data in yolo_objects.items():
        is_candidate = obj_id in selected_in_image
        if not is_candidate:
            continue  # Skip non-candidates
        
        # Get appropriate bbox
        if is_rotated:
            bbox = obj_data.get('rotated_bbox_xyxy')
        else:
            bbox = obj_data.get('bbox_xyxy')
        
        if bbox is None:
            continue
        
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        
        if x1 >= x2 or y1 >= y2:
            continue
        
        label = obj_data.get('label', 'unknown')
        confidence = obj_data.get('confidence', 0.0)
        
        # Get task info for this candidate
        task_info_obj = selected_in_image[obj_id]
        task_id = task_info_obj.get('task_id', '?')
        rank = task_info_obj.get('rank', 0)
        relevance_score = task_info_obj.get('relevance_score', 0.0)
        
        # Draw red bounding box with thicker line
        cv2.rectangle(annotated, (x1, y1), (x2, y2), RED_COLOR, THICKNESS_BOX + 1)
        
        # Draw filled semi-transparent rectangle for candidate highlight
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), RED_COLOR, -1)
        cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
        
        # Draw candidate label with background
        label_text = f"CANDIDATE Task {task_id} Rank #{rank}"
        label_text2 = f"{label} Conf:{confidence:.2f} Rel:{relevance_score:.3f}"
        
        # First line of label
        (text_w1, text_h1), baseline1 = cv2.getTextSize(
            label_text, FONT, FONT_SCALE_LABEL, THICKNESS_TEXT
        )
        # Second line of label
        (text_w2, text_h2), baseline2 = cv2.getTextSize(
            label_text2, FONT, FONT_SCALE_LABEL, THICKNESS_TEXT
        )
        
        text_w_label = max(text_w1, text_w2)
        text_h_label = text_h1 + text_h2 + baseline1 + baseline2 + 5
        baseline_label = baseline2  # Use baseline2 for positioning
        
        # Position label above box, or below if not enough space
        if y1 - text_h_label - baseline_label - 5 >= 0:
            label_y1 = y1 - text_h_label - baseline_label - 5
            label_y2 = y1
            text_y1 = y1 - text_h_label - baseline_label - 2
            text_y2 = y1 - baseline_label - 2
        else:
            label_y1 = y1
            label_y2 = y1 + text_h_label + baseline_label + 5
            text_y1 = y1 + text_h_label + baseline_label - text_h2 - baseline2 - 2
            text_y2 = y1 + text_h_label + baseline_label - 2
        
        label_x2 = min(x1 + text_w_label + 10, w - 1)
        text_bg_color = (RED_COLOR[0] // 2, RED_COLOR[1] // 2, RED_COLOR[2] // 2)
        
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), text_bg_color, -1)
        cv2.putText(annotated, label_text, (x1 + 3, text_y1), FONT, FONT_SCALE_LABEL, TEXT_COLOR, THICKNESS_TEXT)
        cv2.putText(annotated, label_text2, (x1 + 3, text_y2), FONT, FONT_SCALE_LABEL, TEXT_COLOR, THICKNESS_TEXT)
    
    return annotated

def create_frame_info_box(
    all_labels: Set[str],
    total_detections: int,
    total_candidates: int,
    candidates_info: List[Dict],
    width: int,
    height: int
) -> np.ndarray:
    """Create an info box summarizing frame statistics (matching rex_play_vis.py style).
    
    Args:
        all_labels: Set of all unique labels detected in the frame
        total_detections: Total number of bounding boxes across all cameras
        total_candidates: Total number of candidate objects
        candidates_info: List of candidate info dicts with keys: task_id, rank, label, relevance_score, camera_name
        width: Width of the info box
        height: Height of the info box
    
    Returns:
        Info box image (BGR format)
    """
    info_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_header = 0.8
    thickness_header = 2
    header_text = "Frame Info"
    (text_w, text_h), baseline = cv2.getTextSize(header_text, font, font_scale_header, thickness_header)
    
    header_height = text_h + baseline + 20
    cv2.rectangle(info_image, (0, 0), (width, header_height), (100, 100, 100), -1)
    cv2.putText(info_image, header_text, (10, text_h + 10), font, font_scale_header, TEXT_COLOR, thickness_header)
    
    y_offset = header_height + 30
    line_height = 30
    font_scale_text = 0.7
    thickness_text = 2
    
    # Total labels
    total_labels_text = f"Total Labels: {len(all_labels)}"
    cv2.putText(info_image, total_labels_text, (10, y_offset), font, font_scale_text, (0, 0, 0), thickness_text)
    y_offset += line_height
    
    # Total detections
    total_detections_text = f"Total Detections: {total_detections}"
    cv2.putText(info_image, total_detections_text, (10, y_offset), font, font_scale_text, (0, 0, 0), thickness_text)
    y_offset += line_height
    
    # Total candidates
    total_candidates_text = f"Total Candidates: {total_candidates}"
    cv2.putText(info_image, total_candidates_text, (10, y_offset), font, font_scale_text, (0, 0, 255), thickness_text)
    y_offset += line_height + 10
    
    # Candidate info section
    if candidates_info:
        cv2.putText(info_image, "Candidate Info:", (10, y_offset), font, font_scale_text, RED_COLOR, thickness_text)
        y_offset += line_height
        
        # Sort candidates by task_id and rank
        sorted_candidates = sorted(candidates_info, key=lambda x: (x.get('task_id', ''), x.get('rank', 0)))
        
        for cand in sorted_candidates:
            task_id = cand.get('task_id', '?')
            rank = cand.get('rank', 0)
            label = cand.get('label', 'unknown')
            relevance = cand.get('relevance_score', 0.0)
            camera = cand.get('camera_name', 'unknown')
            
            # Draw task, rank (colored), and label
            task_prefix = f"  Task {task_id} "
            rank_text = f"Rank #{rank}"
            separator_text = ": "
            
            # Determine rank color (entire "Rank #X" text)
            if rank == 1:
                rank_color = GREEN_COLOR
            elif rank == 2:
                rank_color = BLUE_COLOR
            elif rank == 3:
                rank_color = ORANGE_COLOR
            else:
                rank_color = (0, 0, 0)  # Black for other ranks
            
            # Calculate positions
            (task_w, _), _ = cv2.getTextSize(task_prefix, font, font_scale_text, thickness_text)
            (rank_w, _), _ = cv2.getTextSize(rank_text, font, font_scale_text, thickness_text)
            (sep_w, _), _ = cv2.getTextSize(separator_text, font, font_scale_text, thickness_text)
            
            # Draw each part
            x_pos = 10
            cv2.putText(info_image, task_prefix, (x_pos, y_offset), font, font_scale_text, (0, 0, 0), thickness_text)
            x_pos += task_w
            cv2.putText(info_image, rank_text, (x_pos, y_offset), font, font_scale_text, rank_color, thickness_text)
            x_pos += rank_w
            cv2.putText(info_image, separator_text, (x_pos, y_offset), font, font_scale_text, (0, 0, 0), thickness_text)
            x_pos += sep_w
            cv2.putText(info_image, label, (x_pos, y_offset), font, font_scale_text, RED_COLOR, thickness_text)
            y_offset += line_height
            
            cand_details = f"    Camera: {camera} | Rel: {relevance:.3f}"
            if len(cand_details) > 40:
                cand_details = cand_details[:37] + "..."
            cv2.putText(info_image, cand_details, (10, y_offset), font, font_scale_text * 0.9, (100, 100, 100), thickness_text - 1)
            y_offset += line_height
        
        y_offset += 5
    
    # Detected list
    if all_labels:
        cv2.putText(info_image, "Detected List:", (10, y_offset), font, font_scale_text, (0, 0, 0), thickness_text)
        y_offset += line_height
        
        detected_vocab = sorted(list(all_labels))
        max_chars_per_line = 35
        detected_str = ", ".join(detected_vocab)
        words = detected_str.split(", ")
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 2 <= max_chars_per_line:
                current_line += word + ", "
            else:
                if current_line:
                    cv2.putText(
                        info_image,
                        current_line.rstrip(", "),
                        (10, y_offset),
                        font,
                        font_scale_text,
                        (0, 0, 0),
                        thickness_text,
                    )
                    y_offset += line_height
                current_line = word + ", "
        
        if current_line:
            cv2.putText(
                info_image,
                current_line.rstrip(", "),
                (10, y_offset),
                font,
                font_scale_text,
                (0, 0, 0),
                thickness_text,
            )
    else:
        cv2.putText(
            info_image,
            "No detections",
            (10, y_offset),
            font,
            font_scale_text,
            (128, 128, 128),
            thickness_text,
        )
    
    return info_image

def visualize_image_with_bboxes(
    image_data_by_id_path: Path,
    object_mapping: Dict[int, Dict],
    selected_object_ids: Set[int],
    object_task_info: Dict[int, Dict],
    output_dir: Path = None,
    task_info: Dict = None
):
    """Load and visualize a single image_data_by_id pickle file with selected objects highlighted (OpenCV version)."""
    
    # Load the pickle file with error handling
    try:
        with open(image_data_by_id_path, 'rb') as f:
            image_data_by_id = pk.load(f)
    except EOFError as e:
        print(f"  ⚠ Warning: Pickle file appears corrupted or incomplete: {e}")
        print(f"    File size: {image_data_by_id_path.stat().st_size} bytes")
        return 0
    except Exception as e:
        print(f"  ⚠ Error loading pickle file: {e}")
        import traceback
        traceback.print_exc()
        return 0
    
    if not isinstance(image_data_by_id, dict):
        print(f"  ⚠ Warning: Expected dict, got {type(image_data_by_id)}")
        return 0
    
    if len(image_data_by_id) == 0:
        print(f"  ⚠ Warning: Empty image_data_by_id dictionary")
        return 0
    
    # Group images by timestamp (frame)
    frames_by_timestamp = {}
    for image_id, image_data in image_data_by_id.items():
        timestamp = image_data.get('timestamp', 'unknown')
        if timestamp not in frames_by_timestamp:
            frames_by_timestamp[timestamp] = []
        frames_by_timestamp[timestamp].append((image_id, image_data))
    
    # Track total candidates highlighted across all frames
    total_candidates_in_file = 0
    
    # Process each frame (group of images with same timestamp)
    for timestamp, frame_images in frames_by_timestamp.items():
        # Sort by camera name for consistent ordering
        frame_images.sort(key=lambda x: x[1].get('rgb_name', ''))
        num_cameras = len(frame_images)
        if num_cameras == 0:
            continue
        
        frame_total_candidates = 0
        frame_total_detections = 0
        frame_all_labels = set()
        frame_candidates_info = []  # List of candidate info dicts
        annotated_images = []
        
        # Process each camera image in the frame
        for image_id, image_data in frame_images:
            # Get the RGB image
            rotated = image_data.get('rotated_rgb_image')
            rgb_image = rotated if rotated is not None else image_data.get('rgb_image')
            if rgb_image is None:
                # Create placeholder for missing image
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"No RGB image", (10, 30), FONT, 0.5, (128, 128, 128), 1)
                cv2.putText(placeholder, str(image_id), (10, 60), FONT, 0.4, (128, 128, 128), 1)
                annotated_images.append(placeholder)
                continue
            
            camera_name = image_data.get('rgb_name', 'unknown')
            
            # Track which objects are selected and their task info
            selected_in_image = {}  # obj_id -> task_info
            
            # First pass: identify selected objects and get their task info
            candidates_checked = 0
            candidates_camera_match = 0
            candidates_bbox_match = 0
            
            # Iterate selected objects first
            for obj_id in selected_object_ids:
                if obj_id not in object_mapping:
                    continue
                
                obj_info = object_mapping[obj_id]
                candidates_checked += 1
                
                frame_id = obj_info.get('frame_id')
                bbox_2d = obj_info.get('bbox_2d')
                label = obj_info.get('label', '')
                
                # Extract camera from frame_id
                target_camera = None
                if frame_id and '/' in frame_id:
                    target_camera = frame_id.split('/')[-1]
                
                # Skip if camera doesn't match
                if target_camera and camera_name != target_camera:
                    continue
                
                candidates_camera_match += 1
                
                # Try to find matching object in current image
                if bbox_2d:
                    matching_obj_id = find_matching_object_in_image_data(
                        image_data, 
                        frame_id,
                        bbox_2d,
                        target_label=label,
                        target_camera=target_camera,
                        camera_name=camera_name,
                        is_rotated=(rotated is not None)
                    )
                    if matching_obj_id and obj_id in object_task_info:
                        candidates_bbox_match += 1
                        selected_in_image[matching_obj_id] = object_task_info[obj_id]
            
            # Debug logging
            if candidates_checked > 0:
                print(f"  [MATCH] Camera: {camera_name} | Checked {candidates_checked} | Camera match: {candidates_camera_match} | Bbox+Label match: {candidates_bbox_match}")
            
            # Draw bounding boxes using OpenCV
            yolo_objects = image_data.get('yolo_object_dict', {})
            annotated_image = draw_bboxes_on_image_opencv(
                rgb_image,
                yolo_objects,
                selected_in_image,
                camera_name,
                is_rotated=(rotated is not None)
            )
            
            annotated_images.append(annotated_image)
            
            # Collect statistics for info box
            num_selected = len(selected_in_image)
            frame_total_candidates += num_selected
            frame_total_detections += len(yolo_objects)
            for obj_data in yolo_objects.values():
                label = obj_data.get('label', '')
                if label:
                    frame_all_labels.add(label)
            
            # Collect candidate information
            for obj_id, task_info_obj in selected_in_image.items():
                obj_data = yolo_objects.get(obj_id, {})
                candidate_info = {
                    'task_id': task_info_obj.get('task_id', '?'),
                    'rank': task_info_obj.get('rank', 0),
                    'label': obj_data.get('label', 'unknown'),
                    'relevance_score': task_info_obj.get('relevance_score', 0.0),
                    'camera_name': camera_name
                }
                frame_candidates_info.append(candidate_info)
        
        # Create info box (6th box)
        if annotated_images:
            # Get dimensions from first camera image for info box
            first_img_height = annotated_images[0].shape[0]
            first_img_width = annotated_images[0].shape[1]
            
            info_box = create_frame_info_box(
                frame_all_labels,
                frame_total_detections,
                frame_total_candidates,
                frame_candidates_info,
                first_img_width,
                first_img_height
            )
            annotated_images.append(info_box)
        
        # Stitch all camera images together horizontally (no gaps)
        if annotated_images:
            # Ensure all images have the same height
            heights = [img.shape[0] for img in annotated_images]
            max_height = max(heights)
            
            # Resize images to same height if needed
            resized_images = []
            for img in annotated_images:
                if img.shape[0] != max_height:
                    scale = max_height / img.shape[0]
                    new_width = int(img.shape[1] * scale)
                    img = cv2.resize(img, (new_width, max_height))
                resized_images.append(img)
            
            # Stitch horizontally
            stitched_frame = np.hstack(resized_images)
            
            # Add frame info text at the top
            info_text = f"Frame: {timestamp} | {num_cameras} cameras | {frame_total_candidates} total candidates"
            (text_w, text_h), baseline = cv2.getTextSize(info_text, FONT, 0.7, 2)
            info_bar = np.ones((text_h + baseline + 20, stitched_frame.shape[1], 3), dtype=np.uint8) * 50
            cv2.putText(info_bar, info_text, (10, text_h + 10), FONT, 0.7, TEXT_COLOR, 2)
            stitched_frame = np.vstack([info_bar, stitched_frame])
            
            # Save or show
            if output_dir:
                safe_timestamp = str(timestamp).replace('/', '_').replace(':', '_')
                output_path = output_dir / f"vis_frame_{safe_timestamp}.png"
                cv2.imwrite(str(output_path), stitched_frame)
                print(f"Saved frame: {output_path} ({frame_total_candidates} candidate objects across {num_cameras} cameras)")
            else:
                cv2.imshow('Frame', stitched_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            total_candidates_in_file += frame_total_candidates
    
    return total_candidates_in_file

def visualize_all_dumps(
    dumps_dir: Path,
    object_list_path: Path,
    selection_results_path: Path,
    output_dir: Path = None
):
    """Visualize all pickle files with selected objects highlighted."""
    dumps_dir = Path(dumps_dir)
    if not dumps_dir.exists():
        print(f"Directory not found: {dumps_dir}")
        return
    
    # Load mappings
    print("Loading object mappings...")
    object_mapping = load_object_mapping(object_list_path)
    print(f"Loaded {len(object_mapping)} objects from object_list.json")
    
    selected_object_ids, object_task_info = load_selected_objects(selection_results_path)
    print(f"Found {len(selected_object_ids)} candidate objects across {len(set(info['task_id'] for info in object_task_info.values()))} tasks")
    
    # Debug: Show sample of selected object IDs and their frame_ids
    print(f"\n[DEBUG] Sample of {min(5, len(selected_object_ids))} selected object IDs:")
    for i, obj_id in enumerate(list(selected_object_ids)[:5]):
        if obj_id in object_mapping:
            frame_id = object_mapping[obj_id].get('frame_id', 'N/A')
            print(f"  {i+1}. object_id={obj_id} | frame_id={frame_id}")
        else:
            print(f"  {i+1}. object_id={obj_id} | NOT FOUND in object_mapping")
    
    # Load task info for display
    with open(selection_results_path, 'r') as f:
        selection_data = json.load(f)
    task_map = {obj.get('object_id'): task for task in selection_data.get('tasks', []) 
                for obj in task.get('selected_objects', [])}
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all pickle files
    pkl_files = sorted(dumps_dir.glob("image_data_*.pkl"))
    print(f"\nFound {len(pkl_files)} pickle files to visualize")
    
    successful = 0
    failed = 0
    total_candidates_highlighted = 0
    
    for pkl_file in pkl_files:
        print(f"\nProcessing: {pkl_file.name} ({pkl_file.stat().st_size / 1024 / 1024:.2f} MB)")
        try:
            # Get task info for objects in this file (if any)
            task_info = None  # Could be enhanced to show which task
            
            # visualize_image_with_bboxes processes multiple images per pickle file
            # Returns total number of candidate objects highlighted across all images in the file
            candidates_in_file = visualize_image_with_bboxes(
                pkl_file, 
                object_mapping, 
                selected_object_ids,
                object_task_info,
                output_dir,
                task_info
            )
            total_candidates_highlighted += candidates_in_file
            successful += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ Error processing {pkl_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"Summary: {successful} successful, {failed} failed out of {len(pkl_files)} files")
    print(f"Total candidate objects highlighted: {total_candidates_highlighted}")
    print(f"{'='*70}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize offline detection results with selected objects highlighted (OpenCV version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize and save to output directory
  python viz_scripts/visualize_offline_detections_v2.py \\
      --dumps-dir logs/current_run_outputs/offline_outputs/image_data_dumps \\
      --object-list logs/current_run_outputs/offline_outputs/object_list.json \\
      --selection-results logs/current_run_outputs/offline_outputs/object_selection_results.json \\
      --output-dir logs/current_run_outputs/offline_outputs/visualizations

  # Visualize interactively (shows images)
  python viz_scripts/visualize_offline_detections_v2.py \\
      --dumps-dir logs/current_run_outputs/offline_outputs/image_data_dumps \\
      --object-list logs/current_run_outputs/offline_outputs/object_list.json \\
      --selection-results logs/current_run_outputs/offline_outputs/object_selection_results.json
        """
    )
    parser.add_argument("--dumps-dir", type=str, 
                       default="logs/current_run_outputs/offline_outputs/image_data_dumps",
                       help="Directory containing image_data pickle files")
    parser.add_argument("--object-list", type=str,
                       default="logs/current_run_outputs/offline_outputs/object_list.json",
                       help="Path to object_list.json")
    parser.add_argument("--selection-results", type=str,
                       default="logs/current_run_outputs/offline_outputs/object_selection_results.json",
                       help="Path to object_selection_results.json")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save visualization images (if None, displays interactively)")
    
    args = parser.parse_args()
    
    visualize_all_dumps(
        Path(args.dumps_dir),
        Path(args.object_list),
        Path(args.selection_results),
        args.output_dir if args.output_dir else None
    )

