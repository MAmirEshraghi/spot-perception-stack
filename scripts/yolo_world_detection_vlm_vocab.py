#!/usr/bin/env python3
"""
YOLO-World Object Detection with Dynamic VLM-Generated Vocabulary.

This script loads robot sensor data, and for each frame:
1. Uses a VLM (via vlm_interface) to identify objects in all camera views.
2. Creates a unified, dynamic vocabulary from the VLM's findings.
3. Sets this dynamic vocabulary in the YOLO-World model.
4. Runs YOLO-World detection on all images in the frame.
5. Saves the detections in the same format as the original script.
6. Saves annotated images with improved bounding box styling.
"""

import pickle
import json
import torch
import time
import cv2
import collections
import numpy as np
from pathlib import Path
from ultralytics import YOLOWorld
import supervision as sv  
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from multiprocessing import freeze_support  

# Import VLM interface
try:
    from src_perception.components.vlm_interface import create_vlm_detector, VLMInterface
except ImportError:
    print("ERROR: vlm_interface.py not found.")
    exit(1)


# --- RGB to Depth Mapping ---
# (Used for resizing images to match depth sensor resolution)
RGB_TO_DEPTH_MAP = {
    "head_rgb_left": "head_stereo_left_depth",
    "head_rgb_right": "head_stereo_right_depth",
    "left_rgb": "left_depth",
    "right_rgb": "right_depth",
    "rear_rgb": "rear_depth"
}

# Vocabulary will be generated dynamically by the VLM.


def load_robot_data(filepath: Path) -> Tuple[Any, List[Dict]]:
    """
    Loads the sensor data from the data.pkl file.
    """
    print(f"Loading data from {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            buffer = pickle.load(f)
            
            static_transforms = buffer.static_transforms
            sensor_data = []
            
            for header_stamp, entry in buffer.entries.items():
                frame_data = {
                    "header_stamp": header_stamp,
                    "rgb_images": entry.rgb_images,
                    "depth_images": entry.depth_images,
                    "odometry": entry.odometry
                }
                sensor_data.append(frame_data)
            
            print(f"  Loaded {len(sensor_data)} sensor data entries.")
            print(f"  Loaded static transforms: {len(static_transforms)}")
            return static_transforms, sensor_data
            
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {filepath}")
        exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load pickle file: {e}")
        exit(1)

def load_yolo_model(model_name: str) -> YOLOWorld:
    """
    Load YOLO-World model.
    Vocabulary will be set dynamically per frame.
    """
    print(f"\nLoading YOLO-World model: {model_name}")
    try:
        model = YOLOWorld(model_name)
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model '{model_name}'.")
        print("Make sure 'ultralytics' is installed and the model file is available.")
        print(f"Details: {e}")
        exit(1)
        
    # Ensure model is on CUDA if available
    if torch.cuda.is_available():
        model.model.to('cuda')
        print("   YOLO-World model loaded successfully (CUDA)")
    else:
        print("   YOLO-World model loaded successfully (CPU)")
        
    return model

def load_vlm_model(model_name: str) -> VLMInterface:
    """
    Load the VLM model using the interface.
    """
    print(f"\nLoading VLM model: {model_name}")
    try:
        vlm_detector = create_vlm_detector(
            model_type="internvlm",  # Use the 'internvlm' implementation
            model_name=model_name,
            verbose=True
        )
        print("   VLM model loaded successfully.")
        return vlm_detector
    except Exception as e:
        print(f"ERROR: Failed to load VLM detector: {e}")
        exit(1)


def get_batch_detections(model: YOLOWorld, images_batch: List[np.ndarray], 
                         conf_threshold: float, iou_threshold: float, 
                         device: str) -> Tuple[List[Any], float]:
    """
    Runs the YOLO model on a batch of images.
    
    Returns:
        A tuple of:
        - results (List[ultralytics.Results]): Raw results from model.predict()
        - duration (float): Time taken for this prediction in seconds.
    """
    start_time = time.perf_counter()
    results = model.predict(images_batch, 
                            conf=conf_threshold, 
                            iou=iou_threshold,
                            verbose=False,
                            device=device)
    duration = time.perf_counter() - start_time
    return results, duration

def draw_custom_bboxes(image: np.ndarray, detections: sv.Detections, vocabulary: List[str]) -> np.ndarray:
    """
    Draw bounding boxes and labels on an image using the improved style.
    
    Args:
        image: The image (numpy array) to draw on.
        detections: A supervision.Detections object.
        vocabulary: The list of class names used for this detection.
    
    Returns:
        The annotated image (numpy array).
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return image
    
    annotated = np.copy(image)
    
    if len(detections) == 0:
        return annotated
    
    h, w = annotated.shape[:2]
    
    for i in range(len(detections)):
        try:
            bbox = detections.xyxy[i]
            class_id = int(detections.class_id[i])
            confidence = float(detections.confidence[i])
            
            if class_id >= len(vocabulary):
                print(f"Warning: class_id {class_id} out of bounds for vocab size {len(vocabulary)}")
                continue
            
            class_name = vocabulary[class_id]
            
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            # Use style from yolo_plus_internvlm.py
            box_color = (0, 255, 0)  # Green
            text_bg_color = (0, 200, 0) # Darker Green
            text_color = (255, 255, 255) # White
            box_thickness = 2 # Thinner than 4, but visible
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, box_thickness)
            
            label = f"{class_name} {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Position label
            if y1 - text_h - baseline - 10 >= 0:
                # Above the box
                label_y1 = y1 - text_h - baseline - 10
                label_y2 = y1 - baseline + 5
                text_y = y1 - baseline - 5
            else:
                # Inside the box
                label_y1 = y1 + baseline - 5
                label_y2 = y1 + text_h + baseline + 10
                text_y = y1 + text_h + 5
            
            label_x2 = min(x1 + text_w + 10, w - 1)
            
            cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), text_bg_color, -1)
            cv2.putText(annotated, label, (x1 + 5, text_y), font, font_scale, text_color, font_thickness)
            
        except Exception as e:
            print(f"Warning: Failed to draw bbox {i}. Error: {e}")
            continue
    
    return annotated


def save_annotated_image(plotted_image: Any, output_dir: Path, 
                         header_stamp: str, image_name: str):
    """
    Saves the annotated image to the output directory. 
    """
    try:
        filename = f"{header_stamp}_{image_name}.jpg"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), plotted_image)
        
    except Exception as e:
        print(f"Warning: Failed to save image {filename}. Error: {e}")


def save_detections(detections_data: Dict, output_filepath: Path):
    """
    Saves the aggregated detection data to a file (JSON or PKL).
    """
    print(f"\nSaving detections to {output_filepath}...")
    try:
        if output_filepath.suffix == ".json":
            with open(output_filepath, 'w') as f:
                json.dump(detections_data, f, indent=4)
        elif output_filepath.suffix == ".pkl":
            with open(output_filepath, 'wb') as f:
                pickle.dump(detections_data, f)
        else:
            print(f"Warning: Unknown file extension {output_filepath.suffix}. Saving as JSON.")
            with open(output_filepath.with_suffix(".json"), 'w') as f:
                json.dump(detections_data, f, indent=4)
                
        print(f"Successfully saved {len(detections_data)} entries.")
        
    except Exception as e:
        print(f"ERROR: Failed to save output file: {e}")


def print_statistics(total_time: float, num_images: int, num_detections: int, 
                     num_frames: int, label_counts: Dict, vlm_time: float):
    """
    Calculates and prints the final performance statistics.
    """
    print("\n--- Detection Statistics ---")
    
    # --- Timing ---
    avg_yolo_time_per_image = total_time / num_images if num_images > 0 else 0
    avg_yolo_fps = 1.0 / avg_yolo_time_per_image if avg_yolo_time_per_image > 0 else 0
    avg_vlm_time_per_frame = vlm_time / num_frames if num_frames > 0 else 0
    
    print(f"Total Frames Processed:   {num_frames}")
    print(f"Total Images Processed: {num_images}")
    
    print("\n" + "-"*10)
    print(f"Total VLM Time (all frames): {vlm_time:.2f} seconds")
    print(f"Average VLM Time per Frame:  {avg_vlm_time_per_frame*1000:.2f} ms")
    
    print("\n" + "-"*10)
    print(f"Total YOLO Inference Time: {total_time:.2f} seconds")
    print(f"Average YOLO Time per Image: {avg_yolo_time_per_image*1000:.2f} ms")
    print(f"Average YOLO Inference FPS:  {avg_yolo_fps:.2f} FPS")
    
    # --- Detections ---
    avg_dets_per_image = num_detections / num_images if num_images > 0 else 0
    avg_dets_per_frame = num_detections / num_frames if num_frames > 0 else 0
    
    print("\n" + "-"*10)
    print(f"Total Detections Found: {num_detections}")
    print(f"Average Detections per Image: {avg_dets_per_image:.2f}")
    print(f"Average Detections per Frame (all 5 cameras): {avg_dets_per_frame:.2f}")

    # --- Label Counts ---
    print("\n" + "-"*10)
    print(f"Top 15 Most Common Detections:")
    if not label_counts:
        print("  None")
    else:
        sorted_labels = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
        for i, (label, count) in enumerate(sorted_labels[:15]):
            print(f"  {i+1}. {label}: {count} occurrences")
    
    print("----------------------------\n")


def main():
    """
    Main execution script to load data, run VLM, run dynamic detection, and save results.
    """
    
    # --- Configuration ---
    DATA_FILE = Path("data/obs_buffer.pkl")
    OUTPUT_FILE = Path("data/obs_buffer_detections_vlm.json") # New output file
    YOLO_MODEL_NAME = "yolov8x-worldv2.pt"
    VLM_MODEL_NAME = "OpenGVLab/InternVL3_5-1B" # VLM model to use
    CONF_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.75
    
    # ---  Qualitative/Stats Config ---
    QUALITATIVE_OUTPUT_DIR = Path("logs/yolo_world_vlm_images/")
    SAVE_IMAGES = True
    SAVE_ONLY_IF_DETECTIONS = True
    # --- End Configuration ---
    
    # 1. Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 2. Create output directory
    if SAVE_IMAGES:
        QUALITATIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Saving annotated images to: {QUALITATIVE_OUTPUT_DIR.resolve()}")
    
    # 3. Load the YOLO-World model (without vocab)
    yolo_model = load_yolo_model(YOLO_MODEL_NAME)
    
    # 4. Load the VLM model
    vlm_detector = load_vlm_model(VLM_MODEL_NAME)

    # 5. Load the robot data
    _, sensor_data = load_robot_data(DATA_FILE)
    
    if not sensor_data:
        print("No sensor data found in the file. Exiting.")
        return

    # 6. Process all frames
    all_detections = {}
    
    # --- Statistics Variables ---
    total_yolo_detection_time = 0.0
    total_vlm_time = 0.0
    total_images_processed = 0
    total_detections_found = 0
    label_counts = collections.defaultdict(int)
    # ---

    print(f"\nProcessing {len(sensor_data)} frames...")
    for frame_data in tqdm(sensor_data, desc="Detecting objects"):
        header_stamp = frame_data["header_stamp"]
        timestamp_detections = {}
        
        rgb_images = frame_data.get("rgb_images", {})
        depth_images = frame_data.get("depth_images", {})
        
        if not rgb_images:
            all_detections[header_stamp] = timestamp_detections
            continue

        # --- VLM Step 1: Prepare all images for this frame ---
        # We apply the same resizing logic from the original script
        images_to_process_map = {}
        camera_names_in_order = []
        
        for image_name, image_array in rgb_images.items():
            image_to_process = image_array
            
            depth_name = RGB_TO_DEPTH_MAP.get(image_name)
            if depth_name:
                depth_image = depth_images.get(depth_name)
                if depth_image is not None:
                    if image_array.shape[:2] != depth_image.shape[:2]:
                        depth_h, depth_w = depth_image.shape[:2]
                        image_to_process = cv2.resize(image_array, (depth_w, depth_h))
                else:
                    tqdm.write(f"\nWarning: No depth image '{depth_name}' found for {image_name}")
            else:
                 tqdm.write(f"\nWarning: No depth map defined for {image_name}")
            
            images_to_process_map[image_name] = image_to_process
            camera_names_in_order.append(image_name)

        # --- VLM Step 2: Get dynamic vocabulary for this frame ---
        vlm_start_time = time.perf_counter()
        # VLM gets all processed images at once
        vlm_objects_by_camera = vlm_detector.detect_objects(images_to_process_map)
        total_vlm_time += (time.perf_counter() - vlm_start_time)
        
        # Unify all found objects into a single vocabulary for YOLO
        frame_vocabulary_set = set()
        for camera_objects in vlm_objects_by_camera.values():
            frame_vocabulary_set.update(camera_objects)
        
        frame_vocabulary = sorted(list(frame_vocabulary_set))
        
        if not frame_vocabulary:
            tqdm.write(f"  Frame {header_stamp}: VLM found no objects. Skipping YOLO.")
            all_detections[header_stamp] = timestamp_detections # Save empty detections
            continue

        # --- YOLO Step 1: Set dynamic vocabulary ---
        yolo_model.set_classes(frame_vocabulary)
        
        # --- YOLO Step 2: Run batch detection ---
        images_batch = [images_to_process_map[name] for name in camera_names_in_order]
        
        yolo_results, duration = get_batch_detections(
            yolo_model, 
            images_batch,
            CONF_THRESHOLD, 
            IOU_THRESHOLD, 
            device
        )
        total_yolo_detection_time += duration

        # --- Post-processing Step: Format results and save images ---
        for i, (camera_name, yolo_res) in enumerate(zip(camera_names_in_order, yolo_results)):
            
            # Get the image that was actually processed
            processed_image = images_to_process_map[camera_name]
            
            # Parse results with Supervision
            detections_sv = sv.Detections.from_ultralytics(yolo_res)
            
            # Format detections into the *original* script's list format
            detections_list = []
            if len(detections_sv) > 0:
                for j in range(len(detections_sv)):
                    bbox = detections_sv.xyxy[j]
                    score = detections_sv.confidence[j]
                    cls_idx = detections_sv.class_id[j]
                    
                    if int(cls_idx) >= len(frame_vocabulary):
                        continue # Should not happen, but safeguard
                        
                    label = frame_vocabulary[int(cls_idx)]
                    
                    detections_list.append({
                        "bbox": [round(b, 2) for b in bbox],
                        "label": label,
                        "score": round(score, 4)
                    })

            # --- Store Detections for JSON/PKL (Original Structure) ---
            timestamp_detections[camera_name] = detections_list

            # --- Aggregate Statistics ---
            total_images_processed += 1
            total_detections_found += len(detections_list)
            for det in detections_list:
                label_counts[det['label']] += 1
            
            # --- Save Qualitative Image ---
            if SAVE_IMAGES:
                save_img = True
                if SAVE_ONLY_IF_DETECTIONS and len(detections_list) == 0:
                    save_img = False
                
                if save_img:
                    # Create the annotated image using the new drawing function
                    # We draw on the 'processed_image' which matches the bboxes
                    plotted_image = draw_custom_bboxes(
                        processed_image, 
                        detections_sv, 
                        frame_vocabulary
                    )
                    
                    save_annotated_image(
                        plotted_image,
                        QUALITATIVE_OUTPUT_DIR,
                        header_stamp,
                        image_name
                    )

        all_detections[header_stamp] = timestamp_detections

    # 7. Save the final detection data (JSON/PKL)
    save_detections(all_detections, OUTPUT_FILE)
    
    # 8. Print the final statistics report
    print_statistics(
        total_yolo_detection_time,
        total_images_processed,
        total_detections_found,
        len(sensor_data),
        label_counts,
        total_vlm_time
    )
    
    print("Detection process complete.")


if __name__ == "__main__":
    freeze_support()  # Recommended for multiprocessing libraries
    main()
