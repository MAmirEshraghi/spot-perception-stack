#!/usr/bin/env python3

import pickle
import json
import torch
import time
import cv2  # Added for saving images
import collections # Added for statistics
from pathlib import Path
from ultralytics import YOLOWorld
from tqdm import tqdm # For a nice progress bar
from typing import List, Dict, Any, Tuple

# --- RGB to Depth Mapping ---
# (Used for resizing images to match depth sensor resolution)
RGB_TO_DEPTH_MAP = {
    "head_rgb_left": "head_stereo_left_depth",
    "head_rgb_right": "head_stereo_right_depth",
    "left_rgb": "left_depth",
    "right_rgb": "right_depth",
    "rear_rgb": "rear_depth"
}
# --- Object Vocabulary ---
VOCABULARY = [
    # People
    "person", "child", "baby", "pet", "cat", "dog",
    # Furniture
    "chair", "table", "desk", "sofa", "couch", "armchair", "stool", "bench",
    "bed", "mattress", "pillow", "blanket", "nightstand", "dresser",
    "wardrobe", "closet", "bookshelf", "shelf", "cabinet", "drawer",
    "coffee table", "dining table", "tv stand", "lamp", "mirror", "carpet", "rug",
    # Doors & Windows
    "door", "window", "curtain", "blind", "doorknob",
    # Electronics
    "tv", "monitor", "computer", "laptop", "keyboard", "mouse", "printer",
    "speaker", "remote", "headphones", "phone", "tablet", "camera", "clock",
    "fan", "heater", "air conditioner", "router", "modem", "light bulb",
    # Kitchen Items
    "fridge", "microwave", "oven", "stove", "toaster", "kettle", "coffee maker",
    "blender", "dishwasher", "sink", "faucet", "pot", "pan", "plate", "bowl",
    "cup", "glass", "mug", "spoon", "fork", "knife", "chopsticks",
    "cutting board", "bottle", "jar", "can", "trash bin", "napkin", "towel",
    # Cleaning Items
    "broom", "mop", "vacuum", "bucket", "soap", "detergent", "sponge",
    "tissue box", "trash bag",
    # Bathroom Items
    "toothbrush", "toothpaste", "towel", "mirror", "shampoo", "soap", "toilet",
    "toilet paper", "shower", "bathtub", "sink",
    # Bedroom Items
    "alarm clock", "lamp", "blanket", "pillow", "sheet", "closet", "hanger",
    # Office & Stationery
    "pen", "pencil", "notebook", "paper", "book", "envelope", "stapler",
    "scissors", "tape", "ruler", "calendar",
    # Personal Items
    "wallet", "bag", "backpack", "handbag", "watch", "glasses", "keys",
    "shoes", "hat", "jacket", "coat", "umbrella",
    # Food & Grocery
    "apple", "banana", "bread", "egg", "milk", "cheese", "bottle of water",
    "cereal box", "orange", "tomato", "carrot", "rice", "pasta",
    # Decor & Misc
    "painting", "photo frame", "vase", "plant", "flower", "candle", "clock",
    "basket", "box", "toy", "ball", "guitar", "remote control", "keychain",
    # Laundry & Storage
    "washing machine", "dryer", "laundry basket", "iron", "ironing board",
    "hanger", "closet", "drawer", "storage box",
    # Outdoor
    "bicycle", "car", "shoe rack", "garden hose", "watering can",
    "lawn mower", "grill", "trash can", "mailbox", "toolbox", "hammer", "screwdriver"
]


def load_robot_data(filepath: Path) -> Tuple[Any, List[Dict]]:
    """
    Loads the sensor data from the data.pkl file.
    """
    print(f"Loading data from {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            buffer = pickle.load(f)
            
            # ObsDataBuffer has entries attribute containing the sensor data
            static_transforms = buffer.static_transforms
            sensor_data = []
            
            # Convert entries to the expected format
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

def load_yolo_model(model_name: str, vocabulary: List[str]) -> YOLOWorld:
    """
    Load YOLO-World model and set vocabulary.
    """
    print(f"\nLoading YOLO-World model: {model_name}")
    try:
        model = YOLOWorld(model_name)
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model '{model_name}'.")
        print("Make sure 'ultralytics' is installed and the model file is available.")
        print(f"Details: {e}")
        exit(1)
        
    print(f"Setting vocabulary with {len(vocabulary)} classes")
    if len(vocabulary) > 10:
        print(f"  e.g., {', '.join(vocabulary[:10])}... (+{len(vocabulary)-10} more)")
    else:
        print(f"  e.g., {', '.join(vocabulary)}")
        
    model.set_classes(vocabulary)
    return model


def get_image_detections(model: YOLOWorld, image: Any, 
                         conf_threshold: float, iou_threshold: float, 
                         device: str) -> Tuple[List[Dict], float, Any]:
    """
    Runs the YOLO model on a single image and formats the results.
    
    Returns:
        A tuple of:
        - detections_list (List[Dict]): Formatted detection data.
        - duration (float): Time taken for this prediction in seconds.
        - plotted_image (np.ndarray): The image with boxes/labels drawn.
    """
    # 1. Time the prediction
    start_time = time.perf_counter()
    results = model.predict(image, 
                            conf=conf_threshold, 
                            iou=iou_threshold,
                            verbose=False,
                            device=device)
    duration = time.perf_counter() - start_time

    detections_list = []
    
    # 2. Get the image with boxes drawn on it
    # .plot() returns a BGR numpy array with annotations
    plotted_image = results[0].plot() 
    
    # 3. Format the detection data
    if results and len(results) > 0:
        res = results[0]
        boxes_xyxy = res.boxes.xyxy.tolist()
        conf_scores = res.boxes.conf.tolist()
        class_indices = res.boxes.cls.tolist()
        class_names_map = res.names
        
        for bbox, score, cls_idx in zip(boxes_xyxy, conf_scores, class_indices):
            detections_list.append({
                "bbox": [round(b, 2) for b in bbox], # [xmin, ymin, xmax, ymax]
                "label": class_names_map[int(cls_idx)],
                "score": round(score, 4)
            })
            
    return detections_list, duration, plotted_image


def save_annotated_image(plotted_image: Any, output_dir: Path, 
                         header_stamp: str, image_name: str):
    """
    Saves the annotated image to the output directory.
    
    Args:
        plotted_image: The numpy array (BGR) from results[0].plot()
        output_dir: The directory to save to.
        header_stamp: The timestamp (for the ID).
        image_name: The camera name.
    """
    try:
        # Create a clean filename: {timestamp}_{camera_name}.jpg
        filename = f"{header_stamp}_{image_name}.jpg"
        filepath = output_dir / filename
        
        # Save the image using cv2.imwrite
        # We use str(filepath) because cv2 doesn't always like Path objects
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
                     num_frames: int, label_counts: Dict):
    """
    Calculates and prints the final performance statistics.
    """
    print("\n--- Detection Statistics ---")
    
    # --- Timing ---
    avg_time_per_image = total_time / num_images if num_images > 0 else 0
    avg_fps = 1.0 / avg_time_per_image if avg_time_per_image > 0 else 0
    
    print(f"Total Images Processed: {num_images}")
    print(f"Total Model Inference Time: {total_time:.2f} seconds")
    print(f"Average Time per Image: {avg_time_per_image*1000:.2f} ms")
    print(f"Average Inference FPS: {avg_fps:.2f} FPS")
    
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
        # Sort the label_counts dictionary by value (count) in descending order
        sorted_labels = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
        for i, (label, count) in enumerate(sorted_labels[:15]):
            print(f"  {i+1}. {label}: {count} occurrences")
    
    print("----------------------------\n")


def main():
    """
    Main execution script to load data, run detection, and save results.
    """
    
    # --- Configuration ---
    DATA_FILE = Path("data/obs_buffer.pkl")
    OUTPUT_FILE = Path("data/obs_buffer_detections.json")
    MODEL_NAME = "yolov8x-worldv2.pt"
    CONF_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.75
    
    # ---  Qualitative/Stats Config ---
    QUALITATIVE_OUTPUT_DIR = Path("logs/yolo_world_json_images/") # Directory to save images
    SAVE_IMAGES = True                 # Set to True to save annotated images
    SAVE_ONLY_IF_DETECTIONS = True # Set to True to only save images *with* detections
    # --- End Configuration ---
    
    # 1. Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 2. Create output directory if it doesn't exist
    if SAVE_IMAGES:
        QUALITATIVE_OUTPUT_DIR.mkdir(exist_ok=True)
        print(f"Saving annotated images to: {QUALITATIVE_OUTPUT_DIR.resolve()}")
    
    # 3. Load the YOLO-World model
    model = load_yolo_model(MODEL_NAME, VOCABULARY)

    # 4. Load the robot data
    _, sensor_data = load_robot_data(DATA_FILE)
    
    if not sensor_data:
        print("No sensor data found in the file. Exiting.")
        return

    # 5. Process all frames
    all_detections = {}
    
    # --- Statistics Variables ---
    total_detection_time = 0.0
    total_images_processed = 0
    total_detections_found = 0
    label_counts = collections.defaultdict(int) # Automatically handles new keys
    # ---

    print(f"\nProcessing {len(sensor_data)} frames...")
    for frame_data in tqdm(sensor_data, desc="Detecting objects"):
        header_stamp = frame_data["header_stamp"]
        timestamp_detections = {}
        rgb_images = frame_data.get("rgb_images", {})
        
        # --- Get the depth images for this frame ---
        depth_images = frame_data.get("depth_images", {})
        
        for image_name, image_array in rgb_images.items():
            
            # --- Find corresponding depth image to match resolution ---
            image_to_process = image_array  # Default to original image
            
            depth_name = RGB_TO_DEPTH_MAP.get(image_name)
            if depth_name:
                depth_image = depth_images.get(depth_name)
                
                if depth_image is not None:
                    # This is your requested logic!
                    if image_array.shape[:2] != depth_image.shape[:2]:
                        depth_h, depth_w = depth_image.shape[:2]
                        # Resize RGB to match depth (w, h)
                        image_to_process = cv2.resize(image_array, (depth_w, depth_h))
                else:
                    print(f"\nWarning: No depth image '{depth_name}' found for {image_name}")
            else:
                 print(f"\nWarning: No depth map defined in RGB_TO_DEPTH_MAP for {image_name}")

            
            # --- Run Detection ---
            # Pass the (potentially resized) image_to_process to the model
            detections_list, duration, plotted_image = get_image_detections(
                model, 
                image_to_process, # Use the correctly-sized image
                CONF_THRESHOLD, 
                IOU_THRESHOLD, 
                device
            )
            
            # --- Store Detections for JSON/PKL ---
            timestamp_detections[image_name] = detections_list

            # --- Aggregate Statistics ---
            total_detection_time += duration
            total_images_processed += 1
            total_detections_found += len(detections_list)
            for det in detections_list:
                label_counts[det['label']] += 1
            
            # --- Save Qualitative Image ---
            if SAVE_IMAGES:
                save_img = True
                # If this flag is True, only save if we found something
                if SAVE_ONLY_IF_DETECTIONS and len(detections_list) == 0:
                    save_img = False
                
                if save_img:
                    save_annotated_image(
                        plotted_image,
                        QUALITATIVE_OUTPUT_DIR,
                        header_stamp,
                        image_name
                    )

        all_detections[header_stamp] = timestamp_detections

    # 6. Save the final detection data (JSON/PKL)
    save_detections(all_detections, OUTPUT_FILE)
    
    # 7. Print the final statistics report
    print_statistics(
        total_detection_time,
        total_images_processed,
        total_detections_found,
        len(sensor_data),
        label_counts
    )
    
    print("Detection process complete.")


if __name__ == "__main__":
    main()