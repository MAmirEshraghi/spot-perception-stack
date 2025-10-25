#!/usr/bin/env python3

"""
Processes 2D bounding box detections and observation data to generate 3D
world coordinates for the center of each bounding box.

This script uses a configurable pixel grid (e.g., 3x3) method,
adapting the color-based segmentation logic from 'point_cloud.py'
to work with small point clusters.
"""

import json
import pickle
import sys
import numpy as np
import open3d as o3d
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

try:
    # Import required functions from helper scripts
    from src_perception.obs_data_buffer import ObsDataBuffer, ObsDataEntry, compose_transforms_optimized, depth_to_pointcloud
    from src_perception.components.point_cloud import generate_distinct_colors
except ImportError as e:
    print(f"Error: Could not import helper modules.")
    print(f"Make sure 'obs_data_buffer.py' and 'point_cloud.py' are in the same directory.")
    print(f"Details: {e}")
    sys.exit(1)

# --- Configuration Parameters  ---
# 1. Input/Output Paths
OBS_DATA_PATH = Path("data/obs_buffer.pkl")
DETECTION_DATA_PATH = Path("data/obs_buffer_detections.json")
DETECTION_DATA_PATH = Path("data/obs_buffer_detections_vlm.json")

OUTPUT_JSON_PATH = Path("data/bbox_3d_positions_v3.json")


# 2. Projection & Mask Parameters
# Radius for the pixel mask (e.g., 1 = 3x3 grid, 2 = 5x5 grid)
MASK_PIXEL_RADIUS = 1
# Conversion factor 
DEPTH_SCALE = 1000.0 #mm

# 3. 3D Point Cloud Filtering
# Max height (in meters) to keep points (filters ceilings)
FILTER_MAX_HEIGHT_METERS = 20.0
# Min height (in meters) to keep points (filters floors/self)
FILTER_MIN_HEIGHT_METERS = 0.1


# --- Robot-Specific Mappings ---
RGB_TO_DEPTH_MAP = {
    "head_rgb_left": "head_stereo_left_depth",
    "head_rgb_right": "head_stereo_right_depth",
    "left_rgb": "left_depth",
    "right_rgb": "right_depth",
    "rear_rgb": "rear_depth"
}

RGB_TO_LINK_MAP = {
    "head_rgb_left": "head_left_rgbd",
    "head_rgb_right": "head_right_rgbd",
    "left_rgb": "left_rgbd",
    "right_rgb": "right_rgbd",
    "rear_rgb": "rear_rgbd"
}

# --- Main Processing Function ---

def main():
    """
    Main function to load data, process detections, and save 3D positions.
    """
    
    # --- Statistics Initialization ---
    stats = {
        "total_detections_read": 0,
        "detections_processed": 0,
        "skipped_no_obs_entry": 0,
        "skipped_missing_sensor_data": 0,
        "skipped_missing_link_name": 0,
        "skipped_zero_depth_at_center": 0,
        "skipped_no_3d_points_generated": 0,
        "skipped_center_out_of_bounds": 0
    }
    
    all_3d_positions = []

    # --- 1. Load Inputs ---
    print(f"Loading observation buffer from {OBS_DATA_PATH}...")
    if not OBS_DATA_PATH.exists():
        print(f"Error: Input file not found: {OBS_DATA_PATH}")
        return
    with open(OBS_DATA_PATH, 'rb') as f:
        data_buffer: ObsDataBuffer = pickle.load(f)

    print(f"Loading detections from {DETECTION_DATA_PATH}...")
    if not DETECTION_DATA_PATH.exists():
        print(f"Error: Input file not found: {DETECTION_DATA_PATH}")
        return
    with open(DETECTION_DATA_PATH, 'r') as f:
        all_detections: dict = json.load(f)

    # Validate static transforms
    if not data_buffer.is_tf_static_ready():
        print("Error: Static transforms (TF) are not fully loaded in the observation buffer. Cannot proceed.")
        return
    
    static_transforms = data_buffer.static_transforms
    print("Data loaded successfully. Starting processing...")

    # --- 2. Iterate Timestamps ---
    for timestamp, camera_detections in all_detections.items():
        
        # Get the corresponding observation data entry
        entry: ObsDataEntry = data_buffer.get_entry_by_timestamp(timestamp)
        if not entry or not entry.is_frame_full():
            num_skipped = sum(len(dets) for dets in camera_detections.values())
            stats["skipped_no_obs_entry"] += num_skipped
            continue
            
        # Get odometry for this timestamp
        odom_to_base = {
            "position": entry.odometry["position"],
            "orientation": entry.odometry["orientation"]
        }

        # --- 3. Iterate Cameras ---
        for camera_name, detections in camera_detections.items():
            
            # Get RGB image
            rgb_image = entry.rgb_images.get(camera_name)
            
            # Get corresponding Depth image
            depth_name = RGB_TO_DEPTH_MAP.get(camera_name)
            depth_image = entry.depth_images.get(depth_name)


            if rgb_image is None or depth_image is None:
                stats["skipped_missing_sensor_data"] += len(detections)
                continue
                
            # Get camera pose
            link_name = RGB_TO_LINK_MAP.get(camera_name)
            if link_name is None:
                stats["skipped_missing_link_name"] += len(detections)
                continue
                
            try:
                w2c = compose_transforms_optimized(odom_to_base, link_name, static_transforms, use_optical=True)
            except ValueError as e:
                print(f"Warning: Could not get transform for {link_name} at {timestamp}. Skipping. Error: {e}")
                stats["skipped_missing_link_name"] += len(detections)
                continue

            camera_position = [w2c["position"]["x"], w2c["position"]["y"], w2c["position"]["z"]]
            camera_quaternion_xyzw = [w2c["orientation"]["x"], w2c["orientation"]["y"], w2c["orientation"]["z"], w2c["orientation"]["w"]]

            # --- 4. Prepare Masks for this Image ---
            masks_for_this_image = []
            for det_index, det in enumerate(detections):
                stats["total_detections_read"] += 1
                
                # A. Get 2D BBox and find center
                bbox = det['bbox']
                
                # A. Get 2D BBox and find center
                bbox = det['bbox']
                center_u = int((bbox[0] + bbox[2]) / 2)
                center_v = int((bbox[1] + bbox[3]) / 2)
                
                # Validate center coordinates
                if not (0 <= center_v < depth_image.shape[0] and 0 <= center_u < depth_image.shape[1]):
                    stats["skipped_center_out_of_bounds"] += 1
                    continue

                # B. Check depth at center pixel
                center_depth_value = float(depth_image[center_v, center_u])
                if center_depth_value == 0:
                    stats["skipped_zero_depth_at_center"] += 1
                    continue
                    
                # C. Create 3x3 mask (boolean array)
                segmentation_mask = np.zeros(depth_image.shape[:2], dtype=bool)
                v_start = max(0, center_v - MASK_PIXEL_RADIUS)
                v_end = min(depth_image.shape[0], center_v + MASK_PIXEL_RADIUS + 1) # Slice end is exclusive
                u_start = max(0, center_u - MASK_PIXEL_RADIUS)
                u_end = min(depth_image.shape[1], center_u + MASK_PIXEL_RADIUS + 1) # Slice end is exclusive
                segmentation_mask[v_start:v_end, u_start:u_end] = True
                
                # D. Store mask data
                mask_data = {
                    "segmentation": segmentation_mask,
                    "label": det['label'],
                    "score": det['score'],
                    "bbox_2d": det['bbox'],
                    "bbox_center_pixel": [center_u, center_v],
                    "depth_value": center_depth_value,
                    "id": f"{timestamp}_{camera_name}_{det_index}"
                }
                masks_for_this_image.append(mask_data)

            if not masks_for_this_image:
                continue

            # --- 5. Convert Masks to 3D Points (Adapted from point_cloud.py) ---
            
            H, W = depth_image.shape
            
            # A. Create a "Color Mask" image
            color_mask_image = np.zeros((H, W, 3), dtype=np.uint8)
            distinct_colors = generate_distinct_colors(len(masks_for_this_image))
            color_to_mask_map = {tuple(color): mask for color, mask in zip(distinct_colors, masks_for_this_image)}

            for i, mask in enumerate(masks_for_this_image):
                color_mask_image[mask['segmentation']] = distinct_colors[i]

            # B. Create a single point cloud using the "Color Mask"
            full_pcd, _ = depth_to_pointcloud(
                depth=depth_image,
                rgb=color_mask_image,
                position=camera_position,
                quaternion_xyzw=camera_quaternion_xyzw,
                depth_scale=DEPTH_SCALE 
                
            )
            if not full_pcd.has_points():
                stats["skipped_no_3d_points_generated"] += len(masks_for_this_image)
                continue
                
            # C. Filtering (using configured parameters)
            # # Note: depth_to_pointcloud also has a filter,
            # # I filter again below for explicit control.
            # points = np.asarray(full_pcd.points)
            # filtering = (points[:, 2] < FILTER_MAX_HEIGHT_METERS) & (points[:, 2] > FILTER_MIN_HEIGHT_METERS) # Filter ceiling and floor
            # indices2keep = np.where(filtering)[0]
            # full_pcd = full_pcd.select_by_index(indices2keep)
            
            # if not full_pcd.has_points():
            #     stats["skipped_no_3d_points_generated"] += len(masks_for_this_image)
            #     continue

            # D. Extract individual objects by color
            points = np.asarray(full_pcd.points)
            colors = (np.asarray(full_pcd.colors) * 255).astype(np.uint8)

            processed_ids_in_batch = set()
            for color in distinct_colors:
                target_color = np.array(color)
                indices = np.where(np.all(colors == target_color, axis=1))[0]
                
                source_mask = color_to_mask_map[tuple(color)]
                processed_ids_in_batch.add(source_mask['id'])

                # expect points (since depth was > 0), but check again
                if len(indices) > 0:
                    object_points = points[indices]
                    obj_pcd = o3d.geometry.PointCloud()
                    obj_pcd.points = o3d.utility.Vector3dVector(object_points)
                    
                    # --- 6. Get Center and Format Output ---
                    center_3d = obj_pcd.get_center()

                    output_data = {
                        "id": source_mask['id'],
                        "timestamp": timestamp,
                        "camera": camera_name,
                        "label": source_mask['label'],
                        "score": source_mask['score'],
                        "bbox_2d": source_mask['bbox_2d'],
                        "bbox_center_pixel": source_mask['bbox_center_pixel'],
                        "depth_value": source_mask['depth_value'],
                        "depth_meters": source_mask['depth_value'] / DEPTH_SCALE, # Use configured scale
                        "position_3d": list(center_3d),
                        "bbox_3d_corners": {}, # Empty as requested
                        "camera_position": list(camera_position)
                    }
                    all_3d_positions.append(output_data)
                    stats["detections_processed"] += 1
                
                else:
                    # This happens if the 3x3 grid points were filtered out
                    stats["skipped_no_3d_points_generated"] += 1
            
            # Account for any masks that weren't found in the color map (shouldn't happen)
            for mask in masks_for_this_image:
                if mask['id'] not in processed_ids_in_batch:
                     stats["skipped_no_3d_points_generated"] += 1

    # --- 7. Save Output ---
    print(f"\nProcessing complete. Saving {len(all_3d_positions)} 3D positions to {OUTPUT_JSON_PATH}...")
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(all_3d_positions, f, indent=2)

    # --- 8. Print Statistics ---
    print("\n--- Processing Statistics ---")
    print(f"Total 2D Detections Read:   {stats['total_detections_read']}")
    print(f"Successfully Processed:     {stats['detections_processed']}")
    
    total_skipped = stats['total_detections_read'] - stats['detections_processed']
    print(f"Total Detections Skipped:   {total_skipped}")
    
    print("\nReasons for Skipping:")
    print(f"  - No ObsDataEntry/Frame:    {stats['skipped_no_obs_entry']}")
    print(f"  - Missing Sensor (RGB/D):   {stats['skipped_missing_sensor_data']}")
    print(f"  - Missing Camera TF Link:   {stats['skipped_missing_link_name']}")
    print(f"  - BBox Center Out of Bounds:{stats['skipped_center_out_of_bounds']}")
    print(f"  - Zero Depth at Center:     {stats['skipped_zero_depth_at_center']}")
    print(f"  - Points Filtered (H/D):    {stats['skipped_no_3d_points_generated']}")
    
    if stats['total_detections_read'] > 0:
        success_rate = (stats['detections_processed'] / stats['total_detections_read']) * 100
        print(f"\nSuccess Rate: {success_rate:.2f}%")
    print("-----------------------------\n")

if __name__ == "__main__":
    main()