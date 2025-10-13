"""
Main entry point for the offline perception processing pipeline.

This script loads pre-recorded data, orchestrates the AI models for perception,
handles data logging, and generates visualizations and performance reports.
"""

# Standard library imports
import os
import pickle
import time
import argparse
import csv

# Third-party library imports
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

# Local application/library specific imports
from .log_manager import PerceptionLog
from .ai_models import get_vlm_description
from .point_cloud import get_organized_point_cloud, get_o3d_cam_intrinsic, calculate_point_cloud_coverage, calculate_point_cloud_coverage_kdtree
from .visualization import create_object_visualization, draw_detection_on_image, create_visualization
from .visualization import (save_merge_tracking_visualization, 
                            save_object_frame_visualization, 
                            save_full_frame_visualization, 
                            save_best_object_instance_visualization)

from .reporting import analyze_and_save_timing_log
from obs_data_buffer import ObsDataBuffer, compose_transforms_optimized

def get_vlm_description_for_bbox(full_image_np, bbox, model, processor):
    """
    Draws a bounding box on an image and gets a VLM description for the object inside.
    """
    image_with_bbox = full_image_np.copy()
    x, y, w, h = [int(c) for c in bbox]
    cv2.rectangle(image_with_bbox, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # This is the specific prompt needed for this task
    prompt_for_bbox = "Describe the object inside the red bounding box."
    
    description = get_vlm_description(image_with_bbox, model, processor, prompt=prompt_for_bbox)

    return description, image_with_bbox 

def create_object_visualization(cropped_image_np, label_text):
    padding, text_area_height, font_scale, font_thickness = 20, 50, 0.6, 2
    obj_h, obj_w, _ = cropped_image_np.shape
    (text_w, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    canvas_w = max(obj_w + (2 * padding), text_w + (2 * padding))
    canvas_h = obj_h + (2 * padding) + text_area_height
    canvas = np.full((canvas_h, canvas_w, 3), 240, dtype=np.uint8)
    paste_x = (canvas_w - obj_w) // 2
    paste_y = padding + text_area_height
    canvas[paste_y : paste_y + obj_h, paste_x : paste_x + obj_w] = cropped_image_np
    text_pos = (padding, padding + 20) 
    cv2.putText(img=canvas, text=label_text, org=text_pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 0), thickness=font_thickness)
    return canvas

def main(args):
    """
    Main function to load data, run the perception pipeline, and save results.
    Accepts a namespace object with configuration parameters.
    """
    # --- Performance Timing Setup ---
    timing_log = {
        "Total Time": 0.0,
        "1. Model Loading": 0.0,
        "2. Data Ingestion": 0.0,
        "3. Full AI Processing": 0.0,
        "3a. SAM per image": [],
        "3b. Dedup Centroid Check per mask": [],
        "3c. Dedup Coverage Check per mask": [],
        "3d. VLM Description per object": [],
        "3e. Masks per image": [],
        "4. Visualization": 0.0
    }
    
    overall_start_time = time.perf_counter()
    
    # --- 1. Load AI Models ---
    print("--- 1. Loading AI Models ---")
    model_load_start = time.perf_counter()
    #SAM
    sam = sam_model_registry["vit_l"](checkpoint=args.SAM_CHECKPOINT_PATH)
    sam.to(args.DEVICE)
    mask_generator = SamAutomaticMaskGenerator(model=sam, min_mask_region_area=args.min_mask_area)
    #VLM
    processor = AutoProcessor.from_pretrained(args.VLM_MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        args.VLM_MODEL_ID, torch_dtype=torch.bfloat16, _attn_implementation="eager"
    ).to(args.DEVICE)
    model.eval()
    #
    timing_log["1. Model Loading"] = time.perf_counter() - model_load_start
    print(f"Models loaded to {args.DEVICE} in {timing_log['1. Model Loading']:.2f} seconds.")

    # --- 2. Load and Ingest Data from .pkl File ---
    print(f"\n--- 2. Ingesting Data from {args.PKL_FILE_PATH} ---")
    ingestion_start = time.perf_counter()
    
    if not os.path.exists(args.PKL_FILE_PATH):
        print(f"[ERROR] PKL file not found at: {args.PKL_FILE_PATH}")
        return None
        
    with open(args.PKL_FILE_PATH, "rb") as f:
        buffer: ObsDataBuffer = pickle.load(f)

    perception_log = PerceptionLog(args)
    #counter for limit input:
    total_pairs = 0
    limit_reached = False 

    for entry in buffer.entries.values():
        if not entry.is_frame_full(): continue
        
        odom_to_base = {"position": entry.odometry["position"], "orientation": entry.odometry["orientation"]}
        
        for rgb_name, rgb_image, depth_name, depth_image in entry.get_rgb_depth_pairs():
            
            limit_is_active = isinstance(args.PROCESS_LIMIT, int) and args.PROCESS_LIMIT > 0
            if limit_is_active and total_pairs >= args.PROCESS_LIMIT:
                limit_reached = True
                break

            if rgb_image.shape[:2] != depth_image.shape[:2]:
                rgb_image = cv2.resize(rgb_image, (depth_image.shape[1], depth_image.shape[0]))

            camera_mapping = {"head_rgb_left": "head_left_rgbd", "head_rgb_right": "head_right_rgbd", "left_rgb": "left_rgbd", "right_rgb": "right_rgbd", "rear_rgb": "rear_rgbd"}
            camera_link_name = camera_mapping.get(rgb_name)
            if not camera_link_name: continue

            w2c = compose_transforms_optimized(odom_to_base, camera_link_name, buffer.static_transforms, use_optical=True)
            
            pos = w2c["position"]
            orient = w2c["orientation"]
            quat_ros_wxyz = np.array([orient['w'], orient['x'], orient['y'], orient['z']])
            pos_xyz = np.array([pos['x'], pos['y'], pos['z']])
            camera_pose_7d = np.concatenate([quat_ros_wxyz, pos_xyz])

            perception_log.add_image(rgb_image, depth_image, camera_pose_7d)
            total_pairs += 1
            
        if limit_reached:
            break

    timing_log["2. Data Ingestion"] = time.perf_counter() - ingestion_start
    print(f"Ingested {total_pairs} RGB-Depth pairs in {timing_log['2. Data Ingestion']:.2f} seconds.")

    # --- 3. Run AI Processing Pipeline ---
    print("\n--- 3. Starting AI Processing Pipeline ---")
    processing_start_time = time.perf_counter()
    
    try:
        log_filepath = os.path.join(perception_log.scan_dir, "deduplication_log.csv")

        with open(log_filepath, 'w', newline='') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow([
                "image_id", "mask_index", "new_mask_centroid", "compared_to_object_id", 
                "existing_object_centroid", "centroid_distance", "centroid_check_passed",
                "coverage_check_performed", "coverage_score_1", "coverage_score_2",
                "coverage_check_passed", "final_match_id"
            ])

            unique_object_point_clouds = {}
            per_frame_detections = {}
            best_instance_for_object = {}

            for image_id, image_data in perception_log.data["images"].items():
                print(f"\n--- Processing {image_id} ---")
                
                sam_start = time.perf_counter()
                rgb_np = np.array(Image.open(image_data["rgb_path"]))
                depth_np = np.load(image_data["depth_path"])
                camera_pose = np.array(image_data["camera_pose_world"])
                
                masks = mask_generator.generate(rgb_np)
                timing_log["3a. SAM per image"].append({"image_id": image_id, "time_ms": (time.perf_counter() - sam_start) * 1000})
                timing_log["3e. Masks per image"].append({"image_id": image_id, "mask_count": len(masks)})
                print(f"  > SAM found {len(masks)} masks.")

                h, w, _ = rgb_np.shape
                o3d_intrinsics = get_o3d_cam_intrinsic(h, w)
                organized_pcd = get_organized_point_cloud(depth_np, camera_pose, o3d_intrinsics)

                per_frame_detections[image_id] = []
                new_objects_in_this_frame = 0

                # initialize counters for image's report
                centroid_checks_this_image = 0
                coverage_checks_this_image = 0
                total_centroid_time_ms = 0.0
                total_coverage_time_ms = 0.0

                for mask_index, mask in enumerate(masks):
                    
                    # filter and Extract 3D Object Points
                    if mask.get('bbox') is None or len(mask['bbox']) != 4: continue 
                    object_points = organized_pcd[mask['segmentation']]
                    if object_points.shape[0] < 50: continue
                    # 3D centroid
                    new_centroid = np.mean(object_points, axis=0)
                    matched_id = None

                    # tracking logic: check current mask obj with all of uniques obj mask list:
                    for obj_id, data in unique_object_point_clouds.items():
                        
                        centroid_start = time.perf_counter()
                        # fast centroid distance check:
                        existing_centroid = data['centroid']
                        dist = np.linalg.norm(new_centroid - existing_centroid)
                        # logs:
                        centroid_duration_ms = (time.perf_counter() - centroid_start) * 1000
                        timing_log["3b. Dedup Centroid Check per mask"].append({
                            "image_id": image_id, "mask_index": mask_index,
                            "time_ms": round(centroid_duration_ms, 3)
                        })

                        # if centroid_passed:
                        centroid_passed = dist < args.centroid_threshold
                        coverage_performed, cov1, cov2, coverage_passed = False, 0.0, 0.0, False
                        if centroid_passed:
                            coverage_performed = True

                            coverage_start = time.perf_counter()
                            # accurate coverage checks:

                            # Voxel Hash:
                            # cov1 = calculate_point_cloud_coverage(data['pcd'], object_points, voxel_size=args.voxel_size)
                            # cov2 = calculate_point_cloud_coverage(object_points, data['pcd'], voxel_size=args.voxel_size)

                            #K-D Trees
                            cov1 = calculate_point_cloud_coverage_kdtree(data['pcd'], object_points, search_radius=args.kdtree_radius)
                            cov2 = calculate_point_cloud_coverage_kdtree(object_points, data['pcd'], search_radius=args.kdtree_radius)

                            # logs:
                            coverage_duration_ms = (time.perf_counter() - coverage_start) * 1000
                            timing_log["3c. Dedup Coverage Check per mask"].append({
                                "image_id": image_id, "mask_index": mask_index,
                                "time_ms": round(coverage_duration_ms, 3)
                            })
                            coverage_checks_this_image += 1
                            total_coverage_time_ms += coverage_duration_ms
                            # if coverage passed:
                            coverage_passed = cov1 > args.coverage_threshold or cov2 > args.coverage_threshold
                            if coverage_passed:
                                matched_id = obj_id # match found 
                        #logs:
                        log_writer.writerow([
                            image_id, mask_index, np.round(new_centroid, 3), obj_id, 
                            np.round(existing_centroid, 3), round(dist, 4), centroid_passed,
                            coverage_performed, round(cov1, 4), round(cov2, 4), coverage_passed, matched_id
                        ])

                        if matched_id: break #loop checking w all unique obj until match found

                    is_new_object = matched_id is None
                    if is_new_object:
                        # create a new unique obj w its details:
                        image_index = int(image_id.split('_')[-1])
                        new_objects_in_this_frame += 1
                        object_id = f"obj_{image_index}_{new_objects_in_this_frame}"
                        unique_object_point_clouds[object_id] = {'pcd': object_points, 'centroid': new_centroid}
                        perception_log.add_or_update_unique_object(object_id, "[AWAITING VLM]", new_centroid)
                    
                        save_merge_tracking_visualization(object_id, image_id, mask_index, rgb_np, mask['bbox'], perception_log.scan_dir)
                    else: #(if not new obj): merge w existing obj and update its info:
                        
                        object_id = matched_id
                        merged_pcd = np.vstack((unique_object_point_clouds[object_id]['pcd'], object_points))
                        new_avg_centroid = np.mean(merged_pcd, axis=0)
                        unique_object_point_clouds[object_id] = {'pcd': merged_pcd, 'centroid': new_avg_centroid}

                        save_merge_tracking_visualization(object_id, image_id, mask_index, rgb_np, mask['bbox'], perception_log.scan_dir)

                    #RECORDING detection for this frame (for visualization)
                    per_frame_detections[image_id].append({
                        "object_id": object_id, 
                        "is_new": is_new_object, 
                        "mask": mask['segmentation'], 
                        "bbox": mask['bbox']})
                    #2D mask w its processed unique ID:
                    current_instance_info = {"mask_area": mask['area'], "image_id": image_id, "bbox": mask['bbox']}
                    # update best_instance_for_object for VLM input if it's larger
                    if object_id not in best_instance_for_object or mask['area'] > best_instance_for_object[object_id]['mask_area']:
                        best_instance_for_object[object_id] = current_instance_info
                    # logs: add to permanent log
                    perception_log.add_object_instance(image_id, object_id, mask['bbox'], mask['area'], mask['segmentation'])
            
            print(f"  > Dedup Centroid: {centroid_checks_this_image} checks took {total_centroid_time_ms:.3f}ms")
            if coverage_checks_this_image > 0:
                print(f"  > Dedup Coverage: {coverage_checks_this_image} checks took {total_coverage_time_ms:.3f}ms")
                
            
            print("\n--- Generating VLM Descriptions for Unique Objects ---")
            for object_id, data in unique_object_point_clouds.items():
                if object_id not in best_instance_for_object: continue
                
                vlm_start = time.perf_counter()
                avg_position = data['centroid']
                best_instance_data = best_instance_for_object[object_id]
                # find path to best view image
                parent_image_path = perception_log.data["images"][best_instance_data["image_id"]]["rgb_path"]
                rgb_np = np.array(Image.open(parent_image_path))
                
                bbox = best_instance_data["bbox"]

                description, vlm_input_image = get_vlm_description_for_bbox(rgb_np, bbox, model, processor)

                # logs:
                vlm_duration_ms = (time.perf_counter() - vlm_start) * 1000
                timing_log["3d. VLM Description per object"].append({'time_ms': vlm_duration_ms})
                print(f"  > {object_id}: '{description}' (took {vlm_duration_ms:.3f}ms).")
                
                perception_log.add_or_update_unique_object(object_id, description, avg_position)
                # Use the new PerceptionLog method to save the image
                perception_log.save_vlm_input_image(object_id, vlm_input_image)

                save_object_frame_visualization(object_id, description, best_instance_data, perception_log)


    except Exception as e:
        print(f"\n--- ERROR DURING AI PROCESSING: {e} ---")
    finally:
        timing_log["3. Full AI Processing"] = time.perf_counter() - processing_start_time
        sam.to('cpu'); model.to('cpu'); torch.cuda.empty_cache()
        print("Moved models to CPU to free VRAM.")

    # --- 4. Generate Final Visualizations ---
    print("\n--- 4. Generating Final Visualizations ---")
    viz_start = time.perf_counter()
    
    # "frame visualization" folder
    for image_id, detections in per_frame_detections.items():
        save_full_frame_visualization(image_id, detections, perception_log)
    
    # object-specific visualizations
    for object_id, best_instance_data in best_instance_for_object.items():
        save_best_object_instance_visualization(object_id, best_instance_data, perception_log)


    timing_log["4. Visualization"] = time.perf_counter() - viz_start
    print(f"All visualizations saved in {timing_log['4. Visualization']:.2f} seconds.")

    perception_log.finalize_and_save(source_file=args.PKL_FILE_PATH)
    timing_log["Total Time"] = time.perf_counter() - overall_start_time
    perception_log.timing_log = timing_log
    return perception_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the offline perception pipeline on a pre-recorded obs_buffer.pkl file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Core Configuration Arguments ---
    parser.add_argument('-p', '--PKL_FILE_PATH', type=str, required=True, 
                        help='Required: Path to the input obs_buffer.pkl file.')
    parser.add_argument('-s', '--SAM_CHECKPOINT_PATH', type=str, required=True, 
                        help='Required: Path to the SAM checkpoint model file (.pth).')
    parser.add_argument('-v', '--VLM_MODEL_ID', type=str, default='HuggingFaceTB/SmolVLM-256M-Instruct',
                        help='Hugging Face model ID for the Vision-Language Model.')
    parser.add_argument('-d', '--DEVICE', type=str, default=None,
                        help='Device to use ("cuda" or "cpu"). If not set, it will auto-detect CUDA.')
    parser.add_argument('-l', '--PROCESS_LIMIT', type=int, default=150,
                        help='Maximum number of images to process. Set to 0 to process all images.')

    # --- Algorithm Tuning Arguments ---
    parser.add_argument('--centroid_threshold', type=int, default=200,
                        help='De-duplication: Maximum distance in millimeters between centroids for a potential match.')
    parser.add_argument('--coverage_threshold', type=float, default=0.70,
                        help='De-duplication: Minimum point cloud overlap percentage to confirm a match.')
    parser.add_argument('--voxel_size', type=float, default=0.30,
                        help='De-duplication: Voxel size in meters for the point cloud coverage check.')
    parser.add_argument('--min_mask_area', type=int, default=100,
                        help='SAM: Minimum area in pixels for a mask to be considered a valid object.')
    parser.add_argument('--vlm_padding', type=float, default=0.15,
                        help='VLM: Percentage of padding to add around an object before sending to the VLM.')
    parser.add_argument('--kdtree_radius', type=float, default=0.4,
                    help='De-duplication: Search radius in meters for the k-d tree coverage check.')
    
    args = parser.parse_args()

    # --- Post-parsing logic ---
    if args.DEVICE is None:
        args.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Execute the main pipeline ---
    perception_log_result = main(args)

    # --- Analyze results ---
    if perception_log_result:
        analyze_and_save_timing_log(perception_log_result, perception_log_result.timing_log, args.PROCESS_LIMIT) 
        print(f"\n✅ Processing complete. Results are in: {perception_log_result.scan_dir}")