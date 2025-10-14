import pickle
import open3d as o3d
import numpy as np
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from obs_data_buffer import ObsDataBuffer, depth_to_pointcloud, compose_transforms_optimized
import torch
import math

SAM_CHECKPOINT_PATH = "models/sam_vit_l_0b3195.pth"

# --- HELPER AND DEDUPLICATION FUNCTIONS (No changes here) ---

def generate_distinct_colors(n):
    """Generates n visually distinct colors."""
    # ... (code is unchanged)
    colors = []
    for i in range(n):
        hue = i * 0.61803398875
        hue %= 1
        if hue < 1/6.: r, g, b = 1, hue*6, 0
        elif hue < 2/6.: r, g, b = 1-(hue-1/6.)*6, 1, 0
        elif hue < 3/6.: r, g, b = 0, 1, (hue-2/6.)*6
        elif hue < 4/6.: r, g, b = 0, 1-(hue-3/6.)*6, 1
        elif hue < 5/6.: r, g, b = (hue-4/6.)*6, 0, 1
        else: r, g, b = 1, 0, 1-(hue-5/6.)*6
        colors.append((int(r*255), int(g*255), int(b*255)))
    return colors

def create_segmented_point_cloud(masks, depth_image, rgb_image, position, quaternion_xyzw):
    """Creates a 3D point cloud where points are pre-colored by their mask ID."""
    # ... (code is unchanged)
    H, W = depth_image.shape
    color_mask_image = np.zeros((H, W, 3), dtype=np.uint8)
    distinct_colors = generate_distinct_colors(len(masks))
    for i, mask in enumerate(masks):
        color_mask_image[mask['segmentation']] = distinct_colors[i]
    full_pcd, _ = depth_to_pointcloud(
        depth=depth_image, rgb=color_mask_image, position=position, quaternion_xyzw=quaternion_xyzw
    )
    if not full_pcd.has_points(): return []
    points = np.asarray(full_pcd.points)
    filtering = (points[:,2] < 20) & (points[:,2] > 0.1)
    indices2keep = np.where(filtering)[0]
    full_pcd = full_pcd.select_by_index(indices2keep)
    if not full_pcd.has_points(): return []
    points = np.asarray(full_pcd.points)
    colors = (np.asarray(full_pcd.colors) * 255).astype(np.uint8)
    object_pcds = []
    for color in distinct_colors:
        target_color = np.array(color)
        indices = np.where(np.all(colors == target_color, axis=1))[0]
        if len(indices) > 50: # Filter small objects
            object_points = points[indices]
            obj_pcd = o3d.geometry.PointCloud()
            obj_pcd.points = o3d.utility.Vector3dVector(object_points)
            obj_pcd.paint_uniform_color([c/255.0 for c in color])
            object_pcds.append(obj_pcd)
    return object_pcds

def mega_optimized_batch_coverage(query_pcd, pcd_list, voxel_size=0.05):
    """Calculates coverage of one query_pcd against a list of pcds."""
    # ... (code from your provided script)
    device = query_pcd.device
    if len(query_pcd) == 0: return torch.zeros(len(pcd_list), device=device)
    P = 73856093
    def _compute_hashes(points, vs, unique=False):
        voxel_points = torch.floor(points / vs).long()
        unique_voxels = torch.unique(voxel_points, dim=0) if unique else voxel_points
        if len(unique_voxels) == 0: return None, 0
        hashes = (unique_voxels[:, 0] + unique_voxels[:, 1] * P + unique_voxels[:, 2] * (P * P))
        return hashes, len(unique_voxels) if unique else 0
    query_hash, query_size = _compute_hashes(query_pcd, voxel_size, unique=True)
    if query_hash is None: return torch.zeros(len(pcd_list), device=device)
    all_pcds, cloud_ids = [], []
    for i, pcd in enumerate(pcd_list):
        if len(pcd) > 0:
            all_pcds.append(pcd)
            cloud_ids.append(torch.full((len(pcd),), i, device=device, dtype=torch.long))
    if not all_pcds: return torch.zeros(len(pcd_list), device=device)
    all_points = torch.cat(all_pcds, dim=0)
    all_cloud_ids = torch.cat(cloud_ids, dim=0)
    all_hashes, _ = _compute_hashes(all_points, voxel_size)
    if all_hashes is None: return torch.zeros(len(pcd_list), device=device)
    matches = torch.isin(all_hashes, query_hash)
    matched_hashes = all_hashes[matches]
    matched_cloud_ids = all_cloud_ids[matches]
    combined = torch.stack([matched_cloud_ids, matched_hashes], dim=1)
    unique_pairs = torch.unique(combined, dim=0)
    counts = torch.bincount(unique_pairs[:, 0], minlength=len(pcd_list))
    return counts.float() / query_size


def main():
    # 1. SETUP AND INITIALIZATION ---
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Deduplication parameters
    VOXEL_SIZE = 0.05
    COVERAGE_THRESHOLD = 0.5

    # Load data and SAM model
    with open("data/obs_buffer.pkl", "rb") as f:
        buffer = pickle.load(f)
    sam = sam_model_registry["vit_l"](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(model=sam, min_mask_region_area=1000)

    # 2. MAIN DATA STRUCTURE FOR TRACKING UNIQUE OBJECTS ---
    # This list will store the point cloud for each unique object we find.
    unique_object_pcds = []
    
    all_timesteps = list(buffer.entries.keys())
    
    # Process multiple timesteps to find objects across time and space
    for t_id in [2,24,]:
        target_stamp = all_timesteps[t_id]
        entry = buffer.entries[target_stamp]
        print(f"\n--- Processing Entry: {target_stamp} ---")
        
        odom_to_base = {"position": entry.odometry["position"], "orientation": entry.odometry["orientation"]}
        camera_mapping = {"head_rgb_left": "head_left_rgbd", "head_rgb_right": "head_right_rgbd", "left_rgb": "left_rgbd", "right_rgb": "right_rgbd", "rear_rgb": "rear_rgbd"}
        
        for rgb_name, rgb_image, depth_name, depth_image in entry.get_rgb_depth_pairs():
            if rgb_image.shape[:2] != depth_image.shape[:2]:
                rgb_image = cv2.resize(rgb_image, (depth_image.shape[1], depth_image.shape[0]))
            
            masks = mask_generator.generate(rgb_image)
            
            w2c = compose_transforms_optimized(odom_to_base, camera_mapping.get(rgb_name), buffer.static_transforms, use_optical=True)
            pos = w2c["position"]; orient = w2c["orientation"]
            position_xyz = np.array([pos['x'], pos['y'], pos['z']])
            quaternion_xyzw = np.array([orient['x'], orient['y'], orient['z'], orient['w']])

            # Get a list of newly detected objects from the current camera view
            candidate_pcds = create_segmented_point_cloud(
                masks, depth_image, rgb_image, position_xyz, quaternion_xyzw
            )
            print(f"  > Found {len(candidate_pcds)} potential objects in view '{rgb_name}'.")

            # 3. DEDUPLICATION LOGIC 
            for candidate in candidate_pcds:
                # If this is the first object ever found, it's automatically unique
                if not unique_object_pcds:
                    unique_object_pcds.append(candidate)
                    print("    -> New unique object 0 (first ever).")
                    continue

                # Convert candidate and unique objects to PyTorch tensors for the GPU
                candidate_tensor = torch.from_numpy(np.asarray(candidate.points)).to(device).float()
                unique_tensors = [torch.from_numpy(np.asarray(pcd.points)).to(device).float() for pcd in unique_object_pcds]

                # Calculate coverage against all existing unique objects at once
                coverages = mega_optimized_batch_coverage(candidate_tensor, unique_tensors, VOXEL_SIZE)
                
                # Check for a match
                if len(coverages) > 0 and torch.max(coverages) > COVERAGE_THRESHOLD:
                    # Match found! Merge the candidate into the best-matching unique object.
                    best_match_index = torch.argmax(coverages).item()
                    unique_object_pcds[best_match_index] += candidate
                    print(f"    -> Matched with unique object {best_match_index} (score: {torch.max(coverages):.2f}). Merging.")
                else:
                    # No match found. This is a new unique object.
                    unique_object_pcds.append(candidate)
                    print(f"    -> New unique object {len(unique_object_pcds)-1} found.")

    # 4. FINAL VISUALIZATION AFTER DEDUPLICATION ---
    print(f"\n--- Total Unique Objects Found: {len(unique_object_pcds)} ---")

    # Create a list of PCDs to visualize, coloring each unique object differently
    final_pcds_to_visualize = []
    final_colors = generate_distinct_colors(len(unique_object_pcds))

    for i, pcd in enumerate(unique_object_pcds):
        # Downsample for cleaner visualization
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        
        # Assign the final unique color
        color_rgb = [c/255.0 for c in final_colors[i]]
        downsampled_pcd.paint_uniform_color(color_rgb)
        final_pcds_to_visualize.append(downsampled_pcd)

    print("Visualizing the final deduplicated scene...")
    o3d.visualization.draw_geometries(
        final_pcds_to_visualize,
        window_name="Final Deduplicated 3D Scene"
    )

if __name__ == "__main__":
    main()