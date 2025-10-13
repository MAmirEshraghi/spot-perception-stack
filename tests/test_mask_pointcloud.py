import pickle
import open3d as o3d
import numpy as np
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from obs_data_buffer import ObsDataBuffer, depth_to_pointcloud, compose_transforms_optimized
from perception_pipeline.point_cloud import generate_distinct_colors, create_segmented_point_cloud


SAM_CHECKPOINT_PATH = "models/sam_vit_l_0b3195.pth"



def main():
    # load buffer from the pickle file
    with open("data/obs_buffer.pkl", "rb") as f:
        buffer = pickle.load(f)
    
    print(buffer.get_buffer_status())
    print("static transforms: ", buffer.entries.keys())

     # === Reconstruct point clouds from all entries ===
    merged_pcd = o3d.geometry.PointCloud() #canvas empty 3d point cloud

    # Data Load
    all_timesteps = list(buffer.entries.keys())


    # SAM
    sam = sam_model_registry["vit_l"](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to("cuda")
    mask_generator = SamAutomaticMaskGenerator(model=sam, min_mask_region_area=1000)


    merged_masks_pcd = o3d.geometry.PointCloud()  #creat empty as a Canvas

    for t_id in [2,33,]:
        target_stamp = all_timesteps[t_id]
        entry = buffer.entries[target_stamp]
        print(f"\n ... Processing single entry: {target_stamp}")


        all_mask_pcds = []
        odom_to_base = {"position": entry.odometry["position"], "orientation": entry.odometry["orientation"]}
        
        camera_mapping = {"head_rgb_left": "head_left_rgbd", "head_rgb_right": "head_right_rgbd", "left_rgb": "left_rgbd", "right_rgb": "right_rgbd", "rear_rgb": "rear_rgbd"}
                
        
        for rgb_name, rgb_image, depth_name, depth_image in entry.get_rgb_depth_pairs():
            total = 0
            if rgb_image.shape[:2] != depth_image.shape[:2]:
                    rgb_image = cv2.resize(rgb_image, (depth_image.shape[1], depth_image.shape[0]))
            
            masks = mask_generator.generate(rgb_image)
            camera_link_name = camera_mapping.get(rgb_name)
            w2c = compose_transforms_optimized(odom_to_base, camera_link_name, buffer.static_transforms, use_optical=True)
            pos = w2c["position"]
            orient = w2c["orientation"]
            
            position_xyz = np.array([pos['x'], pos['y'], pos['z']])
            quaternion_xyzw = np.array([orient['x'], orient['y'], orient['z'], orient['w']])

            # 1) call func to get all masks' point cloud
            extracted_objects  = create_segmented_point_cloud(
                masks,
                depth_image,
                rgb_image,
                position_xyz,
                quaternion_xyzw
            )

            print(f"Found and extracted {len(extracted_objects)} object point clouds.")

            # 2) Merge them into a single PointCloud object
            # Iterate through the list of dictionaries
            for obj_dict in extracted_objects:
                # Access the point cloud using the 'pcd' key
                merged_masks_pcd += obj_dict['pcd']


        
        #print("points length", total)
                
    o3d.visualization.draw_geometries(
        [merged_masks_pcd],
        window_name=f"Merged Masks for Entry: {target_stamp}"
    )

if __name__ == "__main__":
    main()






# def sample_color_from_gradient(t=None):
#     """
#     Generate a smooth RGB color (r, g, b) with values between 0 and 1.
#     If `t` is not provided, a random value will be used.
#     """
#     if t is None:
#         t = random.random()  # Random value between 0 and 1

#     # Smooth gradient using sine waves
#     r = 0.5 + 0.5 * math.sin(2 * math.pi * (t + 0.0))
#     g = 0.5 + 0.5 * math.sin(2 * math.pi * (t + 0.33))
#     b = 0.5 + 0.5 * math.sin(2 * math.pi * (t + 0.66))

#     return (r, g, b)
