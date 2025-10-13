import os
import json
import numpy as np
import datetime
from PIL import Image

class PerceptionLog:
    """Manages the creation, organization, and saving of perception data."""
    def __init__(self, args, base_dir="logs/perception_logs_offline"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        vlm_name_abbr = args.VLM_MODEL_ID.split('/')[-1]
        limit_str = f"lim{args.PROCESS_LIMIT}" if args.PROCESS_LIMIT > 0 else "limAll"
        
        param_str = (
            f"ct{args.centroid_threshold}"
            f"_covt{str(args.coverage_threshold).replace('.', 'p')}"
            f"_vs{str(args.voxel_size).replace('.', 'p')}"
            f"_vlmp{str(args.vlm_padding).replace('.', 'p')}"
            f"_mma{args.min_mask_area}"
            f"_{limit_str}"
            f"_vlm{vlm_name_abbr}"
        )

        self.scan_id = f"{timestamp}_{param_str}"
        self.scan_dir = os.path.join(base_dir, self.scan_id)

        self.image_counter = 0
        self.instance_counter = 0

        # create all necessary directories
        os.makedirs(self.scan_dir, exist_ok=True)
        for sub_dir in ["rgb", "depth", "masks", "masks_png", "obj_frame_visualizations", 
                        "frame_visualizations", "object_visualization", "merge_tracking", 
                        "vlm_input_images"]:
            os.makedirs(os.path.join(self.scan_dir, sub_dir), exist_ok=True)

        print(f"[PerceptionLog] Initialized new scan log at: {self.scan_dir}")

        self.data = {
            "scan_metadata": {"scan_id": self.scan_id, "start_time": datetime.datetime.now().isoformat(), "source_file": None},
            "images": {},
            "unique_objects": {},
            "object_instances": {}
        }

    def add_image(self, rgb_np_0_255, depth_np, camera_pose_7d):
        image_id = f"img_{self.image_counter:03d}"
        rgb_path = os.path.join(self.scan_dir, "rgb", f"{image_id}.png")
        Image.fromarray(rgb_np_0_255).save(rgb_path)

        depth_path = os.path.join(self.scan_dir, "depth", f"{image_id}.npy")
        np.save(depth_path, depth_np)
        
        self.data["images"][image_id] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "camera_pose_world": camera_pose_7d.tolist(),
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "detected_object_instances": []
        }
        self.image_counter += 1
        return image_id

    def add_object_instance(self, image_id, object_id, bbox, mask_area, mask_np):
        instance_id = f"inst_{self.instance_counter:04d}"
        mask_path = os.path.join(self.scan_dir, "masks", f"{instance_id}.npy")
        np.save(mask_path, mask_np)
        
        mask_image_np = (mask_np.astype(np.uint8) * 255)
        mask_png_path = os.path.join(self.scan_dir, "masks_png", f"{instance_id}.png")
        Image.fromarray(mask_image_np).save(mask_png_path)
        
        self.data["object_instances"][instance_id] = {
            "parent_image_id": image_id,
            "parent_object_id": object_id,
            "bounding_box": [int(c) for c in bbox],
            "mask_area": int(mask_area),
            "mask_path": mask_path
        }
        if object_id in self.data["unique_objects"]:
            self.data["unique_objects"][object_id]["instances"].append(instance_id)
            if image_id not in self.data["unique_objects"][object_id]["seen_in_images"]:
                self.data["unique_objects"][object_id]["seen_in_images"].append(image_id)
                   
        self.instance_counter += 1
        return instance_id

    def add_or_update_unique_object(self, object_id, description, world_position):
        if object_id not in self.data["unique_objects"]:
            self.data["unique_objects"][object_id] = {
                "vlm_description": description,
                "avg_world_position": world_position.tolist(),
                "instances": [],
                "seen_in_images": [] 
            }
        else:
            self.data["unique_objects"][object_id]["vlm_description"] = description
            self.data["unique_objects"][object_id]["avg_world_position"] = world_position.tolist()

    def save_vlm_input_image(self, object_id, image_with_bbox_np):
        """Saves the image with a bbox and logs its path."""
        if object_id in self.data["unique_objects"]:
            # define the path for the new image
            file_name = f"{object_id}.png"
            image_path = os.path.join(self.scan_dir, "vlm_input_images", file_name)

            # save the image
            Image.fromarray(image_with_bbox_np).save(image_path)

            # log the path in the unique_object's data
            self.data["unique_objects"][object_id]["vlm_input_image_path"] = image_path
            return image_path
        return None

    def add_visualization_path_to_object(self, object_id, viz_path):
        if object_id in self.data["unique_objects"]:
            self.data["unique_objects"][object_id]["best_instance_image_path"] = viz_path

    def finalize_and_save(self, source_file=None):
        self.data["scan_metadata"]["source_file"] = source_file
        log_path = os.path.join(self.scan_dir, "log.json")
        with open(log_path, "w") as f:
            json.dump(self.data, f, indent=4)
        print(f"[PerceptionLog] Successfully saved log to: {log_path}")