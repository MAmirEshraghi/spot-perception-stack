#!/usr/bin/env python3
"""
Minimal and clean observation data buffer system for real-time processing.
"""
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import open3d as o3d
#!/usr/bin/env python3
from calendar import c
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict, OrderedDict
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import os, shutil


def filter_ceiling_points(pcd, robot_position, clip_height=2, min_distance=0.45, max_distance=None):
    """
    Filter out ceiling points and points too close/far from the robot from a point cloud.

    Args:
        pcd: Open3D point cloud
        robot_position: [x, y, z] robot position in world coordinates
        clip_height: height above robot position to filter (meters)
        min_distance: minimum distance from robot to keep points (meters)
        max_distance: maximum distance from robot to keep points (meters). 
                     If None, no maximum distance filtering is applied (horizon clipping disabled).

    Returns:
        Open3D point cloud with ceiling and near/far points removed
    """
    if len(pcd.points) == 0:
        return pcd

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Filter out ceiling points
    height_mask = points[:, 2] < clip_height

    # Filter out points too close to robot position (efficient vectorized distance)
    # Assume both robot_position and world points are in z-up frame (no axis flip)
    # this assumption is broken, the position is in 
    robot_pos = np.asarray(robot_position).reshape(1, 3)
    dists = np.linalg.norm(points - robot_pos, axis=1)
    distance_mask = dists > min_distance
    
    # Apply horizon clipping: filter out points beyond max_distance
    if max_distance is not None:
        distance_mask = distance_mask & (dists < max_distance)

    # Combine both masks
    valid_mask = np.logical_and(height_mask, distance_mask)

    if np.any(valid_mask):
        filtered_points = points[valid_mask]
        filtered_colors = colors[valid_mask]

        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        return filtered_pcd
    else:
        return o3d.geometry.PointCloud()

def get_camera_intrinsics(depth_image):
    """Get camera intrinsics for Open3D"""
    height, width = depth_image.shape
    
    # Camera intrinsics based on original 720x720 resolution with 90° FOV
    # Original HeadRGBRightSensorConfig: fx = fy = cx = cy = 720 / 2.0 = 360.0
    # When resized to 256x256: fx = fy = cx = cy = 256 / 2.0 = 128.0
    # Scale intrinsics proportionally to image size
    fx = fy = width / 2.0  # Simple pinhole model scaled to current image size
    cx = width / 2.0
    cy = height / 2.0
    
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(width, height, fx, fy, cx, cy)
    return intrinsics

def depth_to_pointcloud(depth, rgb, position, quaternion_xyzw, clip_height=2.5, depth_scale=1000.0, depth_trunc=25.0):
    """
    Convert depth image to colored point cloud using camera pose.   
    
    Args:
        depth: numpy array depth image
        rgb: numpy array RGB image
        position: [x, y, z] camera position in world coordinates
        quaternion_xyzw: [qx, qy, qz, qw] camera orientation quaternion
        clip_height: height above robot position to filter ceiling points (meters)
        depth_scale: depth conversion factor (default: 1000.0 for mm to meters)
        depth_trunc: maximum depth to consider from camera (meters)
    
    Returns:
        Open3D point cloud, position tuple
    """
    
    intrinsics = get_camera_intrinsics(depth)
    
    # The position and quaternion represent the camera's pose in world coordinates
    # This means we have a camera-to-world (c2w) transform
    rotation_matrix = R.from_quat(quaternion_xyzw).as_matrix()
    
    c2w = np.eye(4)
    c2w[:3, :3] = rotation_matrix
    c2w[:3, 3] = position

    # Create depth and RGB images for Open3D
    # Convert numpy arrays to Open3D Image objects
    depth_img = o3d.geometry.Image(depth.astype(np.float32))
    rgb_img = o3d.geometry.Image(rgb.astype(np.uint8))
    
    # Convert depth to point cloud
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_img, depth_img,
        depth_scale=depth_scale,  # Convert mm to meters
        depth_trunc=depth_trunc,    # 15 meters max
        convert_rgb_to_intensity=False
    )
    # Create point cloud in camera coordinates
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

    # Transform to world coordinates
    if len(pcd.points) > 0:
        # Transform to habitat world coordinates
        pcd.transform(c2w)
        
        # Filter out ceiling points only (no horizon clipping at this level)
        pcd = filter_ceiling_points(pcd, position, clip_height, max_distance=None)

        
    return pcd, position
    # return remove_floaters_o3d_builtin(pcd), position





def compose_transforms(parent_to_child, child_to_grandchild):
    """Compose two transforms: parent -> child -> grandchild = parent -> grandchild"""
    # Convert to rotation matrices
    R1 = R.from_quat([parent_to_child["orientation"]["x"], parent_to_child["orientation"]["y"], 
                    parent_to_child["orientation"]["z"], parent_to_child["orientation"]["w"]]).as_matrix()
    R2 = R.from_quat([child_to_grandchild["orientation"]["x"], child_to_grandchild["orientation"]["y"], 
                    child_to_grandchild["orientation"]["z"], child_to_grandchild["orientation"]["w"]]).as_matrix()
    
    # Compose rotations: R_composed = R1 @ R2 (correct order for transform composition)
    R_composed = R1 @ R2
    
    # Compose translations: t_composed = R1 @ t2 + t1 (correct transform composition)
    t1 = np.array([parent_to_child["position"]["x"], parent_to_child["position"]["y"], parent_to_child["position"]["z"]])
    t2 = np.array([child_to_grandchild["position"]["x"], child_to_grandchild["position"]["y"], child_to_grandchild["position"]["z"]])
    t_composed = R1 @ t2 + t1
    
    # Convert back to quaternion
    quat_composed = R.from_matrix(R_composed).as_quat()
    
    return {
        "position": {"x": t_composed[0], "y": t_composed[1], "z": t_composed[2]},
        "orientation": {"x": quat_composed[0], "y": quat_composed[1], "z": quat_composed[2], "w": quat_composed[3]}
    }
    
    # Process odometry first (should be only one per timestamp)
    
def compose_transforms_optimized(odom_to_base, camera_name, static_transforms, use_optical=True):
    """
    Optimized transform composition using loaded static transforms.
    
    Args:
        odom_to_base: measured odom -> base_link transform from odometry (dict)
        camera_name: camera link name (e.g., "head_left_rgbd")
        static_transforms: loaded static transforms dict
        use_optical: if True, compose to optical frame; if False, compose to camera link frame
    
    Returns:
        dict: Composed transform (world -> camera_optical or world -> camera_link)
    """
    # Get body -> camera_optical transform from loaded data
    if camera_name not in static_transforms.get("body_to_opticals", {}):
        raise ValueError(f"Unknown camera: {camera_name}. Available cameras: {list(static_transforms.get('body_to_opticals', {}).keys())}")

    # Compose: world->map->odom
    world_to_odom = compose_transforms(static_transforms["world_to_map"], static_transforms["map_to_odom"])

    # Compose: world->map->odom->base_link 
    # -- BUT static_transforms["odom_to_body"] should NOT be used if odom_to_base is live
    world_to_base = compose_transforms(world_to_odom, odom_to_base)

    # Compose: world->map->odom->base_link->camera (optical or link)
    world_to_camera = compose_transforms(world_to_base, static_transforms["body_to_opticals"][camera_name])

    return world_to_camera

class ObsDataEntry:
    """
    Holds all sensor data for a single timestamp.
    Tracks completeness and processing status.
    """
    
    def __init__(self, header_stamp: str):
        self.header_stamp = header_stamp
        self.processed = False
        
        # Data containers - store actual data, not file paths
        self.rgb_images: Dict[str, np.ndarray] = {}  # name -> rgb_array
        self.depth_images: Dict[str, np.ndarray] = {}  # name -> depth_array
        self.odometry: Optional[Dict] = None
        
        # Expected sensors
        self.expected_rgb = {"head_rgb_left", "head_rgb_right", "left_rgb", "right_rgb", "rear_rgb"}
        self.expected_depth = {"head_stereo_left_depth", "head_stereo_right_depth", "left_depth", "right_depth", "rear_depth"}
        
    @staticmethod
    def get_camera_mapping(rgb_name: str) -> str:
        return {
            "head_rgb_left": "head_left_rgbd",
            "head_rgb_right": "head_right_rgbd",
            "left_rgb": "left_rgbd",
            "right_rgb": "right_rgbd",
            "rear_rgb": "rear_rgbd"
        }[rgb_name]
        
    
    def add_rgb(self, name: str, image: np.ndarray):
        """Add RGB image data"""
        self.rgb_images[name] = image
    
    def add_depth(self, name: str, image: np.ndarray):
        """Add depth image data"""
        self.depth_images[name] = image
    
    def add_odometry(self, odom_data: Dict):
        """Add odometry data"""
        self.odometry = odom_data
    
    def is_frame_full(self, return_info: bool = False):
        """Check if all expected sensor data is present
        
        Args:
            return_info: If True, return tuple of (is_full, (rgb_complete, depth_complete, odom_complete))
                        If False, return only is_full boolean
        
        Returns:
            bool or tuple: Frame completeness status
        """
        rgb_complete = self.expected_rgb.issubset(set(self.rgb_images.keys()))
        depth_complete = self.expected_depth.issubset(set(self.depth_images.keys()))
        has_odometry = self.odometry is not None
        
        is_full = rgb_complete and depth_complete and has_odometry
        
        if return_info:
            return is_full, (rgb_complete, depth_complete, has_odometry)
        else:
            return is_full

    def set_processed(self):
        """Set entry as processed"""
        self.processed = True
    
    def get_rgb_depth_pairs(self) -> List[tuple]:
        """Get matching RGB-depth pairs"""
        pairs = []
        
        # Define matching logic
        matches = {
            "head_rgb_left": "head_stereo_left_depth",
            "head_rgb_right": "head_stereo_right_depth", 
            "left_rgb": "left_depth",
            "right_rgb": "right_depth",
            "rear_rgb": "rear_depth"
        }
        
        for rgb_name, depth_name in matches.items():
            if rgb_name in self.rgb_images and depth_name in self.depth_images:
                pairs.append((rgb_name, self.rgb_images[rgb_name], depth_name, self.depth_images[depth_name]))
        
        return pairs

    def get_pointcloud(self, static_transforms: Dict) -> Tuple[List[o3d.geometry.PointCloud], List[List[float]]]:
        """
        Generate point cloud for a complete entry using static transforms
        
        Args:
            static_transforms: Dictionary of static transforms
        
        Returns:
            Tuple of (point clouds, camera positions):
            - List of Open3D point clouds (one per camera)
            - List of camera positions [[x,y,z], ...] (one per camera)
        """
        import time
        start_time = time.time()
        
        if not self.is_frame_full():
            raise ValueError(f"Entry {self.header_stamp} is not complete")
        
        if static_transforms is None:
            raise ValueError("Static transforms not ready")
        

        # print(f"[PROFILE] Getting pointcloud for entry {self.header_stamp}")
        setup_time = time.time()
        
        # Get odometry transform
        odom_to_base = {"position": self.odometry["position"],"orientation": self.odometry["orientation"]}
        
        combined_pcd = o3d.geometry.PointCloud()
        pcds = []
        camera_positions = []  # Store camera positions for each point cloud
        pairs_time = time.time()
        pairs = self.get_rgb_depth_pairs()
        # print(f"[PROFILE]   Setup: {setup_time - start_time:.3f}s, Get pairs: {pairs_time - setup_time:.3f}s, Found {len(pairs)} pairs")
        
        # Process each RGB-depth pair
        for i, (rgb_name, rgb_image, depth_name, depth_image) in enumerate(pairs):
            pair_start = time.time()
            try:
                # print(f"[PROFILE]   Processing pair {i+1}/{len(pairs)}: {rgb_name}+{depth_name}")
                
                # Resize RGB to match depth if needed
                resize_start = time.time()
                if rgb_image.shape[:2] != depth_image.shape[:2]:
                    import cv2
                    rgb_image = cv2.resize(rgb_image, (depth_image.shape[1], depth_image.shape[0]))
                resize_time = time.time() - resize_start
                
                # Get camera link name from rgb name mapping
                transform_start = time.time()
                # Compose world-to-camera transform
                # World to camera | w2c
                w2c = compose_transforms_optimized(odom_to_base, 
                                                    self.get_camera_mapping(rgb_name), 
                                                    static_transforms, 
                                                    use_optical = True)
                
                # Extract position and orientation
                position = [w2c["position"]["x"],w2c["position"]["y"],w2c["position"]["z"]]
                orientation = [w2c["orientation"]["x"],w2c["orientation"]["y"],w2c["orientation"]["z"],w2c["orientation"]["w"]]
                transform_time = time.time() - transform_start
                
                # Create point cloud (basic ceiling filtering only, no horizon clipping)
                pcd_start = time.time()
                pcd, _ = depth_to_pointcloud(
                    depth_image, rgb_image, position, orientation,
                    clip_height=2.5, depth_scale=1000.0, depth_trunc=25.0
                )
                pcd_time = time.time() - pcd_start

                if len(pcd.points) > 0:
                    pcds.append(pcd)
                    camera_positions.append(position)  # Save camera position for this point cloud
                else:
                    pcds.append(pcd)  # Still append empty point cloud to maintain index alignment
                    camera_positions.append(position)
                
                # Combine point clouds
                # combine_start = time.time()
                # combined_pcd = combined_pcd + pcd if len(pcd.points) > 0 else combined_pcd
                # combine_time = time.time() - combine_start
                
                # pair_total = time.time() - pair_start
                # print(f"[PROFILE]     Resize: {resize_time:.3f}s, Transform: {transform_time:.3f}s, PCD: {pcd_time:.3f}s, Combine: {combine_time:.3f}s, Total: {pair_total:.3f}s, Points: {len(pcd.points)}")
                    
            except Exception as e:
                print(f"Error processing {rgb_name}+{depth_name}: {e}")
                continue
        total_time = time.time() - start_time
        # print(f"[PROFILE] TOTAL get_pointcloud time: {total_time:.3f}s, Final points: {len(combined_pcd.points)}")
        
        # returns 5 point clouds per entry and their corresponding camera positions
        return pcds, camera_positions
    
class ObsDataBuffer:
    """
    Manages observation data entries by timestamp.ObsDataB
    
    Expected tf_static transforms (must be provided before processing):
    - world -> map
    - base_link -> head_left_rgbd, head_right_rgbd, left_rgbd, right_rgbd, rear_rgbd
    - *_rgbd -> *_rgbd_optical (for all cameras)
    """
    
    def __init__(self, session_logger, max_size: int = 10):

        self.logger = session_logger

        self.max_size = max_size
        self.entries: OrderedDict[str, ObsDataEntry] = OrderedDict()  # header_stamp -> ObsDataEntry (FIFO queue)
        
        # 2025-11-14 23:43:40,023 - 2025-11-14_23-43-28_sensor_mapper - INFO - Adding tf_static: world, map
        # 2025-11-14 23:43:40,023 - 2025-11-14_23-43-28_sensor_mapper - INFO - Adding tf_static: map, spot/odom
        # 2025-11-14 23:43:40,023 - 2025-11-14_23-43-28_sensor_mapper - INFO - Adding tf_static: spot/odom, spot/body
        # 2025-11-14 23:43:40,023 - 2025-11-14_23-43-28_sensor_mapper - INFO - Adding tf_static: spot/body, head_right_rgbd_optical
        # 2025-11-14 23:43:40,023 - 2025-11-14_23-43-28_sensor_mapper - INFO - Adding tf_static: spot/body, head_left_rgbd_optical
        # 2025-11-14 23:43:40,023 - 2025-11-14_23-43-28_sensor_mapper - INFO - Adding tf_static: spot/body, left_rgbd_optical
        # 2025-11-14 23:43:40,023 - 2025-11-14_23-43-28_sensor_mapper - INFO - Adding tf_static: spot/body, right_rgbd_optical
        # 2025-11-14 23:43:40,023 - 2025-11-14_23-43-28_sensor_mapper - INFO - Adding tf_static: spot/body, rear_rgbd_optical

        # Static transforms - must be set before processing
        self.static_transforms = {
            "world_to_map": None,
            "map_to_odom": None,  
            "odom_to_body": None, 
            "body_to_opticals":{} # single transform (same for all cameras)
        }


    def add_tf_static(self, parent_frame: str, child_frame: str, position: Dict, orientation: Dict):
        """Add tf_static transform"""
        transform = {"position": position, "orientation": orientation}
        
        if parent_frame == "world" and child_frame == "map":
            self.static_transforms["world_to_map"] = transform
        if parent_frame == "map" and child_frame == "spot/odom":
            self.static_transforms["map_to_odom"] = transform
        if parent_frame == "spot/odom" and child_frame == "spot/body":
            self.static_transforms["odom_to_body"] = transform
        elif parent_frame == "spot/body" and child_frame.endswith("_optical"):
            self.static_transforms["body_to_opticals"][child_frame[:-8]] = transform

        # Check if we have all required static transforms
        self._check_tf_static_complete()
    
    def _check_tf_static_complete(self):
        """Check if all required tf_static transforms are available"""
        expected_cameras = {"head_left_rgbd", "head_right_rgbd", "left_rgbd", "right_rgbd", "rear_rgbd"}
        
        has_world_to_map = self.static_transforms["world_to_map"] is not None
        has_map_to_odom = self.static_transforms["map_to_odom"] is not None
        has_odom_to_body = self.static_transforms["odom_to_body"] is not None
        has_all_cameras = expected_cameras.issubset(set(self.static_transforms["body_to_opticals"].keys()))

        self.logger.info(f"expected cameras: {expected_cameras}")
        self.logger.info(f"static transforms we got: {set(self.static_transforms['body_to_opticals'].keys())}")
        self.logger.info(f"has_all_cameras: {has_all_cameras}")
        
        return has_world_to_map and has_map_to_odom and has_odom_to_body and has_all_cameras

    def _debug_tf_static_complete(self):
        """Debug tf_static complete"""
        print("TF static complete:")
        print(f"World to map: {self.static_transforms['world_to_map']}")
        print(f"Map to odom: {self.static_transforms['map_to_odom']}")
        print(f"Odom to body: {self.static_transforms['odom_to_body']}")
        print(f"Base link to cameras (optical): {self.static_transforms['body_to_opticals']}")

    def is_tf_static_ready(self):
        return self._check_tf_static_complete()

    def init_entry_if_not_exists(self, header_stamp: str):
        """Initialize entry if it doesn't exist. Automatically removes oldest entry if buffer is full."""
        if header_stamp not in self.entries:
            # If buffer is full, remove oldest entry (FIFO)
            if len(self.entries) >= self.max_size:
                self._remove_oldest_entry()
            
            # Add new entry (OrderedDict maintains insertion order)
            self.entries[header_stamp] = ObsDataEntry(header_stamp)
    
    def add_rgb(self, header_stamp: str, name: str, image: np.ndarray):
        """Add RGB image to entry"""
        self.init_entry_if_not_exists(header_stamp)
        self.entries[header_stamp].add_rgb(name, image)
    
    def add_depth(self, header_stamp: str, name: str, image: np.ndarray):
        """Add depth image to entry"""
        self.init_entry_if_not_exists(header_stamp)
        self.entries[header_stamp].add_depth(name, image)
    
    def add_odometry(self, header_stamp: str, odom_data: Dict):
        """Add odometry to entry"""
        self.init_entry_if_not_exists(header_stamp)
        self.entries[header_stamp].add_odometry(odom_data)
        self.logger.info(f"Added odometry to entry {header_stamp}, so now the buffer has {len(self.entries)} entries")
        self.logger.info(f"Now attempting to process entry.")
    
    def fill_in_odometry_if_not_present(self, entry):
        rgb_complete = entry.expected_rgb.issubset(set(entry.rgb_images.keys()))
        depth_complete = entry.expected_depth.issubset(set(entry.depth_images.keys()))
        has_odometry = entry.odometry is not None

        if rgb_complete and depth_complete and not has_odometry:
            self.logger.info(f"Filling in odometry for entry {entry.header_stamp} because it is not present and we have rgb and depth images")
            self.logger.info(f"rgb_keys {list(entry.rgb_images.keys())}")
            self.logger.info(f"depth_keys {list(entry.depth_images.keys())}")
            
            future_odom_ht = self.get_closest_future_odom(entry)
            if future_odom_ht is not None:
                entry.odometry = self.entries[str(future_odom_ht)].odometry
                self.logger.info(f"Filled in odometry for entry {entry.header_stamp} with future odom {future_odom_ht}")
            else:
                self.logger.info(f"No future odom found for entry {entry.header_stamp}")

    # def is_frame_full_v2(self, entry) -> bool:
    #     """Check if all expected sensor data is present"""
    #     rgb_complete = entry.expected_rgb.issubset(set(entry.rgb_images.keys()))
    #     depth_complete = entry.expected_depth.issubset(set(entry.depth_images.keys()))
    #     has_odometry = entry.odometry is not None
    #     print(f"V2: Frame has rgb_images {list(entry.rgb_images.keys())}")
    #     print(f"V2: Frame has depth_images {list(entry.depth_images.keys())}")
    #     print(f"V2: Frame has odometry {entry.odometry}")

    #     if rgb_complete and depth_complete and not has_odometry:
    #         future_odom_ht = self.get_closest_future_odom(entry)
    #         if future_odom_ht is not None:
    #             entry.odometry = self.entries[str(future_odom_ht)].odometry

    #     print(f"Is Full: {rgb_complete}, {depth_complete}, {has_odometry}")
    #     print(f"entry.is_frame_full = {entry.is_frame_full()}")

    #     return rgb_complete and depth_complete and has_odometry

    def get_closest_future_odom(self, entry):
        headerstamp = float(entry.header_stamp)
        all_timestamps_ixs = np.argsort(np.array(list(map(float, self.entries.keys()))))
        all_timestamps = np.array(list(map(float, self.entries.keys())))[all_timestamps_ixs]
        future_timestamps_ixs = all_timestamps_ixs[all_timestamps > headerstamp]
        if len(future_timestamps_ixs) > 0:
            return list(self.entries.keys())[future_timestamps_ixs[0]]
        else:
            return None

    def _remove_oldest_entry(self):
        """Remove oldest entry from buffer (FIFO). Handles processed/unprocessed entries."""
        if len(self.entries) == 0:
            return
        
        # Get oldest entry (first in OrderedDict)
        oldest_stamp = next(iter(self.entries))
        oldest_entry = self.entries[oldest_stamp]
        
        # Preferentially remove processed entries
        if oldest_entry.processed:
            del self.entries[oldest_stamp]
            self.logger.info(f"Removed oldest processed entry {oldest_stamp} (buffer full)")
            return
        
        # For unprocessed entries, check if they're complete
        if oldest_entry.is_frame_full():
            # Don't remove complete unprocessed entries - log warning
            if self._check_tf_static_complete():
                self.logger.info(f"[DATA BUFFER ERROR] TF static is complete, but entry {oldest_stamp} is not processed. Please check if you are processing each entry.")
                self.logger.info(f"[DATA BUFFER ERROR] Is frame full: {oldest_entry.is_frame_full()}")
            # Try to find next oldest that can be removed
            for stamp in list(self.entries.keys())[1:]:  # Skip the first one we already checked
                entry = self.entries[stamp]
                if entry.processed or not entry.is_frame_full():
                    del self.entries[stamp]
                    self.logger.info(f"Removed entry {stamp} (buffer full, oldest was unprocessed)")
                    return
            # If all remaining entries are full and unprocessed, we can't remove any
            self.logger.warning(f"[DATA BUFFER] Cannot remove any entries - all are full and unprocessed")
        else:
            # Remove incomplete entries
            del self.entries[oldest_stamp]
            self.logger.info(f"Removed oldest incomplete entry {oldest_stamp} (buffer full, speed issue)")

    def get_next_entry_to_process(self) -> Optional[ObsDataEntry]:
        """Get next entry to process"""
        self.logger.info("Trying to get next entry to process")

        if not self.is_tf_static_ready():
            self.logger.info("Not returning to process entry because tf static is not ready")
            return None

        for i, entry in enumerate(self.entries.values()):
            # self.fill_in_odometry_if_not_present(entry)
            if_frame_full = entry.is_frame_full()
            self.logger.info(f"Checking entry {i} with header stamp {entry.header_stamp}: is frame full? {if_frame_full} Is processed? {entry.processed}")
            if if_frame_full and not entry.processed:
                self.logger.info(f"Found full entry {entry.header_stamp}")
                return entry

        return None
    
    def get_entry_by_timestamp(self, header_stamp: str) -> Optional[ObsDataEntry]:
        """Get specific entry by timestamp"""
        return self.entries.get(header_stamp)
    
    def get_queue_len_of_to_process(self) -> int:
        """Get length of queue of to process entries"""
        return len([entry for entry in self.entries.values() if entry.is_frame_full() and not entry.processed])
    
    def get_buffer_status(self) -> Dict:
        """Get buffer status information"""
        complete_count = len([entry for entry in self.entries.values() if entry.is_frame_full()])
        processed_count = sum(1 for entry in self.entries.values() if entry.processed)
        
        return {
            "total_entries": len(self.entries),
            "complete_entries": complete_count,
            "processed_entries": processed_count,
            "entries_to_process": self.get_queue_len_of_to_process(),
            "tf_static_ready": self._check_tf_static_complete(),
            "buffer_size": self.max_size
        }
    
    def get_pointcloud_for_entry(self, header_stamp: str) -> Tuple[List[o3d.geometry.PointCloud], List[List[float]]]:
        """Get pointcloud and camera positions for entry"""
        return self.entries[header_stamp].get_pointcloud(self.static_transforms)
    
    def mark_processed(self, header_stamp: str):
        """Mark entry as processed"""
        self.entries[header_stamp].set_processed()
    
    def delete_processed_entry(self, header_stamp: str):
        """Delete a processed entry to free memory"""
        if header_stamp in self.entries and self.entries[header_stamp].processed:
            del self.entries[header_stamp]
            self.logger.info(f"Deleted processed entry {header_stamp}")
            return True
        return False
    
    def delete_entries_before_timestamp(self, timestamp: str, preserve_full  = False):
        """Delete all entries before a given timestamp"""
        for stamp, entry in self.entries.items():
            if float(stamp) < float(timestamp):
                if preserve_full  and entry.is_frame_full():
                    continue
                del self.entries[stamp]
                self.logger.info(f"Deleted entry {stamp} because it is before {timestamp}")
                return True
        return False
    
    def cleanup_processed_entries(self):
        """Remove all processed entries to free memory"""
        processed_stamps = [stamp for stamp, entry in self.entries.items() if entry.processed]
        for stamp in processed_stamps:
            del self.entries[stamp]
        return len(processed_stamps)

    # def get_occupancy_grid(self, pcd: o3d.geometry.PointCloud, resolution: float = 0.1):
    #     """Generate occupancy grid using the get_occ_grid function."""
    #     return get_occ_and_height_map(pcd, resolution)

