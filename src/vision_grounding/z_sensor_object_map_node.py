#!/usr/bin/env python3
"""
Real-time sensor mapper that maintains a live point cloud map of the world.
Processes sensor data as it comes in without dumping frames to disk.

update for new container augmented with object pipeline

initial 10/31/25
version 2.0: 11/17/2025
Updated: 11/21/2025

Author: Robin Eshraghi

"""
import os
import time
import threading
import json
from pathlib import Path
from functools import wraps
from importlib.util import find_spec
from typing import Dict
import pickle as pk
import copy
# Get logs folder path (same as session_logger)
LOGS_FOLDER = Path(os.path.dirname(os.path.dirname(find_spec('tiamat_agent').origin)), "logs")
OUTPUT_DIR = LOGS_FOLDER / "current_run_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration for mask saving in image dumps
SAVE_MASKS_IN_DUMPS = True  # Set to False to disable mask saving (reduces file size)

# ROS2 imports - optional for debug mode
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, Image
    from nav_msgs.msg import Odometry  # OccupancyGrid, MapMetaData
    from tf2_msgs.msg import TFMessage
    from std_msgs.msg import Header
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    # Create minimal mock classes for debug mode
    class Node:
        def __init__(self, name):
            self.name = name
        def get_logger(self):
            import logging
            logger = logging.getLogger(self.name)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                logger.addHandler(handler)
            return logger
    class CvBridge:
        pass
    ROS2_AVAILABLE = False
import cv2
import numpy as np
import traceback
import matplotlib
from src.utils.session_logger import SessionLogger

# Check if tkinter is available, if not use Agg backend to avoid errors
try:
    import tkinter
except ImportError:
    # tkinter not available, use non-interactive Agg backend
    matplotlib.use('Agg')

# Import pyplot - if backend still fails, catch and use Agg
try:
    import matplotlib.pyplot as plt
except (ImportError, RuntimeError) as e:
    # If backend fails for any reason, fall back to Agg
    error_str = str(e).lower()
    if 'tkinter' in error_str or 'tk' in error_str or 'no display' in error_str or 'backend' in error_str:
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    else:
        raise


from src.vision_grounding.obs_data_buffer import ObsDataBuffer, compose_transforms_optimized, ObsDataEntry
# import src.vision_grounding.object_grounding_viz as object_grounding_viz
from src.vision_grounding.object_detection_pipeline import (
        initialize_models as initialize_object_models,
        parse_rgbd_image_dicts_for_objects,
        FUNCTION_CONFIGS as OBJECT_PIPELINE_CONFIGS,
        calculate_statistics as calculate_object_statistics,
        extract_robot_pose,
        reset_color_index,
        get_color_index,
        save_segmented_pointcloud,
        save_individual_object_pointclouds,
        build_scene_pointcloud,
        save_pointcloud,
    )
# Import coordinate transformation helpers
from tiamat_agent.mapping.occupancy_grid import get_robot_world_coords, quaternion_to_yaw
from tiamat_agent.data.config.yaml_utils import load_config


def stamp_to_str(msg):
    """Convert ROS message timestamp to string"""
    if hasattr(msg, "header") and msg.header.stamp.sec != 0:
        return f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
    else:
        return f"{int(time.time())}.000000000"

from munch import Munch
import shutil

class ObjectLibrary:
    """
    Object library that maintains a list of detected objects.
    """
    def __init__(self):
        self.objects_by_id = {}
    
    def add_object(self, object_data_dict):
        obj_id = len(self.objects_by_id)
        self.objects_by_id[obj_id] = object_data_dict
        self.objects_by_id[obj_id]["object_id"] = obj_id

    
    def add_objects(self, objects_data_dicts):
        for object_data_dict in objects_data_dicts:
            self.add_object(object_data_dict)
    
    def get_object_by_id(self, object_id):
        return self.objects_by_id[object_id]
    
    def __len__(self):
        return len(self.objects_by_id)  

    def __iter__(self):
        return iter(self.objects_by_id.values())

    def dump_object_list(self, folder_path: Path):
        """Dump object list to file"""
        folder_path.mkdir(parents=True, exist_ok=True)
        with open(folder_path / f"object_list_{time.time()}.json", "w") as f:
            json.dump(self.objects_by_id, f)

    def gen_object_records_for_viz_pipe(self):
        """
        Generate object records formatted for visualization pipeline.
        
        Returns a list of object records with only the minimal fields needed by 
        parse_object_records in plotters.py:
        - spatial_metadata: position_3d, is_valid_depth
        - semantic_metadata: label
        - frame_metadata: robot_position, robot_yaw
        
        Returns:
            List[Dict]: List of filtered object records ready for visualization
        """
        object_records = []
        for obj_id, obj in self.objects_by_id.items():
            # Extract only the required fields for visualization
            spatial_meta = obj.get("spatial_metadata", {})
            semantic_meta = obj.get("semantic_metadata", {})
            frame_meta = obj.get("frame_metadata", {})
            
            # Create minimal record with only needed fields
            viz_record = {
                "spatial_metadata": {
                    "position_3d": spatial_meta.get("position_3d"),
                    "is_valid_depth": spatial_meta.get("is_valid_depth", False)
                },
                "semantic_metadata": {
                    "label": semantic_meta.get("label", "unknown")
                },
                "frame_metadata": {
                    "robot_position": frame_meta.get("robot_position"),
                    "robot_yaw": frame_meta.get("robot_yaw")
                }
            }
            object_records.append(viz_record)
        return object_records

    


from src.utils.func_utils import time_fn

class ObjectSensorMapper(Node):
    """
    Real-time sensor mapper that maintains a live point cloud map.
    
    Features:
    - Processes sensor data in real-time without disk I/O
    - Maintains a growing point cloud map of the world
    - Periodically downsamples the map to prevent memory issues
    - Periodically dumps the map to disk for persistence
    - Uses ObsDataBuffer for efficient data management
    """
    
    def __init__(self, mapping_config: Munch, debug_mode: bool = False):
        # This calls the constructor of the parent class (Node) and names this ROS2 node "sensor_mapper".
        # It is required to initialize the ROS2 node infrastructure, such as publishers, subscribers, and logging.
        self.debug_mode = debug_mode or not ROS2_AVAILABLE

        self.config = mapping_config
        
        if ROS2_AVAILABLE and not self.debug_mode:
            super().__init__("sensor_mapper")
        # In debug mode or if ROS2 not available, skip Node initialization

        # setup session logging
        self.session_logger = SessionLogger(mapping_config.session_id, "sensor_object_mapper")
        self.logger = self.session_logger.get_logger()
        
        # Initialize components
        if ROS2_AVAILABLE and not self.debug_mode:
            self.bridge = CvBridge()
        else:
            # In debug mode, CvBridge is not needed
            self.bridge = None
        
        self.logger.info("SensorMapper initialization started for real-time mapping")
        
        # ---- object detection buckets ---- #
        # obsdatabuffer session logger passed by reference 
        self.buffer = ObsDataBuffer(session_logger=self.session_logger.get_child_logger("obs_data_buffer"), max_size=mapping_config.max_buffer_size)  # Keep buffer small for real-time processing
        self.robot_odom = None # robot odometry
        self.robot_odom_timestamp = None

        
        # config vars 
        self.session_id = mapping_config.session_id # session id for the mapping session
        # self.resolution = mapping_config.resolution
        # self.voxel_size = mapping_config.voxel_size
        # self.downsample_interval = mapping_config.downsample_interval
        self.stats_interval = mapping_config.stats_interval # Print stats every 5 seconds (keep for object stats)
        self.sensor_config = load_config("sensor_topics.yaml")["sensors"]


        # Housekeeping variables
        self.processed_frames = 0 # processed frames
        self.most_recent_processed_timestep = 0 # most reset processed timestep =
        self.map_lock = threading.Lock()
        # self.buffer_lock = threading.Lock()  # Lock for thread-safe buffer operations
        self.subscriptions_paused = False  # Flag to pause subscription callbacks
        # self.total_points_added = 0  # Mapping statistics, not needed
        # self.last_downsample_time = time.time()  # Mapping maintenance, not needed
        self.last_stats_time = time.time()
        
        
        
        # Object detection state
        self.object_library = ObjectLibrary()
        self.copy_of_object_library = ObjectLibrary()
        self.all_image_data_by_id = {}  # Accumulate image data across all entries
        self.global_color_offset = 0  # Track color offset across all entries for unique colors

        # o_seed_path = "tiamat_agent/data/saved_maps/object_library_seed_data_dec_3.pkl"
        # self.object_library = pk.load(open(o_seed_path, "rb"))
        # self.copy_of_object_library = pk.load(open(o_seed_path, "rb"))
        # self.logger.info(f"[OBJECT MAPPER] Seed node loaded from {o_seed_path}, {len(self.object_library)} objects loaded")


        self.vlm_detector, self.yolo_model, self.fastsam_model = initialize_object_models(logger = self.logger)
        if self.vlm_detector is None or self.yolo_model is None:
            self.logger.error("Failed to initialize object detection models.")
            import traceback
            traceback.print_exc()
            raise RuntimeError("Object detection models failed to initialize")

        self.object_output_path = Path("object_output")
        self.object_output_path.mkdir(parents=True, exist_ok=True)

        # Object detection config
        self.object_processing_config = OBJECT_PIPELINE_CONFIGS
        self.object_batch_size = 20  # default batch size for real-time processing

        # RGB frame saving config (for offline visualization)
        self.save_rgb_frames = getattr(mapping_config, "save_rgb_frames", False)
        self.rgb_output_root = Path("logs/detection_images")
        if self.save_rgb_frames:
            # delete the folder if it exists
            if self.rgb_output_root.exists():
                shutil.rmtree(self.rgb_output_root)
            # create the folder
            self.rgb_output_root.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"[RGB SAVING] Enabled - saving to {self.rgb_output_root.absolute()}")

        # Initialize ZMQ publisher for detection status
        try:
            from tiamat_agent.agent_viz.zmq_publish import DetectionStatusServer
            self.detection_status_server = DetectionStatusServer(host=mapping_config.viz_host, port=mapping_config.viz_port)
            self.logger.info(f"[DETECTION STATUS] DetectionStatusServer initialized on {mapping_config.viz_host}: {mapping_config.viz_port}")
        except Exception as e:
            self.logger.warning(f"[DETECTION STATUS] Failed to initialize DetectionStatusServer: {e}")
            self.detection_status_server = None

        # Set up subscriptions (skip in debug mode)
        if not debug_mode:
            self._setup_subscriptions()
        self.logger.info("SensorMapper initialized and ready")

        self.image_callback_count = 0
        

    @staticmethod
    def _with_map_lock(fn):
        """Decorator to run a method under map_lock and pass copies of attributes as needed."""
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            # return fn(self, *args, **kwargs)
            with self.map_lock:
                return fn(self, *args, **kwargs)
        return wrapper
    
    @staticmethod
    def _check_subscriptions_paused(fn):
        """Decorator to skip callback execution if subscriptions are paused."""
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            if self.subscriptions_paused:
                return
            return fn(self, *args, **kwargs)
        return wrapper
    
    @staticmethod
    def _time_handle(handle_name: str):
        """Decorator factory to time handle function execution.
        
        Args:
            handle_name: Name to use in log messages (e.g., "RGB", "Depth", "Odom", "TF")
        """
        def decorator(fn):
            @wraps(fn)
            def wrapper(self, *args, **kwargs):
                start_time = time.time()
                try:
                    result = fn(self, *args, **kwargs)
                    elapsed = time.time() - start_time
                    self.logger.info(f"[TIMING] {handle_name} handle took {elapsed:.5f}s")
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    self.logger.error(f"[TIMING] {handle_name} handle failed after {elapsed:.5f}s: {e}")
                    raise
            return wrapper
        return decorator
    
    
    # TODO: [Low Priority] [Soham] Figure out if any topics are handled by odom_fast that need not be handled here.
    def _setup_subscriptions(self):
        """Set up all ROS2 subscriptions"""
        
        # Subscribe to sensor topics
        for group, entries in self.sensor_config.items():
            for data_type in ("rgb", "depth"):
                if data_type not in entries:
                    self.logger.info(f"No {data_type} topics found for group {group}")
                    continue

                for name, topic in entries[data_type].items():                    
                    if data_type == "rgb":
                        rgb_callback = lambda msg, name=name: self._handle_rgb(msg, name)
                        self.create_subscription( CompressedImage, topic, rgb_callback, 25)
                        self.logger.info(f"Created Subscription for Topic: {topic}, Read for name {name} | Data Type: {data_type}")
                    elif data_type == "depth":
                        depth_callback = lambda msg, name=name: self._handle_depth(msg, name)
                        self.create_subscription(Image, topic, depth_callback, 25)
                        self.logger.info(f"Created Subscription for Topic: {topic}, Read for name {name} | Data Type: {data_type}")

        # Subscribe to odometry
        if "platform" in self.sensor_config and "odometry" in self.sensor_config["platform"]:
            odom_topic = self.sensor_config["platform"]["odometry"]["odom"]
            self.create_subscription(Odometry, odom_topic, self._handle_odom, 5)
            self.logger.info(f"Created Subscription for Topic: {odom_topic}, Read for name {odom_topic} | Data Type: {data_type}")

        # Subscribe to odometry for process callback
        if "platform" in self.sensor_config and "odometry" in self.sensor_config["platform"]:
            odom_topic = self.sensor_config["platform"]["odometry"]["odom"]
            self.create_subscription(Odometry, odom_topic, self._process_entry_callback, 2)
            self.logger.info(f"Created Subscription for Topic: {odom_topic}, Read for name {odom_topic} | Data Type: {data_type}")
        
        # Subscribe to TF topics
        self.create_subscription(TFMessage, "/tf", 
                               lambda msg: self._handle_tf(msg, "tf"), 500)
        self.logger.info("Created Subscription for Topic: /tf, Read for name tf | Data Type: tf")

        # Subscribe to TF static topics
        self.create_subscription(TFMessage, "/tf_static", 
                               lambda msg: self._handle_tf(msg, "tf_static"), 500)
        self.logger.info("Created Subscription for Topic: /tf_static, Read for name tf_static | Data Type: tf_static")
        
    @_check_subscriptions_paused
    @_time_handle("RGB")
    def _handle_rgb(self, msg, name: str):
        """Handle incoming RGB message"""
        self.logger.info(f"Handling RGB message for name: {name}")
        try:
            #print("Handling RGB", name)
            header_stamp = stamp_to_str(msg)
            self.logger.info(f"Received RGB with header stamp: {header_stamp}")

            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            rgb_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB) # need to convert to rgb image (checked that imdecode is giving bgr image) 
           
            
            if rgb_image is not None:
                # Add to buffer (no disk I/O)
                self.buffer.add_rgb(header_stamp, name, rgb_image)
                self.logger.info(f"Added RGB image with name {name} to buffer with header stamp {header_stamp}")
                self.logger.info(f"TF static ready status: {self.buffer.is_tf_static_ready()}")

                # Update matplotlib visualization
                # self._update_rgb_visualization(name, rgb_image)

        except Exception as e:
            self.logger.error(f"Error processing RGB image with name {name}: {e}")
            import traceback
            traceback.print_exc()


    @_check_subscriptions_paused
    @_time_handle("Depth")
    def _handle_depth(self, msg, name: str):
        """Handle incoming depth message"""
        self.logger.info(f"Handling Depth message for name: {name}")
        # print he queue backlog for this topic rgb topci

        try:
            header_stamp = stamp_to_str(msg)
            self.logger.info(f"Received Depth with header stamp: {header_stamp}")

            # Convert depth image
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            if depth_image is not None:
                # Add to buffer (no disk I/O) - protected by buffer_lock
                self.buffer.add_depth(header_stamp, name, depth_image)
                self.logger.info(f"Added depth image with name {name} to buffer with header stamp {header_stamp}")
                self.logger.info(f"TF static ready status: {self.buffer.is_tf_static_ready()}")

            else:
                self.logger.info("Buffer is not TF static ready!, waiting for next entry")

        except Exception as e:
            self.logger.error(f"Error processing depth {name}: {e}")
            import traceback
            traceback.print_exc()

    
    def get_next_entry_to_process(self, buffer):
        """Get next entry to process"""
        self.logger.info("Trying to get next entry to process")

        if not buffer.is_tf_static_ready():
            self.logger.info("Not returning to process entry because tf static is not ready")
            return None

        latest_entry = None
        latest_timestamp = float("-inf")

        for i, entry in enumerate(buffer.entries.values()):
            # self.fill_in_odometry_if_not_present(entry)
            if_frame_full, (rgb_complete, depth_complete, odom_complete) = entry.is_frame_full(return_info=True)

            # Use mapping_data to backfill synchronized odometry
            if hasattr(self, "mapping_data"):
                if rgb_complete and depth_complete:
                    if entry.header_stamp in self.mapping_data:
                        entry.add_odometry(self.mapping_data[entry.header_stamp])
                        if_frame_full, (rgb_complete, depth_complete, odom_complete) = entry.is_frame_full(return_info=True)
                        self.logger.info(f"Backfilled odometry for entry {entry.header_stamp} from mapping_data")

            self.logger.info(f"Checking entry {i} with header stamp {entry.header_stamp}: is frame full? {if_frame_full} Is processed? {entry.processed}")
            if if_frame_full and not entry.processed:
                entry_timestamp = float(entry.header_stamp)
                if entry_timestamp > latest_timestamp:
                    latest_timestamp = entry_timestamp
                    latest_entry = entry
            else:
                self.logger.info(f"Entry {i} has frame full status: {if_frame_full} and processed status: {entry.processed}")
                self.logger.info(f"Entry {i} has rgb complete status: {rgb_complete} and depth complete status: {depth_complete} and odometry complete status: {odom_complete}")

        if latest_entry is not None:
            self.logger.info(f"Found latest full entry {latest_entry.header_stamp}")
            latest_entry.static_transforms = self.buffer.static_transforms.copy()
        else:
            self.logger.info("No full entries available to process")

        return latest_entry
    
            
    def _process_entry_callback(self, msg):
        st = time.time()
        entry = self.get_next_entry_to_process(self.buffer)
        self.logger.info(f"Time taken to check for next entry at callback {time.time()-st:5f}")

        if entry is None:
            return 

        self.logger.info(f"Found entry to process: {entry.header_stamp}")
        st = time.time()
        self._process_entry_for_objects(entry, dump_entry=True)

        self.most_recent_processed_timestep = max(float(self.most_recent_processed_timestep), float(entry.header_stamp))
        self.logger.info(f"Processed entry {entry.header_stamp} in {time.time() - st:.3f}s")

        self.logger.info(f"Calling Delete entries before timestamp: {self.most_recent_processed_timestep}")
        self.buffer.delete_entries_before_timestamp(self.most_recent_processed_timestep, preserve_full = False)

        self.logger.info(f"Calleing Delete processed entry: {entry.header_stamp}")
        self.buffer.delete_processed_entry(entry.header_stamp)

        self.logger.info(f"Calleing periodic maintenance")
        self._periodic_maintenance()
        self.logger.info(f"Buffer status: {self.buffer.get_buffer_status()}")

    @_check_subscriptions_paused
    @_time_handle("Odom")
    def _handle_odom(self, msg):
        """Handle incoming odometry message"""
        self.logger.info("Adding Odom ...")
        try:
            self.logger.info("Handling Odom")
            header_stamp = stamp_to_str(msg)
            self.logger.info(f"Adding Odom at timestamp: {header_stamp}")

            odom_data = {
                "position": {
                    "x": msg.pose.pose.position.x,
                    "y": msg.pose.pose.position.y,
                    "z": msg.pose.pose.position.z
                },
                "orientation": {
                    "x": msg.pose.pose.orientation.x,
                    "y": msg.pose.pose.orientation.y,
                    "z": msg.pose.pose.orientation.z,
                    "w": msg.pose.pose.orientation.w
                }
            }
            
            # Add to buffer
            self.buffer.add_odometry(header_stamp, odom_data)
            self.robot_odom = odom_data # dict with keys position and orientation
            self.robot_odom_timestamp = header_stamp


        except Exception as e:
            self.logger.error(f"Error processing odometry: {e}")
            import traceback
            traceback.print_exc()

    @_check_subscriptions_paused
    @_time_handle("TF")
    def _handle_tf(self, msg, tf_type: str):
        """Handle incoming TF messages"""
        try:
            # Only process tf_static for our buffer
            if tf_type == "tf_static":
                self.logger.info(f"Handling TF STATIC: {tf_type}")
                for transform in msg.transforms:
                    self.logger.info(f"Adding tf_static {transform.header.frame_id} {transform.child_frame_id}")
                    position = {
                        "x": transform.transform.translation.x,
                        "y": transform.transform.translation.y,
                        "z": transform.transform.translation.z
                    }
                    orientation = {
                        "x": transform.transform.rotation.x,
                        "y": transform.transform.rotation.y,
                        "z": transform.transform.rotation.z,
                        "w": transform.transform.rotation.w
                    }
                    
                    # Add to buffer
                    self.buffer.add_tf_static(
                        transform.header.frame_id,
                        transform.child_frame_id,
                        position,
                        orientation
                    )

                    self.logger.info(f"TF statics: {str(self.buffer.static_transforms)}")
            else:
                for transform in msg.transforms:
                    self.logger.info(f"Ignoring tf {transform.header.frame_id} {transform.child_frame_id}")
                
        except Exception as e:
            self.logger.error(f"Error processing TF: {e}")
            import traceback
            traceback.print_exc()


    def _split_full_entry_for_object_detection(self, entry):
        """Convert ObsDataEntry to pair dictionaries expected by pipeline."""
        if not entry.is_frame_full():
            self.logger.error(f"Entry {entry.header_stamp} is not full, skipping object detection")
            return None
        
        self.logger.info(f"Splitting full entry for object detection: {entry.header_stamp}")
        # Camera rotation mapping: rotate images before any processing
        # TODO: Do not hard code this later, hardcoded for now.
        ROTATION_MAP = {
            "rear_rgb": {"forward": None, "backward": None},
            "left_rgb": {"forward": None, "backward": None},
            "right_rgb": {"forward": cv2.ROTATE_180, "backward": cv2.ROTATE_180},
            "head_rgb_left": {"forward": cv2.ROTATE_90_CLOCKWISE, "backward": cv2.ROTATE_90_COUNTERCLOCKWISE},
            "head_rgb_right": {"forward": cv2.ROTATE_90_CLOCKWISE, "backward": cv2.ROTATE_90_COUNTERCLOCKWISE},
        }

        rgbd_image_dicts = []
        for (rgb_name, rgb_image, depth_name, depth_image) in entry.get_rgb_depth_pairs():
            # Apply rotation if needed (before any processing, saving, or detection)
            if ROTATION_MAP[rgb_name]["forward"] is not None:
                rotated_rgb_image = cv2.rotate(rgb_image, ROTATION_MAP[rgb_name]["forward"])
            else:
                rotated_rgb_image = rgb_image
                # depth_image = cv2.rotate(depth_image, ROTATION_MAP[rgb_name])
            
            curr_image_data = {
                "image_id":f"{entry.header_stamp}/{rgb_name}",
                "rgb_name": rgb_name,
                "rgb_image": rgb_image, 
                "rotated_rgb_image": rotated_rgb_image, # Now rotated if needed
                "depth_name": depth_name,
                "depth_image": depth_image,
                "timestamp": entry.header_stamp,
                "frame_id": self.processed_frames,
                "odometry": entry.odometry,
                "static_transforms": entry.static_transforms,
                "rotation_map": ROTATION_MAP[rgb_name],
            }
            # ------------------------------------------------------------
            # Get Camera and Robot Pose
            # ------------------------------------------------------------
            curr_image_data["w2c_transform"] = compose_transforms_optimized(curr_image_data['odometry'],
                                                                        ObsDataEntry.get_camera_mapping(curr_image_data['rgb_name']),
                                                                        curr_image_data['static_transforms'],
                                                                        use_optical=True
                                                                    )
            curr_image_data['camera_position'] = [curr_image_data["w2c_transform"]["position"][c] for c in ["x", "y", "z"]]
            curr_image_data['camera_orientation'] = [curr_image_data["w2c_transform"]["orientation"][c] for c in ["x", "y", "z", "w"]]

            robot_pose = extract_robot_pose(curr_image_data['odometry'], curr_image_data['static_transforms'])
            curr_image_data['robot_position'] = robot_pose['robot_position']
            curr_image_data['robot_orientation'] = robot_pose['robot_orientation']
            curr_image_data['robot_yaw'] = robot_pose['robot_yaw']


            rgbd_image_dicts.append(curr_image_data)
            self.logger.info(f"{rgb_name} image shape: {rgb_image.shape}")
            self.logger.info(f"{depth_name} image shape: {depth_image.shape}")

        return rgbd_image_dicts

    def _write_rgbd_image_dicts_to_disk(self, rgbd_image_dicts):
        """Write rgbd image dicts to disk"""
        import pickle as pk
        # Create the directory if it doesn't exist
        rgbd_image_dicts_dir = OUTPUT_DIR / "rgbd_image_dict_dumps"
        rgbd_image_dicts_dir.mkdir(parents=True, exist_ok=True)
        
        for rgbd_image_dict in rgbd_image_dicts:
            rgbd_image_dicts_path = rgbd_image_dicts_dir / f"rgbd_image_dict_{time.time()}.pkl"
            with open(rgbd_image_dicts_path, "wb") as f:
                pk.dump(rgbd_image_dicts, f)
            self.logger.info(f"Wrote rgbd image dicts to {rgbd_image_dicts_path}")

    def _process_entry_for_objects(self, entry, dump_entry: bool = False):
        """Run object detection pipeline on a single ObsDataEntry."""
        self.logger.info(f"[OBJECT DETECTION] Starting detection for entry {entry.header_stamp}")
        self.logger.info(f"Publishing detection status for entry {entry.header_stamp} with buffer status: {self.buffer.is_tf_static_ready()}")
        main_start_time = time.time()

        if dump_entry:
            # dump entry in log folder in a sepaate foldre named with time.time()
            entry_dumps_dir = OUTPUT_DIR / "entry_dumps"
            entry_dumps_dir.mkdir(parents=True, exist_ok=True)
            entry_dump_path = entry_dumps_dir / f"entry_{time.time()}.pkl"
            with open(entry_dump_path, "wb") as f:
                pk.dump(entry, f)
            self.logger.info(f"Dumped entry to {entry_dump_path} and deleted all files in entry_dumps folder")

        try:
            rgbd_image_dicts = self._split_full_entry_for_object_detection(entry)
            time_fn(self.logger, self._write_rgbd_image_dicts_to_disk, rgbd_image_dicts)

            # Set module-level color index before processing
            reset_color_index(self.global_color_offset)
            
            image_data_by_id, object_data_dicts = parse_rgbd_image_dicts_for_objects(
                rgbd_image_dicts=rgbd_image_dicts,
                vlm_detector=self.vlm_detector,
                yolo_model=self.yolo_model,
                fastsam_model=self.fastsam_model,
                max_batch_size=20,
                logger=self.logger
            )
            
            # Get updated color index after processing
            self.global_color_offset = get_color_index()
            
            # Accumulate image data across all entries for later segmented PC saving
            self.all_image_data_by_id.update(image_data_by_id)
            # Dump image_data_by_id for offline visualization
            self._dump_image_data_by_id(image_data_by_id, entry.header_stamp)
            self.object_library.add_objects(object_data_dicts)
            time_fn(self.logger, self.object_library.dump_object_list, folder_path=OUTPUT_DIR / "object_list_dumps")
            time_fn(self.logger, self._write_object_statistics)
            time_fn(self.logger, self._write_object_list)
            self.copy_of_object_library = copy.deepcopy(self.object_library)
            self.logger.info(f"[OBJECT DETECTION] Detected {len(object_data_dicts)} objects (total library: {len(self.object_library)})")

            # Save annotated composite frame (if enabled)
            # if self.save_rgb_frames:
            #     time_fn(self.logger, self._save_annotated_composite, rgbd_image_dicts, object_data_dicts, self.processed_frames)

            self.processed_frames += 1
            entry.set_processed()
            main_time_taken = time.time() - main_start_time
            self.logger.info(
                f"[OBJECT DETECTION] Entry {entry.header_stamp}: "
                f"{len(rgbd_image_dicts)} pairs processed, {len(object_data_dicts)} objects detected in {main_time_taken:.3f}s"
            )

            # Publish detection status for visualization (right before detection)
            if self.detection_status_server is not None:
                batch_info = {"status": "processing", "pairs": [], "queue_positions": [], "object_records": self.object_library.gen_object_records_for_viz_pipe()}
                self.logger.info(f"SEnding object records fo length: {len(batch_info['object_records'])}")
                for rgbd_image_dict in rgbd_image_dicts:
                    batch_info["pairs"].append({
                        "timestamp": rgbd_image_dict['timestamp'],
                        "camera_name": rgbd_image_dict['rgb_name'],
                        "robot_position": rgbd_image_dict['robot_position'],
                        "robot_yaw": rgbd_image_dict['robot_yaw'],
                        "camera_position": rgbd_image_dict['camera_position'],
                        "camera_orientation": rgbd_image_dict['camera_orientation'],
                    })
            
                try:
                    self.detection_status_server.send_batch(batch_info)
                    print(f"Published {len(batch_info['pairs'])} pairs for entry {entry.header_stamp}")
                    self.logger.info(f"[DETECTION STATUS] Published {len(batch_info['pairs'])} pairs for entry {entry.header_stamp}")
                except Exception as e:
                    self.logger.warning(f"[DETECTION STATUS] Failed to publish batch info: {e}")
                    self.logger.error(f"Error traceback: {traceback.format_exc()}")
                    traceback.print_exc()
            return True

        except Exception as e:
            self.logger.error(f"[OBJECT DETECTION] Error processing entry {entry.header_stamp}: {e}\n")
            self.logger.error(f"Error traceback: {traceback.format_exc()}")
            traceback.print_exc()
            return False

    def _write_object_statistics(self):
        """Persist object statistics for inspection."""
        if not self.object_library:
            return
        try:
            stats = calculate_object_statistics(self.object_library)
            output = {
                "metadata": {
                    "total_objects": len(self.object_library),
                    "processed_frames": self.processed_frames,
                    "timestamp": time.time(),
                },
                "statistics": stats,
            }
            stats_path = OUTPUT_DIR / "object_statistics.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)
        except Exception as exc:
            self.logger.error(f"Failed to write object statistics: {exc}")
    
    def _write_object_list(self):
        """Persist full object list with all metadata including robot position/yaw."""
        if not self.object_library:
            return
        try:
            # Calculate statistics
            statistics = calculate_object_statistics(self.object_library)
            
            # Prepare metadata
            output = {
                "metadata": {
                    "total_objects": len(self.object_library),
                    "processed_frames": self.processed_frames,
                    "last_updated": time.time(),
                    "session_id": self.session_id,
                    "processing_mode": "real_time",
                },
                "objects": list(self.object_library),  # Full list with all object metadata
                "statistics": statistics,
            }
            
            object_list_path = OUTPUT_DIR / "object_list.json"
            with open(object_list_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, default=str)  # default=str handles numpy types
            
            self.logger.info(f"[OBJECT LIST] Saved {len(self.object_library)} objects to {object_list_path}")
        except Exception as exc:
            self.logger.error(f"Failed to write object list: {exc}")
            import traceback
            traceback.print_exc()

    def _dump_image_data_by_id(self, image_data_by_id: Dict, entry_header_stamp: str):
        """Dump image_data_by_id to pickle file for offline visualization.
        
        Only saves essential data for visualization (images, bboxes, labels)
        NO complex objects like Open3D point clouds and YOLO models.
        """
        if OUTPUT_DIR.name != "offline_outputs":
            # Only dump in offline mode
            return
        
        try:
            # Create subdirectory for image data dumps
            image_data_dir = OUTPUT_DIR / "image_data_dumps"
            image_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save with entry timestamp for easy matching (replace / with _ for filename safety)
            safe_stamp = entry_header_stamp.replace('/', '_')
            dump_path = image_data_dir / f"image_data_{safe_stamp}.pkl"
            
            # Extract only visualization-essential data (avoid complex objects like point clouds)
            viz_data = {}
            for image_id, image_data in image_data_by_id.items():
                viz_data[image_id] = {
                    'image_id': image_data.get('image_id'),
                    'rgb_image': image_data.get('rgb_image'),
                    'rotated_rgb_image': image_data.get('rotated_rgb_image'),
                    'rgb_name': image_data.get('rgb_name'),
                    'timestamp': image_data.get('timestamp'),
                    'yolo_object_dict': {}
                }
                
                # Extract only essential fields from yolo_object_dict
                yolo_objects = image_data.get('yolo_object_dict', {})
                for obj_id, obj_data in yolo_objects.items():
                    #viz_data[image_id]['yolo_object_dict'][obj_id] = {
                    obj_dict = {
                        'bbox_xyxy': obj_data.get('bbox_xyxy'),
                        'rotated_bbox_xyxy': obj_data.get('rotated_bbox_xyxy'),
                        'label': obj_data.get('label'),
                        'confidence': obj_data.get('confidence'),
                        'description': obj_data.get('description'),
                        'class_id': obj_data.get('class_id'),
                    }
                    
                    # Optionally save FastSAM masks for visualization
                    if SAVE_MASKS_IN_DUMPS and 'scaled_bbox_masks' in obj_data:
                        mask = obj_data.get('scaled_bbox_masks')
                        if mask is not None:
                            # Convert bool to uint8 for efficient pickle storage
                            obj_dict['scaled_bbox_masks'] = mask.astype(np.uint8)
                            # Flag to indicate if it's FastSAM (True) or fallback center mask (False)
                            obj_dict['mask_is_fastsam'] = (
                                image_data.get("raw_sam_rotated_detections") is not None
                            )
                    
                    viz_data[image_id]['yolo_object_dict'][obj_id] = obj_dict
            
            with open(dump_path, 'wb') as f:
                # Use protocol 4 for better compatibility and to handle large files
                pk.dump(viz_data, f, protocol=4)
            
            # Verify the file was written correctly
            file_size = dump_path.stat().st_size
            self.logger.info(f"[IMAGE DATA DUMP] Saved {len(viz_data)} images to {dump_path} ({file_size / 1024 / 1024:.2f} MB)")
        except Exception as e:
            self.logger.warning(f"[IMAGE DATA DUMP] Failed to save: {e}")
            import traceback
            traceback.print_exc()

    def _periodic_maintenance(self):
        """Perform periodic maintenance tasks"""
        current_time = time.time()
        
        # Print statistics periodically
        if current_time - self.last_stats_time > self.stats_interval:
            self._print_statistics()
            self.last_stats_time = current_time

    
    # ❌ TODO: REPLACE with object detection statistics
    @_with_map_lock
    def _print_statistics(self):
        """Print statistics - TODO: Replace with object detection stats"""
        buffer_status = self.buffer.get_buffer_status()
        object_count = len(self.object_library)
        self.logger.info(
            f"Object Detection Stats - Processed: {self.processed_frames} frames, "
            f"Objects detected: {object_count}, "
            f"Buffer: {buffer_status['complete_entries']}/{buffer_status['total_entries']} ready, {buffer_status['entries_to_process']} to process "
            f"TF ready: {buffer_status['tf_static_ready']}, "
        )
    


    def get_robot_odom(self):
        """Access the current robot pose"""
        return self.robot_odom
    
    def get_static_transforms(self):
        """Access the current static transforms"""
        return self.buffer.static_transforms

    def destroy_node(self, save_final_map: bool = False):
        """Clean shutdown"""
        self.processing_active = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)
        
        # Close detection status server
        if hasattr(self, 'detection_status_server') and self.detection_status_server is not None:
            self.detection_status_server.close()
            self.logger.info("[SHUTDOWN] Closed DetectionStatusServer")

        
        super().destroy_node()

    def dump_buffer(self, filename="final_buffer_dump.pkl"):
        """Dump buffer to file, pausing all subscribers during the operation"""
        import pickle as pk
        import copy
        
        # Pause subscriptions to prevent new data from being added
        buffer_path = OUTPUT_DIR / filename if not Path(filename).is_absolute() else Path(filename)
        self.logger.info(f"Pausing subscriptions for buffer dump to {buffer_path}")
        self.subscriptions_paused = True
        
        # Also acquire buffer_lock for extra safety (in case any callback is still running)
        # with self.buffer_lock:
        self.logger.info(f"Starting buffer dump to {buffer_path}")
        my_buffer = copy.deepcopy(self.buffer)
        with open(buffer_path, 'wb') as f:
            pk.dump(self.buffer, f)
        
        # Resume subscriptions
        self.subscriptions_paused = False
        self.logger.info(f"Buffer successfully dumped to {buffer_path} (subscriptions resumed)")

def make_obs_node(mapping_config: Munch, max_buffer_size: int = 100) -> ObjectSensorMapper:
    """Factory: load config and return a ObjectSensorMapper node."""
    return ObjectSensorMapper(mapping_config)


def build_scene_pointcloud_from_entries(entries, voxel_size: float):
    """
    Offline helper: reuse object_detection_pipeline2.build_scene_pointcloud
    using ObsDataEntry objects loaded from entry_dumps.
    """
    if not entries:
        print("  ⚠ No entries provided to build scene point cloud from")
        return o3d.geometry.PointCloud()

    # Collect static_transforms from the first entry that has it
    static_transforms = None
    for e in entries:
        if hasattr(e, "static_transforms") and e.static_transforms is not None:
            static_transforms = e.static_transforms
            break

    if static_transforms is None:
        print("  ⚠ No static_transforms available on entries; cannot build scene point cloud")
        return o3d.geometry.PointCloud()

    class _DummyBuffer:
        """Minimal buffer wrapper exposing get_pointcloud_for_entry() like ObsDataBuffer."""
        def __init__(self, entries_list, static_transforms):
            # Map header_stamp -> ObsDataEntry
            self.entries = {e.header_stamp: e for e in entries_list}
            self.static_transforms = static_transforms

        def get_pointcloud_for_entry(self, header_stamp: str):
            # Reuse ObsDataEntry.get_pointcloud(static_transforms)
            return self.entries[header_stamp].get_pointcloud(self.static_transforms)

    # Build list of (timestamp, entry) tuples as expected by build_scene_pointcloud
    processed_entries = [
        (e.header_stamp, e) for e in entries if e.is_frame_full()
    ]

    if not processed_entries:
        print("  ⚠ No complete entries to build scene point cloud from")
        return o3d.geometry.PointCloud()

    dummy_buf = _DummyBuffer(entries, static_transforms)
    return build_scene_pointcloud(dummy_buf, processed_entries, voxel_size)


def main(args=None):
    """Minimal debug main function - processes entries from logs/current_run_outputs"""
    import argparse
    from munch import munchify
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Debug sensor object mapper')
    parser.add_argument('--entry-dir', default='logs/current_run_outputs nov 25 backup/entry_dumps',
                       help='Path to entry dumps directory')
    
    parsed_args = parser.parse_args(args)
    #parsed_args.entry_dir = "logs/current_run_outputs nov 25 backup/entry_dumps"
    
    # Override OUTPUT_DIR for offline processing to avoid overwriting real-time outputs
    global OUTPUT_DIR
    OUTPUT_DIR = LOGS_FOLDER / "current_run_outputs" / "offline_outputs"
    
    # Clear existing offline outputs before starting (like main.py does)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clear object_list.json if it exists (in case OUTPUT_DIR wasn't fully cleared)
    object_list_path = OUTPUT_DIR / "object_list.json"
    if object_list_path.exists():
        object_list_path.unlink()

    # Minimal config similar to main.py
    SESSION_ID = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    debug_config = munchify({
        "session_id": SESSION_ID,
        "max_buffer_size": 15,
        "stats_interval": 5.0,
        "save_rgb_frames": True,
        "viz_host": "127.0.0.1",
        "viz_port": "10016"
    })
    
    # Initialize ROS2 only if available
    use_debug_mode = not ROS2_AVAILABLE
    if ROS2_AVAILABLE:
        try:
            rclpy.init(args=args)
        except Exception as e:
            print(f"ROS2 initialization failed ({e}), using debug mode")
            use_debug_mode = True
    
    try:
        # Create mapper node (minimal initialization)
        mapper = ObjectSensorMapper(debug_config, debug_mode=use_debug_mode)
        
        # Load entries from entry_dumps directory
        entry_dir = Path(parsed_args.entry_dir)
        if not entry_dir.exists():
            print(f"Error: Entry directory not found: {entry_dir}")
            return
        
        # Get all pickle files sorted by name (which includes timestamp)
        entry_files = sorted(entry_dir.glob("entry_*.pkl"))
        print(f"Found {len(entry_files)} entry files to process")

        all_entries = [ pk.load(open(entry_file, 'rb')) for entry_file in entry_files ]
        
        # Process each entry
        for i, entry in enumerate(all_entries, 1):
            print(f"\n[{i}/{len(all_entries)}] Processing {entry.header_stamp}...")
            try:
                # Check if entry is full
                if not entry.is_frame_full():
                    print(f"  Warning: Entry {entry.header_stamp} is not full, skipping")
                    continue
                
                # Process entry for objects
                mapper._process_entry_for_objects(entry, dump_entry=False)
                print(f"  ✓ Processed entry {entry.header_stamp}")
            except Exception as e:
                print(f"  ✗ Error processing {entry.header_stamp}: {e}")
                traceback.print_exc()
        
        # # Save accumulated segmented point cloud from all entries
        # from tiamat_agent.vision_grounding.object_detection_pipeline import save_segmented_pointcloud
        # if mapper.all_image_data_by_id:
        #     save_segmented_pointcloud(mapper.all_image_data_by_id, output_path=OUTPUT_DIR / "objects_segmented_pc.ply")
        # Save accumulated segmented point cloud from all entries
        
        if mapper.all_image_data_by_id:
            # # Use relative path from project root
            # ply_output_path = OUTPUT_DIR / "objects_segmented_pc.ply"
            # print(f"\n[PLY SAVE] Saving to: {ply_output_path}")
            # ply_output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            # save_segmented_pointcloud(mapper.all_image_data_by_id, output_path=ply_output_path)

            # Option B: new per-object PLYs (ID-based, no future color matching needed)
            indiv_output_dir = OUTPUT_DIR / "individual_objects"
            print(f"\n[PLY SAVE] Saving individual object PCDs to: {indiv_output_dir}")
            save_individual_object_pointclouds(
                mapper.all_image_data_by_id,
                output_dir=indiv_output_dir
            )

        # ------------------------------------------------------------------
        # Build and save full RGB scene point cloud (like object_detection_pipeline2)
        # ------------------------------------------------------------------
        if all_entries:
            pcd_config = OBJECT_PIPELINE_CONFIGS["pointcloud_generation"]
            voxel_size = pcd_config["voxel_size"]

            print(f"\n[SCENE PLY SAVE] Building full RGB scene point cloud...")
            scene_pcd = build_scene_pointcloud_from_entries(all_entries, voxel_size)
            scene_output_path = OUTPUT_DIR / "all_points.ply"
            print(f"  Saving scene point cloud to: {scene_output_path}")
            save_pointcloud(scene_pcd, scene_output_path)
    
        # Save object list
        print(f"\n✓ Processing complete!")
        print(f"  Total objects detected: {len(mapper.object_library)}")
        print(f"  Processed frames: {mapper.processed_frames}")
        print(f"  Output saved to: {OUTPUT_DIR}")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        if ROS2_AVAILABLE and not use_debug_mode:
            try:
                rclpy.shutdown()
            except:
                pass


if __name__ == "__main__":
    main()