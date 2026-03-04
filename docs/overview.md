## Spot Robot Embodied AI – Overview

This repository is part of an **embodied AI agent for a quadruped robot (Boston Dynamics Spot)** operating in household-like environments (simulation via Habitat, and later real world).

The system has several high-level components:

- **Perception (this repo’s focus)**
  - Understands the scene from RGB‑D sensor data.
  - Builds a **3D object library** with:
    - **Semantic knowledge** (what each object is, via VLM + detection).
    - **Spatial knowledge** (where it is in the world).
    - **Geometric knowledge** (3D point cloud of each object from multiple views).
  - Supports both **offline batch processing** and **online ROS2-based real-time mapping**.

- **Planning & Control** (designed, code may live elsewhere or be added later)
  - Plans **frontiers / exploration targets**.
  - Navigates the robot to collect views and interact with objects.
  - Uses the object library and world map to ground language commands.

## Main perception entrypoints

- **`src/vision_grounding/object_detection_pipeline2.py`**
  - Offline **object detection & 3D localization pipeline**.
  - Takes buffered RGB‑D observations, runs:
    - Vision-Language Model (VLM) for semantics,
    - YOLO‑World for 2D boxes,
    - FastSAM for segmentation,
    - Depth for 3D world coordinates.
  - Outputs:
    - A rich **object database** (`all_objects.json`).
    - A **scene point cloud** (`all_points.ply`).

- **`src/vision_grounding/z_sensor_object_map_node.py`**
  - **ROS2 node** for **real-time sensor mapping + object grounding**.
  - Subscribes to RGB‑D and odometry topics, maintains:
    - A **live point cloud map**.
    - A **growing object library** with semantic + spatial + geometric info.
  - Designed for interactive use: the planner can query the library and map.

## Supporting modules (perception)

- **`src/vision_grounding/obs_data_buffer.py`**: Efficient data buffer + camera/transform helpers.
- **`src/vision_grounding/vlm_interface.py`**: Interface to the chosen VLM (e.g. InternVL / Qwen VL).
- **`src/vision_grounding/fast_sam_helper2.py`**: FastSAM-based segmentation helpers.
- **`src/vision_grounding/deduplicate_objects_by_label*.py`**: Merge or filter duplicate detections.
- **`src/vision_grounding/visualization/`**: Scripts to visualize detections, point clouds, and deduplicated objects.
- **`src/utils/`**: Shared utilities (bounding boxes, plotting, ROS utils, logging, small helpers).

Together, these components provide the **perception and grounding layer** that the higher-level planner and language interface can use to instruct the robot (e.g., “go to the red mug on the table”).