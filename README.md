## Spot Robot Perception – Vision Grounding

This repository contains the **perception and vision-grounding stack** for an embodied AI agent running on a quadruped robot (Boston Dynamics Spot) in household-like environments (e.g. Habitat simulation).

The goal of this codebase is to turn raw **RGB‑D sensor data** into a **3D object library** and **world map** that can be used by planning, control, and language modules to perform tasks like:
- Discovering and mapping objects while exploring.
- Understanding **what** each object is (semantic knowledge).
- Knowing **where** each object is (spatial knowledge).
- Building **3D point clouds** of objects from multiple views (geometric knowledge).
- Enabling language-grounded behavior (e.g., “go to the red mug on the table”).

---

## Main components

- **Real-time ROS2 node**
  - `src/vision_grounding/z_sensor_object_map_node.py`  
    Maintains a **live point cloud map** and **ObjectLibrary** from streaming RGB‑D + odometry:
    - Subscribes to sensor topics.
    - Calls the perception pipeline to detect, segment, and localize objects.
    - Logs maps and object data for later analysis and visualization.

- **Offline perception pipeline**
  - `src/vision_grounding/object_detection_pipeline2.py`  
    Processes **recorded observation buffers** into:
    - `all_objects.json`: a rich object database with semantic + spatial metadata.
    - `all_points.ply`: a full scene point cloud.
    - Uses a Vision-Language Model, YOLO‑World, FastSAM, and depth to detect and ground objects.

---

## Repository structure (perception-focused)

- **`src/vision_grounding/`**
  - `z_sensor_object_map_node.py` – ROS2 sensor mapping + object grounding node.
  - `object_detection_pipeline2.py` – main object detection & 3D grounding pipeline.
  - `obs_data_buffer.py` – observation buffer, camera geometry, depth → point cloud.
  - `vlm_interface.py` – interface to the Vision-Language Model.
  - `fast_sam_helper2.py` – FastSAM segmentation utilities.
  - `pcd_coverage.py` – point cloud coverage analysis.
  - `deduplicate_objects_by_label*.py` – object/detection deduplication.
  - `visualization/` – scripts to visualize detections, objects, and point clouds.
  - `playground/` – experimental and debug scripts (e.g. alternative SAM helpers).

- **`src/utils/`**
  - `bbox_utils.py` – bounding box utilities (IoU, transforms, etc.).
  - `plotters.py` – plotting helpers (maps, poses, RGB‑D collages).
  - `ros_utils.py` – ROS-related helpers (e.g., quaternions, time).
  - `session_logger.py` – run/session logging to `logs/current_run_outputs/...`.
  - `func_utils.py` – small generic helpers (timing, etc.).

- **`docs/`**
  - `overview.md` – high-level description of the perception stack.
  - `quickstart.md` – how to run the offline pipeline and ROS2 node.
  - `nodes_and_pipelines.md` – details of the main nodes and pipelines.
  - `development.md` – guidance for extending and organizing the code.

---

## Getting started

For a quick, practical introduction:

- See **`docs/quickstart.md`** for:
  - Setting up the environment (Python, GPU, key dependencies).
  - Running the **offline object detection pipeline**.
  - Running the **real-time ROS2 sensor mapping node**.
  - Basic visualization of outputs.

For more context on how everything fits together, read:

- **`docs/overview.md`** – what the perception stack does and how it fits into the broader embodied agent.
- **`docs/nodes_and_pipelines.md`** – details of `z_sensor_object_map_node.py` and `object_detection_pipeline2.py`.

---

## Status and scope

This repo focuses on the **perception / grounding layer** of the overall Spot embodied agent.  
Other parts of the full system (e.g., high-level planning, navigation, language interfaces) are designed to sit on top of the **object library** and **world map** produced here, and may live in separate repositories or be added in future work.## Spot Robot Perception – Vision Grounding

This repository contains the **perception and vision-grounding stack** for an embodied AI agent running on a quadruped robot (Boston Dynamics Spot) in household-like environments (e.g. Habitat simulation).

The goal of this codebase is to turn raw **RGB‑D sensor data** into a **3D object library** and **world map** that can be used by planning, control, and language modules to perform tasks like:
- Discovering and mapping objects while exploring.
- Understanding **what** each object is (semantic knowledge).
- Knowing **where** each object is (spatial knowledge).
- Building **3D point clouds** of objects from multiple views (geometric knowledge).
- Enabling language-grounded behavior (e.g., “go to the red mug on the table”).

---

## Main components

- **Real-time ROS2 node**
  - `src/vision_grounding/z_sensor_object_map_node.py`  
    Maintains a **live point cloud map** and **ObjectLibrary** from streaming RGB‑D + odometry:
    - Subscribes to sensor topics.
    - Calls the perception pipeline to detect, segment, and localize objects.
    - Logs maps and object data for later analysis and visualization.

- **Offline perception pipeline**
  - `src/vision_grounding/object_detection_pipeline2.py`  
    Processes **recorded observation buffers** into:
    - `all_objects.json`: a rich object database with semantic + spatial metadata.
    - `all_points.ply`: a full scene point cloud.
    - Uses a Vision-Language Model, YOLO‑World, FastSAM, and depth to detect and ground objects.

---

## Repository structure (perception-focused)

- **`src/vision_grounding/`**
  - `z_sensor_object_map_node.py` – ROS2 sensor mapping + object grounding node.
  - `object_detection_pipeline2.py` – main object detection & 3D grounding pipeline.
  - `obs_data_buffer.py` – observation buffer, camera geometry, depth → point cloud.
  - `vlm_interface.py` – interface to the Vision-Language Model.
  - `fast_sam_helper2.py` – FastSAM segmentation utilities.
  - `pcd_coverage.py` – point cloud coverage analysis.
  - `deduplicate_objects_by_label*.py` – object/detection deduplication.
  - `visualization/` – scripts to visualize detections, objects, and point clouds.
  - `playground/` – experimental and debug scripts (e.g. alternative SAM helpers).

- **`src/utils/`**
  - `bbox_utils.py` – bounding box utilities (IoU, transforms, etc.).
  - `plotters.py` – plotting helpers (maps, poses, RGB‑D collages).
  - `ros_utils.py` – ROS-related helpers (e.g., quaternions, time).
  - `session_logger.py` – run/session logging to `logs/current_run_outputs/...`.
  - `func_utils.py` – small generic helpers (timing, etc.).

- **`docs/`**
  - `overview.md` – high-level description of the perception stack.
  - `quickstart.md` – how to run the offline pipeline and ROS2 node.
  - `nodes_and_pipelines.md` – details of the main nodes and pipelines.
  - `development.md` – guidance for extending and organizing the code.

---

## Getting started

For a quick, practical introduction:

- See **`docs/quickstart.md`** for:
  - Setting up the environment (Python, GPU, key dependencies).
  - Running the **offline object detection pipeline**.
  - Running the **real-time ROS2 sensor mapping node**.
  - Basic visualization of outputs.

For more context on how everything fits together, read:

- **`docs/overview.md`** – what the perception stack does and how it fits into the broader embodied agent.
- **`docs/nodes_and_pipelines.md`** – details of `z_sensor_object_map_node.py` and `object_detection_pipeline2.py`.

---

## Status and scope

This repo focuses on the **perception / grounding layer** of the overall Spot embodied agent.  
Other parts of the full system (e.g., high-level planning, navigation, language interfaces) are designed to sit on top of the **object library** and **world map** produced here, and may live in separate repositories or be added in future work.