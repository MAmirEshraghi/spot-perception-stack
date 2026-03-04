## Spot Robot Perception – Vision Grounding

Modern embodied AI aims to build an **“AI brain for robots”**: systems that can not only move, but also **understand and reason about the world** at a level closer to how people think. A key part of that brain is **perception** – answering questions like:

- **How does the robot perceive and understand its environment?**
- **What kinds of human-like knowledge does the robot need about objects and spaces?**

Robots operating in complex human environments must deal with:

- **Multiple camera views and sensor fusion** (RGB‑D, odometry, transforms).
- **Semantic vs. geometric distinctions** between objects, even with identical labels.
- **Real-time constraints** inside ROS-based systems.

This repository focuses on the **perception and vision-grounding stack** for an embodied AI agent running on a quadruped robot (Boston Dynamics Spot) in household-like environments (e.g. Habitat simulation). It builds up the robot’s **semantic, spatial, and geometric knowledge** of the world:

- From raw **RGB‑D sensor data** to a **3D object library** and **world map**.
- So planning, control, and language modules can:
  - Discover and map objects while exploring.
  - Understand **what** each object is (semantic knowledge).
  - Know **where** each object is (spatial knowledge).
  - Build **3D point clouds** of objects from multiple views (geometric knowledge).
  - Enable language-grounded behavior (e.g., “go to the red mug on the table”).

![Spot perception stack overview](/docs/images/overview.png)

---

## Main components

- **Real-time ROS2 node**
  - `src/vision_grounding/z_sensor_object_map_node.py`  
  Maintains a **live point cloud map** and **ObjectLibrary** from streaming RGB‑D + odometry:
    - Subscribes to sensor topics.
    - Calls the perception pipeline to detect, segment, and localize objects.
    - Logs maps and object data for later analysis and visualization.
- **Offline perception pipeline**
  - `src/vision_grounding/object_detection_pipeline.py`  
  Processes **recorded observation buffers** into:
    - `all_objects.json`: a rich object database with semantic + spatial metadata.
    - `all_points.ply`: a full scene point cloud.
    - Uses a Vision-Language Model, YOLO‑World, FastSAM, and depth to detect and ground objects.

---

## Repository structure (perception-focused)

- `**src/vision_grounding/`**
  - `z_sensor_object_map_node.py` – ROS2 sensor mapping + object grounding node.
  - `object_detection_pipeline2.py` – main object detection & 3D grounding pipeline.
  - `obs_data_buffer.py` – observation buffer, camera geometry, depth → point cloud.
  - `vlm_interface.py` – interface to the Vision-Language Model.
  - `fast_sam_helper2.py` – FastSAM segmentation utilities.
  - `pcd_coverage.py` – point cloud coverage analysis.
  - `deduplicate_objects_by_label*.py` – object/detection deduplication.
  - `visualization/` – scripts to visualize detections, objects, and point clouds.
  - `playground/` – experimental and debug scripts (e.g. alternative SAM helpers).
- `**src/utils/`**
  - `bbox_utils.py` – bounding box utilities (IoU, transforms, etc.).
  - `plotters.py` – plotting helpers (maps, poses, RGB‑D collages).
  - `ros_utils.py` – ROS-related helpers (e.g., quaternions, time).
  - `session_logger.py` – run/session logging to `logs/current_run_outputs/...`.
  - `func_utils.py` – small generic helpers (timing, etc.).
- `**docs/**`
  - `overview.md` – high-level description of the perception stack.
  - `quickstart.md` – how to run the offline pipeline and ROS2 node.
  - `nodes_and_pipelines.md` – details of the main nodes and pipelines.
  - `development.md` – guidance for extending and organizing the code.

---

## Getting started

For a quick, practical introduction:

- See `**docs/quickstart.md**` for:
  - Setting up the environment (Python, GPU, key dependencies).
  - Running the **offline object detection pipeline**.
  - Running the **real-time ROS2 sensor mapping node**.
  - Basic visualization of outputs.

For more context on how everything fits together, read:

- `**docs/overview.md`** – what the perception stack does and how it fits into the broader embodied agent.
- `**docs/nodes_and_pipelines.md`** – details of `z_sensor_object_map_node.py` and `object_detection_pipeline2.py`.

---

## Status and scope

This repo focuses on the **perception / grounding layer** of the overall Spot embodied agent.  
Other parts of the full system (e.g., high-level planning, navigation, language interfaces) are designed to sit on top of the **object library** and **world map** produced here, and may live in separate repositories or be added in future work.