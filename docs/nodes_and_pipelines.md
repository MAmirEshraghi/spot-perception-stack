## Nodes and Pipelines

This document summarizes the **main nodes and pipelines** in the perception stack.

---

## 1. `z_sensor_object_map_node.py` – Real-time ROS2 node

**Path**: `src/vision_grounding/z_sensor_object_map_node.py`

**Purpose**: Maintain a **live world model** of the environment while the robot moves.

- **Inputs (conceptual)**:
  - RGB and depth images from the robot’s camera(s).
  - Odometry / TF for robot pose (position + yaw).
- **Core responsibilities**:
  - Keep an **incrementally growing point cloud map** of the environment.
  - Maintain an **ObjectLibrary**:
    - Stores object entries with:
      - Semantic label.
      - 3D position and “valid depth” status.
      - Robot pose at time of observation.
  - Periodically logs maps, objects, and diagnostics under `logs/current_run_outputs/...`.
- **Integration**:
  - Uses `ObsDataBuffer` to collect and pre-process sensor data.
  - Calls into the **object detection & grounding pipeline** (see below) to populate and update the object library.
  - Designed for **interactive use** with a planner that chooses **frontiers / viewpoints** and uses the object library for language-grounded behavior.

---

## 2. `object_detection_pipeline2.py` – Perception & grounding pipeline

**Path**: `src/vision_grounding/object_detection_pipeline2.py`

**Purpose**: Given buffered observations, produce a **rich object database** and **3D point clouds**.

**High-level steps** (conceptual):

1. **Input**:
   - Reads an `ObsDataBuffer` or buffer-like data structure (RGB, depth, intrinsics, transforms).
2. **Semantic understanding**:
   - Uses a **Vision-Language Model (VLM)** via `vlm_interface.py` to get semantic context or candidate labels.
   - Uses **YOLO‑World** for 2D detections (bounding boxes with labels).
3. **Segmentation**:
   - Uses **FastSAM** (via `fast_sam_helper2.py`) to segment pixels within/around detections.
4. **3D localization**:
   - Combines segmentation masks with depth and camera intrinsics/extrinsics (from `obs_data_buffer.py`) to:
     - Project pixels into 3D world coordinates.
     - Build **object point clouds** and a **full scene point cloud**.
5. **Post-processing and deduplication**:
   - Uses `deduplicate_objects_by_label*.py` and bounding box utilities to merge duplicate detections.
6. **Outputs**:
   - **`all_objects.json`**: object entries with:
     - Semantic metadata (labels, confidence).
     - Spatial metadata (positions, coverage).
     - References to 3D geometry.
   - **`all_points.ply`**: full scene point cloud for visualization and mapping.

This pipeline can be run **offline** on recorded buffers, or its logic can be called **online** from `z_sensor_object_map_node.py`.

---

## 3. Supporting perception modules

- **`obs_data_buffer.py`**
  - Data structure and utilities for:
    - Storing RGB‑D frames, intrinsics, extrinsics.
    - Converting depth to point clouds.
    - Handling transforms and sensor geometry.

- **`vlm_interface.py`**
  - Wraps the chosen VLM (e.g., InternVL / Qwen VL).
  - Provides a consistent API for:
    - Asking questions about images.
    - Generating or ranking textual labels / descriptions.

- **`fast_sam_helper2.py`**
  - Helpers for FastSAM segmentations based on bounding boxes.
  - Used by the main pipeline to extract pixel-level object regions.

- **`deduplicate_objects_by_label.py` and `deduplicate_objects_by_label_individual.py`**
  - Implement strategies for:
    - Merging objects with overlapping bounding boxes and identical labels.
    - Cleaning the object database to avoid redundancy.

- **`pcd_coverage.py`**
  - Utilities to analyze how well the 3D point cloud **covers** the scene / objects.
  - Useful for debugging and planning view coverage.

Together, these modules form the **perception + grounding layer** used by the planning and control parts of the agent.