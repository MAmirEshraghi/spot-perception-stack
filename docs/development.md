## Development Guide

This file describes how to work with and extend the **perception stack** in this repository.

---

## 1. Repo structure (perception-relevant)

- **`src/vision_grounding/`**
  - Core perception and grounding logic (nodes + pipelines).
  - Key files:
    - `z_sensor_object_map_node.py` – real-time ROS2 mapping + object library node.
    - `object_detection_pipeline2.py` – offline object detection & 3D grounding pipeline.
    - `obs_data_buffer.py` – observation buffer and geometry helpers.
    - `vlm_interface.py` – VLM integration.
    - `fast_sam_helper2.py` – FastSAM segment helpers.
    - `deduplicate_objects_by_label*.py` – deduplication logic.
  - Subfolders:
    - `visualization/`: scripts like `viz_detections.py`, `viz_segment_pc.py`, etc.
    - `playground/`: experimental / debug scripts (e.g., `sam_3_*`, `object_candidate_selection.py`).

- **`src/utils/`**
  - Shared utilities:
    - `bbox_utils.py` – bounding box calculations and helpers.
    - `plotters.py` – generic plotting utilities (occupancy grids, height maps, collages).
    - `ros_utils.py` – ROS-related helpers (e.g., quaternion to yaw).
    - `session_logger.py` – logging with run/session IDs.
    - `func_utils.py` – small functional/time helpers.

---

## 2. Where to add new code

- **New core perception features**
  - If the feature is part of the **main pipeline or node behavior**, add or extend:
    - `object_detection_pipeline2.py` (offline/processing side), or
    - `z_sensor_object_map_node.py` (online ROS side).
  - Keep functions **small and focused**; consider moving reusable chunks into helper modules.

- **New reusable helpers**
  - If it’s primarily about:
    - **Vision/grounding internals** → add a new module under `src/vision_grounding/` (e.g., `something_helper.py`).
    - **General-purpose utilities** → add under `src/utils/` (e.g., new bbox or plotting helpers).
  - Try to reuse `bbox_utils.py`, `obs_data_buffer.py`, and `func_utils.py` instead of duplicating logic.

- **Experiments / debug scripts**
  - Put prototypes in `src/vision_grounding/playground/`.
  - Once they mature and are used by other modules, move them into the main `vision_grounding` or `utils` namespace.

- **Visualization**
  - New diagnostic / debug visualizations should go under `src/vision_grounding/visualization/`.
  - If they become very generic, refactor shared pieces into `src/utils/plotters.py`.

---

## 3. Coding style / conventions (lightweight)

- **Imports**
  - Prefer local package imports like:
    - `from src.vision_grounding.obs_data_buffer import ObsDataBuffer`
    - `from src.utils.bbox_utils import ...`
  - Keep external, non-local modules (if any) clearly separated at the top.

- **Naming**
  - `*_helper.py` for self-contained helper modules that wrap a specific model or operation.
  - `viz_*` for visualization scripts.
  - `*_pipeline.py` or `*_node.py` for main flows / ROS nodes.

- **Logging and outputs**
  - Use `SessionLogger` (`src/utils/session_logger.py`) for runs that write to `logs/current_run_outputs/...`.
  - Avoid hard-coding absolute paths; prefer relative paths from project root or log roots.

---

## 4. Working modes

- **Online (ROS2)**
  - Extend `z_sensor_object_map_node.py` when you want real-time behavior:
    - New subscriptions / publishers.
    - New ways to log or manage objects and maps.

- **Offline / batch**
  - Use and extend `object_detection_pipeline2.py` for:
    - Algorithm experiments.
    - Running on saved buffers to debug perception and grounding.

By keeping **entrypoints** (nodes/pipelines) small and well documented, and pushing shared logic into helpers under `vision_grounding` and `utils`, the codebase stays easier to navigate and extend.