# Perception repository for DRAIL_tiamat

# (1) Offline Perception Pipeline

containing a Object Grounding pipeline for processing pre-recorded robotic sensor data. It uses the Segment Anything Model (SAM) to identify objects in RGB images, projects them into 3D space using depth data, and employs a de-duplication algorithm to track unique objects across multiple frames. A Vision-Language Model (VLM) is used to generate descriptions for each unique object discovered. The final output is a structured log directory containing all metadata, visualizations, and detailed performance reports.

# (2) Repository Structure

```
perception_tiamat_DRAIL/
├── data/                  # Raw input data (e.g., .pkl files)
├── logs/                  # Default output directory for pipeline runs
├── models/                # Storage for downloaded AI model checkpoints (e.g., SAM)
├── scripts/               # Helper scripts (e.g., creating sample data)
│
├── src/
│   ├── obs_data_buffer.py   # Data structures for handling sensor streams
│   └── perception_pipeline/ # The core application logic as a Python package
│       ├── ai_models.py
│       ├── log_manager.py
│       ├── point_cloud.py
│       ├── reporting.py
│       ├── run_pipeline.py  # Main entry point for the application
│       └── visualization.py
│
└── tests/                   # All tests for the project
    ├── data/                # Sample data for running tests
    ├── test_ai_models.py
    ├── test_benchmarks.py
    ├── test_e2e_pipeline.py
    ├── test_integration_pipeline.py
    └── ... (other unit test files)
```
# (3) How to Use

### a. Clone and install dependencies:

```
pip install -e .
```
### b. Sample command to run

```
python -m src.perception_pipeline.run_pipeline /
    -p "tests/data/sample_obs_buffer.pkl" /
    -s "models/sam_vit_l_0b3195.pth" /
    -v "HuggingFaceTB/SmolVLM-256M-Instruct" /
    -d "cuda" /
    -l 40 /
    --centroid_threshold 300 /
    --coverage_threshold 0.60 /
    --voxel_size 1.5 /
    --min_mask_area 100 /
    --kdtree_radius 0.1 /

```

### c. Output:

a new timestamped directory will be created in logs/perception_logs_offline/. This directory contains the complete output, including:

- log.json: A structured JSON file with metadata for all images and unique objects.

- performance_summary.txt: A human-readable report of the processing time for each stage.

- frame_visualizations/: Images showing the detected objects in each processed frame.

- timing_analysis/: Detailed raw timing data and plots.


# (4) Testing Strategy

This project implements three-level testing strategy based on the standard "Testing Pyramid":


## Test Levels:

1. End-to-End (tests/test_e2e_pipeline.py): Runs the main script as a subprocess and asserts that all expected output files (logs, visualizations) are created correctly.

2. Integration Tests (tests/test_integration_pipeline.py): Tests the data handoff between modules, like ensuring the mask format from SAM works as a valid input for the point cloud extraction logic.

3. Unit Tests (tests/test_*.py): Includes tests for mathematical correctness (test_point_cloud.py), data structure management (test_obs_data_buffer.py) and etc.

4. Performance Benchmarking (tests/test_benchmarks.py): This provides statistical analysis of the speed of critical functions.

## How to Run

```
export SAM_CHECKPOINT_PATH="models/sam_vit_l_0b3195.pth"

# run all tests:
pytest -v  

# run all tests with a code coverage report
pytest -v --cov=src  

# run ONLY fast unit tests
pytest -v -m "not integration and not e2e" 

# run only integration
pytest -v -m "integration"  

# run only e2e
pytest -v -m "e2e"   

# run only performance benchmarks
pytest --benchmark-only 


```