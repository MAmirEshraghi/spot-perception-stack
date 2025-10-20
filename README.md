# Perception Tiamat Drail - Segment Deduplication

This repository contains a perception pipeline focused on the efficient management and deduplication of 3D object point clouds. The core functionality revolves around identifying and merging similar objects detected across various frames and camera views, aiming to build a library of unique 3D objects. It leverages advanced techniques for point cloud processing and utilizes the Segment Anything Model (SAM) for object mask generation.

```
perception_tiamat_drail-develop-seg_dedup/
├── .coverage
├── .gitignore
├── README.md
├── pyproject.toml
├── scripts/
│   ├── create_sample_data.py
│   ├── generate_object_pointclouds.py
│   ├── interactive_dedup_viewer.py
│   ├── object_library.py
│   └── visualize_library.py
├── src_perception/
│   ├── components/
│   │   ├── batch_sam.py
│   │   ├── pcd_coverage.py
│   │   └── point_cloud.py
│   └── obs_data_buffer.py
└── tests/
    ├── test_mask_pc_coverages_check.py
    ├── test_mask_pointcloud.py
    ├── test_obs_data_buffer.py
    └── test_point_cloud.py
```
## Project Overview

This project implements a pipeline to process raw observation data (e.g., RGB-D images from multiple cameras), extract 3D object point clouds using the Segment Anything Model (SAM), and then deduplicate these objects into a concise library of unique 3D representations. The ObjectLibrary class, defined in object_library.py, is central to this process. It efficiently stores and manages unique objects, performing batched many-vs-many coverage calculations to identify and merge redundant detections. The goal is to reduce a large set of observed object masks into a smaller, representative collection of distinct objects.

### Key features include:

- Object Segmentation: Utilizes SAM to generate masks from RGB images, which are then used to create 3D point clouds from corresponding depth data.
- Efficient Deduplication: Employs an optimized ObjectLibrary to add new object candidates and perform self-deduplication, ensuring that only truly unique objects are retained.
- Point Cloud Management: Stores and manipulates 3D point cloud data using PyTorch tensors for GPU acceleration.
- Visualization Tools: Provides scripts to visualize the generated object library and raw data.


## Setup and Installation

```
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Adjust for your CUDA version or remove if CPU only
pip install numpy opencv-python open3d segment_anything
```

```
mkdir models
wget -O models/sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

## Usage and Commands

#### 1. Generate Sample Data

If you don't have an obs_buffer.pkl file, you can create a sample one using create_sample_data.py. This script loads a full data buffer and creates a smaller, sampled version.

Bash


python scripts/create_sample_data.py


This will generate tests/data/sample_obs_buffer.pkl (or similar, depending on configuration within the script). You might need to adjust FULL_DATA_PATH in create_sample_data.py if your source data is elsewhere.

#### 2. Generate Object Point Clouds

This script processes the observation data, applies SAM to segment objects, and generates 3D point clouds for each detected object. The results are saved to `data/object_pcds.pkl.`


```python scripts/generate_object_pointclouds.py
```

Note: Ensure `SAM_CHECKPOINT_PATH` and `BUFFER_PATH` are correctly configured in `generate_object_pointclouds.py` before running.

#### 3. Deduplicate Objects

The `object_library.py` script is the main entry point for the deduplication process. It loads the generated object point clouds, adds them to the `ObjectLibrary`, performs deduplication, and saves the unique objects.


```python scripts/object_library.py
```

This script will output `logs/object_library/unique_objects.pkl` containing the deduplicated set of unique 3D objects.

#### 4. Visualize the Object Library

Use `visualize_library.py` to interactively view the unique objects in the library. It uses Open3D for 3D visualization and allows cycling through objects and toggling different background views.


```
python scripts/visualize_library.py
```

Controls in the viewer:

`[->] / [<-]` : Cycle through unique objects.

`[A] `: Toggle GRAY UNIQUE objects background.

`[B] `: Toggle COLORFUL UNIQUE objects background.

`[C] `: Toggle ALL RAW objects background (press again to revert).

