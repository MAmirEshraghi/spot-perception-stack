import pytest
import numpy as np
import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from perception_pipeline.point_cloud import calculate_point_cloud_coverage
#from perception_pipeline.ai_models import get_vlm_description
from perception_pipeline.point_cloud import get_organized_point_cloud, get_o3d_cam_intrinsic

@pytest.fixture
def benchmark_pcds():
    """Provides two simple point clouds for coverage benchmarks."""
    pcd1 = np.random.rand(5000, 3)
    pcd2 = np.random.rand(5000, 3)
    return pcd1, pcd2

@pytest.fixture
def sam_mask_generator():
    """Loads the real SAM model, skipping if not available."""
    sam_checkpoint_path = os.environ.get("SAM_CHECKPOINT_PATH")
    if not sam_checkpoint_path or not os.path.exists(sam_checkpoint_path):
        pytest.skip("SAM_CHECKPOINT_PATH env var not set. Skipping SAM benchmark.")
    
    sam = sam_model_registry["vit_l"](checkpoint=sam_checkpoint_path)
    return SamAutomaticMaskGenerator(model=sam, min_mask_region_area=100)

# Benchmark Tests:


def test_benchmark_get_organized_point_cloud(benchmark):
    """
    Benchmarks the performance of the depth image to point cloud projection.
    `benchmark` is a special fixture from pytest-benchmark.
    """
    # ARRANGE: Set up realistic-sized inputs
    height, width = 480, 640
    depth_image = np.random.rand(height, width).astype(np.float32)
    camera_pose = np.random.rand(7)
    intrinsics = get_o3d_cam_intrinsic(height, width)

    # ACT & ASSERT (handled by benchmark):
    # Pass the function and its arguments to the benchmark fixture.
    # It will run the function many times to get a reliable performance measure.
    benchmark(get_organized_point_cloud, depth_image, camera_pose, intrinsics)


def test_benchmark_coverage_check(benchmark, benchmark_pcds):
    """Benchmarks the point cloud coverage function."""
    pcd1, pcd2 = benchmark_pcds

    benchmark(calculate_point_cloud_coverage, pcd1, pcd2, voxel_size=0.1)

def test_benchmark_centroid_check(benchmark):
    """Benchmarks the centroid distance check (np.linalg.norm)."""
    centroid1 = np.random.rand(3)
    centroid2 = np.random.rand(3)

    benchmark(np.linalg.norm, centroid1 - centroid2)

def test_benchmark_sam_per_image(benchmark, sam_mask_generator):
    """Benchmarks a full SAM run on a sample image."""
    sample_image = np.zeros((480, 640, 3), dtype=np.uint8)

    benchmark(sam_mask_generator.generate, sample_image)