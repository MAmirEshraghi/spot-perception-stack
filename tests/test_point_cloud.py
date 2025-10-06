import numpy as np
import pytest

from perception_pipeline.point_cloud import calculate_point_cloud_coverage


# The @pytest.mark.parametrize decorator runs the test for each tuple in the list
@pytest.mark.parametrize("pcd1, pcd2, expected_coverage", [
    # Case 1: Small clouds, 50% overlap
    (np.array([[0,0,0], [1,0,0]]), np.array([[0,0,0], [2,0,0]]), 0.5),
    
    # Case 2: Large clouds, 25% overlap
    (np.array([[i,0,0] for i in range(100)]), np.array([[i,0,0] for i in range(25)]), 0.25),

    # Case 3: Mismatched sizes, small cloud is a 100% subset of the large one
    (np.array([[1,1,1], [2,2,2]]), np.array([[0,0,0], [1,1,1], [2,2,2], [3,3,3]]), 1.0),

    # Case 4: Edge case with an empty point cloud
    (np.array([[0,0,0], [1,0,0]]), np.array([]), 0.0),
])

def test_calculate_point_cloud_coverage_parameterized(pcd1, pcd2, expected_coverage):
    """Tests the coverage function against various point cloud sizes and overlaps."""
    coverage = calculate_point_cloud_coverage(pcd1, pcd2, voxel_size=0.1)
    assert coverage == pytest.approx(expected_coverage)

def test_calculate_point_cloud_coverage_partial_overlap():
    """
    Tests the coverage function with two simple point clouds that have a known 50% overlap.
    """
    # ARRANGE: Create two point clouds. pcd1 has 4 points. pcd2 shares 2 of them.
    pcd1 = np.array([[0,0,0], [1,0,0], [2,0,0], [3,0,0]])
    pcd2 = np.array([[0,0,0], [1,0,0], [8,8,8], [9,9,9]]) # Shares first two points
    voxel_size = 0.1

    # ACT: Run the function to be tested.
    coverage = calculate_point_cloud_coverage(pcd1, pcd2, voxel_size)

    # ASSERT: The coverage of pcd1 by pcd2 should be 2/4 = 0.5
    assert coverage == pytest.approx(0.5)

def test_calculate_point_cloud_coverage_no_overlap():
    """Tests the coverage function when there is no overlap."""
    pcd1 = np.array([[0,0,0], [1,1,1]])
    pcd2 = np.array([[5,5,5], [6,6,6]])
    voxel_size = 0.1

    coverage = calculate_point_cloud_coverage(pcd1, pcd2, voxel_size)
    assert coverage == pytest.approx(0.0)