import pytest
import numpy as np
import open3d as o3d
from obs_data_buffer import (
    ObsDataEntry,
    ObsDataBuffer,
    compose_transforms,
    compose_transforms_optimized,
    filter_ceiling_points,
    get_camera_intrinsics
)

# ==============================================================================
# Fixtures for Reusable Test Data
# ==============================================================================

@pytest.fixture
def sample_transform_a():
    """A sample transform (e.g., world to base) with no rotation."""
    return {
        "position": {"x": 1, "y": 2, "z": 3},
        "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}
    }

@pytest.fixture
def sample_transform_b():
    """A sample transform (e.g., base to camera) with a 90-degree Z rotation."""
    val = np.sqrt(2) / 2
    return {
        "position": {"x": 0.1, "y": 0.2, "z": 0.3},
        "orientation": {"x": 0, "y": 0, "z": val, "w": val}
    }

@pytest.fixture
def sample_static_transforms():
    """A mock static_transforms dictionary for testing."""
    return {
        "world_to_map": {"position": {"x": 0, "y": 0, "z": 0}, "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}},
        "base_link_to_cameras": {
            "head_left_rgbd": {"position": {"x": 0.1, "y": 0.2, "z": 0.3}, "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}},
            "head_right_rgbd": {"position": {}, "orientation": {}},
            "left_rgbd": {"position": {}, "orientation": {}},
            "right_rgbd": {"position": {}, "orientation": {}},
            "rear_rgbd": {"position": {}, "orientation": {}},
        },
        "camera_to_optical": {"position": {"x": 0, "y": 0, "z": 0}, "orientation": {"x": -0.5, "y": 0.5, "z": -0.5, "w": 0.5}}
    }

@pytest.fixture
def full_obs_data_entry():
    """Provides a fully populated ObsDataEntry instance."""
    entry = ObsDataEntry("stamp_full")
    dummy_img = np.zeros((10, 10))
    for name in entry.expected_rgb:
        entry.add_rgb(name, dummy_img)
    for name in entry.expected_depth:
        entry.add_depth(name, dummy_img)
    entry.add_odometry({"position": {}, "orientation": {}})
    return entry

# ==============================================================================
# Tests for Helper Functions
# ==============================================================================

def test_compose_transforms(sample_transform_a, sample_transform_b):
    """Tests the basic transform composition math."""
    composed = compose_transforms(sample_transform_a, sample_transform_b)

    # Assert position: t_composed = R1 @ t2 + t1
    assert composed["position"]["x"] == pytest.approx(1.1)
    assert composed["position"]["y"] == pytest.approx(2.2)
    assert composed["position"]["z"] == pytest.approx(3.3)
    
    # Assert orientation: R_composed = R1 @ R2
    assert composed["orientation"]["z"] == pytest.approx(np.sqrt(2) / 2)

def test_compose_transforms_optimized_raises_error(sample_static_transforms):
    """Tests that compose_transforms_optimized raises a ValueError for an unknown camera."""
    with pytest.raises(ValueError):
        compose_transforms_optimized({}, "unknown_camera", sample_static_transforms)

def test_get_camera_intrinsics():
    """Tests that intrinsics are correctly generated from image dimensions."""
    depth_image = np.zeros((480, 640))
    intrinsics = get_camera_intrinsics(depth_image)
    
    assert intrinsics.width == 640
    assert intrinsics.height == 480
    assert intrinsics.get_focal_length() == (320.0, 320.0)
    assert intrinsics.get_principal_point() == (320.0, 240.0)

def test_filter_ceiling_points():
    """Tests that points above a certain height and too close are removed."""
    points = np.array([
        [0, 0, 1.0],  # This point will be filtered (distance == 0.5)
        [0, 0, 3.0],  # This point will be filtered (too high)
        [0, 0, 0.1],  # This point will be filtered (too close)
        [5, 5, 1.5],  # This is the only point that should remain
    ])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.ones_like(points)) # Must have colors

    robot_pos = [0, 0, 0.5]
    filtered_pcd = filter_ceiling_points(pcd, robot_pos, clip_height=2.0, min_distance=0.5)

    # ASSERT: Check that exactly one point remains
    assert len(filtered_pcd.points) == 1

    # ASSERT: Check that the remaining point is the correct one.
    remaining_points = np.asarray(filtered_pcd.points)
    expected_point = np.array([5, 5, 1.5])
    assert np.allclose(remaining_points[0], expected_point)
    
# ==============================================================================
# Tests for ObsDataEntry Class
# ==============================================================================

def test_obs_data_entry_is_frame_full(full_obs_data_entry):
    """Tests the logic of the is_frame_full method."""
    entry = ObsDataEntry("stamp_1")
    assert not entry.is_frame_full()
    assert full_obs_data_entry.is_frame_full()

def test_get_rgb_depth_pairs(full_obs_data_entry):
    """Tests that RGB and Depth images are paired correctly."""
    pairs = full_obs_data_entry.get_rgb_depth_pairs()
    assert len(pairs) == 5
    
    # Check one pair for correctness
    rgb_names = [p[0] for p in pairs]
    assert "left_rgb" in rgb_names

# ==============================================================================
# Tests for ObsDataBuffer Class
# ==============================================================================

def test_obs_data_buffer_add_data_creates_entry():
    """Tests that adding data to a buffer creates a new entry."""
    buffer = ObsDataBuffer()
    assert len(buffer.entries) == 0

    buffer.add_rgb("stamp_1", "left_rgb", np.zeros((10, 10)))
    
    assert len(buffer.entries) == 1
    assert "stamp_1" in buffer.entries
    assert "left_rgb" in buffer.entries["stamp_1"].rgb_images

def test_is_tf_static_ready(sample_static_transforms):
    """Tests the logic for checking if all static transforms are present."""
    buffer = ObsDataBuffer()
    assert not buffer.is_tf_static_ready()

    # Add transforms one by one
    buffer.add_tf_static("world", "map", {}, {})
    assert not buffer.is_tf_static_ready()

    for cam_name, transform in sample_static_transforms["base_link_to_cameras"].items():
        buffer.add_tf_static("base_link", cam_name, {}, {})
    assert not buffer.is_tf_static_ready()
    
    buffer.add_tf_static("any_cam", "any_cam_optical", {}, {})
    assert buffer.is_tf_static_ready()

def test_get_next_entry_to_process(full_obs_data_entry):
    """Tests that the correct entry is returned for processing."""
    buffer = ObsDataBuffer()

    # Add an incomplete entry
    buffer.add_rgb("stamp_incomplete", "left_rgb", np.zeros((10,10)))
    assert buffer.get_next_entry_to_process() is None

    # Add a full entry (using the fixture)
    buffer.entries["stamp_full"] = full_obs_data_entry
    
    entry = buffer.get_next_entry_to_process()
    assert entry is not None
    assert entry.header_stamp == "stamp_full"

    # Mark it as processed and check again
    entry.set_processed()
    assert buffer.get_next_entry_to_process() is None

def test_maintain_buffer_size():
    """Tests that the buffer prunes the oldest processed entry when full."""
    buffer = ObsDataBuffer(max_size=2)
    buffer.add_rgb("stamp_1", "left_rgb", np.zeros((10,10)))
    buffer.add_rgb("stamp_2", "left_rgb", np.zeros((10,10)))
    buffer.add_rgb("stamp_3", "left_rgb", np.zeros((10,10)))
    
    # Mark entries as processed
    buffer.entries["stamp_1"].processed = True
    buffer.entries["stamp_2"].processed = True
    buffer.entries["stamp_3"].processed = True

    # The buffer should be at size 3, calling add again should prune it
    buffer.add_rgb("stamp_4", "left_rgb", np.zeros((10,10)))

    assert len(buffer.entries) == 3 # Should be 3: stamp_2, stamp_3, stamp_4
    assert "stamp_1" not in buffer.entries
    assert "stamp_2" in buffer.entries