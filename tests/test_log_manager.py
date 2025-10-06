import pytest
import numpy as np
from argparse import Namespace
from perception_pipeline.log_manager import PerceptionLog

@pytest.fixture
def mock_args():
    """A pytest fixture to create a reusable, fake args object."""
    return Namespace(
        VLM_MODEL_ID='Test/Model-v1',
        PROCESS_LIMIT=10,
        centroid_threshold=100,
        coverage_threshold=0.5,
        voxel_size=0.1,
        vlm_padding=0.1,
        min_mask_area=50
    )
# This fixture can be shared by all tests in this file
@pytest.fixture
def log_manager(tmp_path, mock_args):
    """A fixture to provide an initialized PerceptionLog instance."""
    return PerceptionLog(args=mock_args, base_dir=tmp_path)

def test_add_image_creates_files_and_updates_data(log_manager, tmp_path):
    """
    Tests if the add_image method correctly saves image/depth files
    and updates the internal data dictionary.
    """
    # ARRANGE: Create dummy image, depth, and pose data
    dummy_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_depth = np.random.rand(480, 640).astype(np.float32)
    dummy_pose = np.array([1.0, 0, 0, 0, 0.1, 0.2, 0.3])

    # ACT: Call the method to be tested
    image_id = log_manager.add_image(dummy_rgb, dummy_depth, dummy_pose)

    # ASSERT: Check that the output is correct
    assert image_id == "img_000"
    
    # Assert that the files were physically created in the temp directory
    expected_rgb_path = tmp_path / log_manager.scan_id / "rgb" / "img_000.png"
    expected_depth_path = tmp_path / log_manager.scan_id / "depth" / "img_000.npy"
    assert expected_rgb_path.exists()
    assert expected_depth_path.exists()

    # Assert that the internal data dictionary was correctly updated
    assert "img_000" in log_manager.data["images"]
    assert log_manager.data["images"]["img_000"]["camera_pose_world"] == dummy_pose.tolist()

def test_add_object_instance_updates_data(log_manager):
    """
    Tests if adding an object instance correctly updates the data dictionary.
    """
    # ARRANGE: First add a dummy image and unique object to link to
    log_manager.add_image(np.zeros((10,10,3), dtype=np.uint8), np.zeros((10,10)), np.zeros(7))
    log_manager.add_or_update_unique_object("obj_0_1", "test object", np.array([1,1,1]))
    
    # ACT: Add a new object instance
    instance_id = log_manager.add_object_instance(
        image_id="img_000",
        object_id="obj_0_1",
        bbox=[10, 20, 30, 40],
        mask_area=1200,
        mask_np=np.ones((10,10), dtype=bool)
    )

    # ASSERT: Check the internal data structures
    assert instance_id == "inst_0000"
    assert "inst_0000" in log_manager.data["object_instances"]
    assert log_manager.data["object_instances"]["inst_0000"]["parent_object_id"] == "obj_0_1"
    assert "inst_0000" in log_manager.data["unique_objects"]["obj_0_1"]["instances"]