# tests/test_integration_pipeline.py
import pytest
import numpy as np
import os
from argparse import Namespace

# Import components from your pipeline
from perception_pipeline.log_manager import PerceptionLog
from perception_pipeline.point_cloud import get_organized_point_cloud, get_o3d_cam_intrinsic
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# ==============================================================================
# Fixtures for the Test Harness
# ==============================================================================

@pytest.fixture(scope="module")
def mock_args():
    """A fixture to create a reusable, fake args object for tests."""
    return Namespace(
        VLM_MODEL_ID='Test/Fake-Model-v1', PROCESS_LIMIT=10, centroid_threshold=100,
        coverage_threshold=0.5, voxel_size=0.1, vlm_padding=0.1, min_mask_area=100
    )

@pytest.fixture
def predictable_image_data():
    """
    Creates simple, predictable fake data: a black image with a white square,
    a flat depth map, and a basic camera pose.
    """
    # a 50x50 white square on a 200x300 black background
    rgb_image = np.zeros((200, 300, 3), dtype=np.uint8)
    rgb_image[75:125, 100:150] = 255  # y1:y2, x1:x2

    # a depth map where everything is 2 meters away
    depth_image = np.full((200, 300), 2.0, dtype=np.float32)
    
    # camera pose: at origin, looking forward
    camera_pose = np.array([1.0, 0, 0, 0, 0, 0, 0])

    return rgb_image, depth_image, camera_pose

# ==============================================================================
# Integration Test Cases
# ==============================================================================

@pytest.mark.integration
def test_sam_to_point_cloud_integration(predictable_image_data):
    """
    Tests the integration between SAM and the point cloud generation.
    - Does the real SAM model find our predictable object?
    - Can we use its mask to correctly extract points from the point cloud?
    """
    # skip this test if the SAM checkpoint path isn't set as an environment variable
    sam_checkpoint_path = os.environ.get("SAM_CHECKPOINT_PATH")
    if not sam_checkpoint_path or not os.path.exists(sam_checkpoint_path):
        pytest.skip("SAM_CHECKPOINT_PATH environment variable not set or file not found. Skipping integration test.")

    # 1) ARRANGE: Load the real SAM model and get our predictable data
    sam = sam_model_registry["vit_l"](checkpoint=sam_checkpoint_path)
    mask_generator = SamAutomaticMaskGenerator(model=sam, min_mask_region_area=100)
    rgb_image, depth_image, camera_pose = predictable_image_data
    
    # 2) ACT part:
    # run SAM to get masks
    masks = mask_generator.generate(rgb_image)
    
    # generate the full point cloud
    h, w = depth_image.shape
    intrinsics = get_o3d_cam_intrinsic(h, w)
    full_pcd_array = get_organized_point_cloud(depth_image, camera_pose, intrinsics)
    
    # use the largest mask from SAM to select points
    assert len(masks) > 0, "SAM should have found at least one mask for the white square."
    largest_mask = max(masks, key=lambda m: m['area'])
    object_points = full_pcd_array[largest_mask['segmentation']]
    
    # 3) ASSERT:
    # number of points we extracted should be roughly equal to the area of the mask.
    assert object_points.shape[0] > 0
    assert object_points.shape[0] == pytest.approx(largest_mask['area'], abs=10) # abs tolerance for edge pixels

@pytest.mark.integration
def test_processing_to_logging_integration(mock_args, tmp_path):
    """
    Tests the integration between the main processing logic and the PerceptionLog.
    - When the pipeline identifies and logs an object, is the data structure updated correctly?
    """
    # 1) ARRANGE part: Create a logger and some fake processing results
    log_manager = PerceptionLog(args=mock_args, base_dir=tmp_path)
    object_id = "obj_0_1"
    image_id = "img_000"
    
    # 2) ACT part:
    # simulate finding a new unique object
    log_manager.add_or_update_unique_object(
        object_id=object_id,
        description="[AWAITING VLM]",
        world_position=np.array([1.0, 2.0, 3.0])
    )
    # simulate logging the instance of that object found in a frame
    instance_id = log_manager.add_object_instance(
        image_id=image_id,
        object_id=object_id,
        bbox=[10, 20, 30, 40],
        mask_area=1200,
        mask_np=np.ones((100, 100), dtype=bool)
    )

    # 3) ASSERT part: Check that the internal data dictionary was linked correctly
    unique_object_data = log_manager.data["unique_objects"][object_id]
    object_instance_data = log_manager.data["object_instances"][instance_id]
    
    assert object_instance_data["parent_object_id"] == object_id
    assert instance_id in unique_object_data["instances"]