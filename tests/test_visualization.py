# tests/test_visualization.py
import numpy as np
from perception_pipeline.visualization import (
    draw_detection_on_image,
    create_object_visualization,
    create_visualization
)

def test_draw_detection_on_image():
    """
    Tests that the drawing function runs without error and modifies the image.
    """
    # ARRANGE
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[25:75, 25:75] = True
    bbox = [25, 25, 50, 50]
    color = (0, 255, 0)

    # ACT
    draw_detection_on_image(image, mask, bbox, "test", color)

    # ASSERT
    assert image.sum() > 0
    assert np.any(image[50, 50] != [0, 0, 0])


def test_create_object_visualization_canvas_shape():
    """
    Tests that the created canvas has the correct dimensions based on the input image.
    """
    # ARRANGE
    cropped_image = np.zeros((150, 100, 3), dtype=np.uint8) # H=150, W=100
    label = "obj_0_1: test"

    # ACT
    canvas = create_object_visualization(cropped_image, label)

    # ASSERT
    # Expected height = img_h(150) + padding(20*2) + text_area(50) = 240
    assert isinstance(canvas, np.ndarray)
    assert canvas.shape[0] == 240
    assert canvas.shape[1] >= 140


def test_create_visualization_resizes_correctly():
    """
    Tests the complex visualization function to ensure it resizes the image
    and creates a canvas with the correct final dimensions.
    """
    # ARRANGE
    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8) # A 4:3 aspect ratio image
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:200, 100:200] = True
    bbox = [100, 100, 100, 100]
    long_text = "This is a very long description designed to test the text wrapping logic."
    
    # Constants from the function itself
    FIXED_IMAGE_HEIGHT = 256
    FINAL_CANVAS_HEIGHT = 430

    # ACT
    canvas = create_visualization(rgb_image, mask, bbox, long_text)

    # ASSERT
    assert isinstance(canvas, np.ndarray)
    
    # Check that the final canvas height is correct
    assert canvas.shape[0] == FINAL_CANVAS_HEIGHT
    
    # Check that the width was resized correctly based on the aspect ratio
    # Expected width = new_h * (orig_w / orig_h) = 256 * (640 / 480) = 341.33 -> 341
    assert canvas.shape[1] == int(FIXED_IMAGE_HEIGHT * (640 / 480))