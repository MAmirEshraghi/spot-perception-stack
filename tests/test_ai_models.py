import numpy as np
from unittest.mock import patch, MagicMock
from perception_pipeline.ai_models import get_vlm_description

@patch('perception_pipeline.ai_models.Image.fromarray')
def test_get_vlm_description_with_mocking(mock_fromarray):
    """
    Tests the VLM description function by mocking the model and processor
    with a more realistic data flow.
    """
    # ARRANGE: Create mocks for the model and processor
    mock_model = MagicMock()
    mock_processor = MagicMock()
    
    # Create a mock for the 'inputs' object that the processor returns
    mock_inputs = MagicMock()
    
    # Configure it to handle dictionary-style access for 'input_ids'
    mock_inputs.__getitem__.return_value = MagicMock(shape=[1, 10])
    
    # Configure its .to() method to simply return itself
    mock_inputs.to.return_value = mock_inputs
    
    # Now, make the processor return our fully configured mock_inputs object
    mock_processor.return_value = mock_inputs
    
    # Configure the rest of the mocks as before
    mock_model.generate.return_value = np.array([[0, 1, 2, 3, 4]])
    mock_processor.batch_decode.return_value = [" a test description "]

    # Create a dummy input image
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # ACT: Call the function
    description = get_vlm_description(dummy_image, mock_model, mock_processor)

    # ASSERT: Check the result
    assert description == "a test description"
    mock_model.generate.assert_called_once()
    mock_processor.batch_decode.assert_called_once()