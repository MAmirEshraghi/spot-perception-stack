#!/usr/bin/env python3
"""
Fast SAM Segmentation Helpers
Extracted and adapted for modular usage in object detection pipeline.

Provides two segmentation modes:
1. Box Prompt Mode: Single Fast SAM call per image with multiple box prompts
2. Crop-based Mode: Individual Fast SAM call per cropped bbox region

Author: Robin Eshraghi 
initial 12/15/25
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
from contextlib import redirect_stdout
from io import StringIO

try:
    from ultralytics import FastSAM
    from PIL import Image
    FASTSAM_AVAILABLE = True
except ImportError:
    FASTSAM_AVAILABLE = False
    FastSAM = None


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_FASTSAM_CONFIG = {
    "conf": 0.4,
    "iou": 0.9,
    "imgsz": 1024,
    "retina_masks": True,
    "device": "cuda"
}


# ============================================================================
# MODE 1: BOX PROMPT (One call per image with all bboxes)
# ============================================================================

def get_fastsam_masks_from_boxes(
    fastsam_model: FastSAM,
    image: np.ndarray,
    bboxes: List[np.ndarray],
    device: str = "cuda",
    config: Optional[dict] = None
) -> List[np.ndarray]:
    """
    Fast SAM segmentation using box prompts: one model call for all bboxes.
    
    This mode runs Fast SAM once with all bounding boxes as prompts.
    
    Args:
        fastsam_model: Initialized FastSAM model
        image: Full RGB image (H, W, 3)
        bboxes: List of [x1, y1, x2, y2] bounding boxes
        device: 'cuda' or 'cpu'
        config: Optional config dict with 'conf', 'iou', 'imgsz', 'retina_masks'
    
    Returns:
        List of boolean masks (H, W), one per bbox. Empty mask if inference fails.
    """
    if config is None:
        config = DEFAULT_FASTSAM_CONFIG
    
    H, W = image.shape[:2]
    
    # Handle empty bboxes
    if not bboxes:
        return []
    
    # Convert image to PIL
    pil_image = Image.fromarray(image)
    
    # Prepare box prompts (convert to list of lists)
    box_prompts = [bbox.tolist() if isinstance(bbox, np.ndarray) else bbox for bbox in bboxes]
    
    # Run FastSAM with box prompts
    masks = []
    try:
        with redirect_stdout(StringIO()):
            results = fastsam_model(
                pil_image,
                bboxes=box_prompts,
                device=device,
                retina_masks=config.get("retina_masks", True),
                imgsz=config.get("imgsz", 1024),
                conf=config.get("conf", 0.4),
                iou=config.get("iou", 0.9),
                verbose=False
            )
        
        # Extract masks from results
        result = results[0] if len(results) > 0 else None
        
        if result is not None and result.masks is not None and result.masks.data is not None:
            masks_tensor = result.masks.data.cpu().numpy()
            
            # Process each mask
            for i, mask_data in enumerate(masks_tensor):
                # Convert to uint8 for cv2.resize (doesn't support bool)
                mask_uint8 = mask_data.astype(np.uint8)
                
                # Resize to match image dimensions if needed
                if mask_uint8.shape != (H, W):
                    mask = cv2.resize(
                        mask_uint8,
                        (W, H),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                else:
                    mask = mask_uint8.astype(bool)
                
                masks.append(mask)
        else:
            # No masks found - return empty masks for all bboxes
            for _ in bboxes:
                masks.append(np.zeros((H, W), dtype=bool))
    
    except Exception as e:
        print(f"Warning: FastSAM box prompt inference failed: {e}")
        # Return empty masks for all bboxes
        for _ in bboxes:
            masks.append(np.zeros((H, W), dtype=bool))
    
    # Ensure we return same number of masks as bboxes
    while len(masks) < len(bboxes):
        masks.append(np.zeros((H, W), dtype=bool))
    
    return masks[:len(bboxes)]  # Trim if we got more masks than bboxes


# ============================================================================
# MODE 2: CROP-BASED (One call per bbox with cropping)
# ============================================================================

def get_fastsam_mask_from_crop(
    fastsam_model: FastSAM,
    image: np.ndarray,
    bbox: np.ndarray,
    full_image_shape: Tuple[int, int],
    mode: str = "box",
    object_label: Optional[str] = None,
    object_description: Optional[str] = None,
    device: str = "cuda",
    config: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast SAM segmentation: crop detection, segment, return mask.
    
    Supports two modes:
    - "text": Uses CLIP-based semantic matching with text prompt (object_label)
    - "box": Uses FastSAM's native box prompt feature with bounding box
    
    Both modes crop image with 20% bbox expansion. If no mask is found, logs warning and returns empty mask.
    
    Args:
        fastsam_model: Initialized FastSAM model
        image: Full RGB image (H, W, 3)
        bbox: [x1, y1, x2, y2] bounding box in full image coordinates
        full_image_shape: (H, W) of full image (for mask resizing)
        mode: Segmentation mode - "text" or "box" (default: "box")
        object_label: Optional object label - required for "text" mode
        object_description: Optional object description - not used in current implementation
        device: 'cuda' or 'cpu'
        config: Optional config dict with 'conf', 'iou', 'imgsz', 'retina_masks'
    
    Returns:
        Tuple of (full_mask, cropped_image, crop_mask, bbox_crop):
        - full_mask: Boolean mask (H, W) in full image coordinates
        - cropped_image: Cropped RGB image (crop_h, crop_w, 3)
        - crop_mask: Boolean mask (crop_h, crop_w) in crop coordinates
        - bbox_crop: Bbox [x1, y1, x2, y2] in crop coordinates
    """
    if config is None:
        config = DEFAULT_FASTSAM_CONFIG
    
    # 1. Validate inputs
    if mode not in ["text", "box"]:
        raise ValueError(f"mode must be 'text' or 'box', got '{mode}'")
    
    # 2. Crop image with 20% expansion
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    
    # Expand by 20%
    new_w = w * 1.2
    new_h = h * 1.2
    x1_expanded = cx - new_w / 2.0
    y1_expanded = cy - new_h / 2.0
    x2_expanded = cx + new_w / 2.0
    y2_expanded = cy + new_h / 2.0
    
    # Clamp to image bounds
    x1_crop = max(0, int(x1_expanded))
    y1_crop = max(0, int(y1_expanded))
    x2_crop = min(image.shape[1], int(x2_expanded))
    y2_crop = min(image.shape[0], int(y2_expanded))
    
    cropped_image = image[y1_crop:y2_crop, x1_crop:x2_crop]
    
    if cropped_image.size == 0:
        return np.zeros(full_image_shape, dtype=bool), np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((0, 0), dtype=bool), np.array([0, 0, 0, 0], dtype=np.float32)
    
    crop_h, crop_w = cropped_image.shape[:2]
    pil_crop = Image.fromarray(cropped_image)
    
    # 3. Convert bbox to crop coordinates
    bbox_crop = np.array([
        bbox[0] - x1_crop,
        bbox[1] - y1_crop,
        bbox[2] - x1_crop,
        bbox[3] - y1_crop
    ], dtype=np.float32)
    
    # 4. Run FastSAM based on mode
    combined_mask = None
    
    # Check if text mode has label
    if mode == "text" and (object_label is None or object_label.strip() == ""):
        print(f"Warning: No object_label provided for text mode, returning empty mask")
        combined_mask = np.zeros((crop_h, crop_w), dtype=bool)
    else:
        with redirect_stdout(StringIO()):
            try:
                if mode == "text":
                    results = fastsam_model(
                        pil_crop,
                        texts=object_label,
                        device=device,
                        retina_masks=config.get("retina_masks", True),
                        imgsz=config.get("imgsz", 1024),
                        conf=config.get("conf", 0.4),
                        iou=config.get("iou", 0.9),
                        verbose=False
                    )
                else:  # mode == "box"
                    results = fastsam_model(
                        pil_crop,
                        bboxes=[bbox_crop.tolist()],
                        device=device,
                        retina_masks=config.get("retina_masks", True),
                        imgsz=config.get("imgsz", 1024),
                        conf=config.get("conf", 0.4),
                        iou=config.get("iou", 0.9),
                        verbose=False
                    )
            except Exception as e:
                print(f"Warning: FastSAM inference failed: {e}, returning empty mask")
                combined_mask = np.zeros((crop_h, crop_w), dtype=bool)
        
        # 5. Extract mask from results
        if combined_mask is None:
            result = results[0] if len(results) > 0 else None
            
            if result is not None and result.masks is not None and result.masks.data is not None:
                masks_tensor = result.masks.data.cpu().numpy()
                
                if len(masks_tensor) > 0:
                    # Use first mask (best match for text mode, direct output for box mode)
                    # Convert to uint8 for cv2.resize (doesn't support bool)
                    selected_mask_uint8 = masks_tensor[0].astype(np.uint8)
                    
                    # 6. Resize mask to match crop size if needed
                    if selected_mask_uint8.shape != (crop_h, crop_w):
                        combined_mask = cv2.resize(
                            selected_mask_uint8,
                            (crop_w, crop_h),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    else:
                        combined_mask = selected_mask_uint8.astype(bool)
                else:
                    print(f"Warning: No masks found for {'text' if mode == 'text' else 'box'} mode, returning empty mask")
                    combined_mask = np.zeros((crop_h, crop_w), dtype=bool)
            else:
                print(f"Warning: FastSAM returned no results for {'text' if mode == 'text' else 'box'} mode, returning empty mask")
                combined_mask = np.zeros((crop_h, crop_w), dtype=bool)
    
    # 7. Map mask back to full image coordinates
    full_mask = np.zeros(full_image_shape, dtype=bool)
    full_mask[y1_crop:y2_crop, x1_crop:x2_crop] = combined_mask
    
    return full_mask, cropped_image, combined_mask, bbox_crop
