from typing import List, Tuple, Optional
import numpy as np
import cv2


def generate_distinct_color_for_index(i: int) -> Tuple[int, int, int]:
    """
    Generate a single visually distinct color for a given index using golden ratio in HSV space.
    
    This is the pure function version that allows for globally unique color assignment
    across multiple images by controlling the index externally.
    
    Args:
        i: Index of the color (0, 1, 2, ...)
        
    Returns:
        (R, G, B) tuple with values in range [0, 255]
        
    Example:
        >>> color = generate_distinct_color_for_index(0)
        >>> color
        (255, 0, 0)    # Red
        >>> color = generate_distinct_color_for_index(1)
        >>> color
        (147, 255, 0)  # Yellow-green
    """
    golden_ratio = 0.618033988749895  # (√5 - 1) / 2
    
    # Use golden ratio to space hues optimally
    hue = (i * golden_ratio) % 1.0
    
    # Convert HSV to RGB (S=1.0, V=1.0 for maximum distinction)
    if hue < 1/6.:
        r, g, b = 1, hue * 6, 0
    elif hue < 2/6.:
        r, g, b = 1 - (hue - 1/6.) * 6, 1, 0
    elif hue < 3/6.:
        r, g, b = 0, 1, (hue - 2/6.) * 6
    elif hue < 4/6.:
        r, g, b = 0, 1 - (hue - 3/6.) * 6, 1
    elif hue < 5/6.:
        r, g, b = (hue - 4/6.) * 6, 0, 1
    else:
        r, g, b = 1, 0, 1 - (hue - 5/6.) * 6
    
    # Convert to 0-255 range
    return (int(r * 255), int(g * 255), int(b * 255))


def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate n visually distinct colors using golden ratio in HSV space.
    
    Args:
        n: Number of unique colors needed
        
    Returns:
        List of (R, G, B) tuples with values in range [0, 255]
        
    Example:
        >>> colors = generate_distinct_colors(5)
        >>> colors[0]  # First color
        (255, 0, 0)    # Red
        >>> colors[1]  # Second color  
        (147, 255, 0)  # Yellow-green
    """
    return [generate_distinct_color_for_index(i) for i in range(n)]



def scale_bboxes(
    bboxes: np.ndarray,
    original_shape: Tuple[int, int],
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Scale bounding boxes to different image resolution.
    
    Args:
        bboxes: Nx4 array of [x1, y1, x2, y2] coordinates
        original_shape: (height, width) of original image
        target_shape: (height, width) of target image
        
    Returns:
        Scaled bounding boxes in target resolution
    """
    if len(bboxes) == 0:
        return bboxes
    
    orig_h, orig_w = original_shape
    target_h, target_w = target_shape
    
    # Early return if no scaling needed (defensive check)
    # This avoids unnecessary copy when shapes already match
    if orig_h == target_h and orig_w == target_w:
        return bboxes  # No copy needed - shapes already match
    
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    
    # Only copy when scaling is actually needed
    scaled_bboxes = bboxes.copy()
    scaled_bboxes[:, [0, 2]] *= scale_x  # x coordinates
    scaled_bboxes[:, [1, 3]] *= scale_y  # y coordinates
    
    return scaled_bboxes



def scale_bbox(
    bbox: np.ndarray,
    original_shape: Tuple[int, int],
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Scale a bounding box to different image resolution.
    """
    return scale_bboxes(np.array([bbox]), original_shape, target_shape)[0]

def unrotate_bboxes(
    bboxes: np.ndarray,
    rotated_image_shape: Tuple[int, int],
    backward_rotation: Optional[int]
) -> np.ndarray:
    """
    Transform bounding boxes from rotated image coordinates back to original image coordinates.
    
    Uses OpenCV's rotation logic by creating corner marker images, rotating them, and extracting
    the transformed coordinates. This leverages cv2.rotate's proven transformation logic.
    
    Args:
        bboxes: Nx4 array of [x1, y1, x2, y2] coordinates in rotated image space
        rotated_image_shape: (height, width) of the rotated image
        backward_rotation: cv2 rotation constant for backward rotation (from rotation_map['backward'])
                          or None if no rotation was applied
    Returns:
        Nx4 array of [x1, y1, x2, y2] coordinates in original image space
    """
    if len(bboxes) == 0:
        return bboxes
    
    if backward_rotation is None:
        return bboxes
    
    h_rot, w_rot = rotated_image_shape
    
    # Process each bbox by creating corner markers, rotating them, and finding transformed positions
    unrotated_bboxes = np.zeros_like(bboxes)
    
    for i in range(len(bboxes)):
        x1_rot, y1_rot = int(bboxes[i, 0]), int(bboxes[i, 1])
        x2_rot, y2_rot = int(bboxes[i, 2]), int(bboxes[i, 3])
        
        # Clamp coordinates to valid range
        x1_rot = max(0, min(w_rot - 1, x1_rot))
        y1_rot = max(0, min(h_rot - 1, y1_rot))
        x2_rot = max(0, min(w_rot - 1, x2_rot))
        y2_rot = max(0, min(h_rot - 1, y2_rot))
        
        # Create a binary image with corner markers, rotate it, then find markers
        # Use small rectangles (3x3) for each corner to make them more robust to rotation
        corner_img = np.zeros((h_rot, w_rot), dtype=np.uint8)
        marker_size = 2  # Half-size of marker (total size will be 2*marker_size+1)
        
        # Mark each corner with a distinct value in a small region
        # Top-left corner (value 1)
        y_tl = max(0, y1_rot - marker_size)
        y_tl_end = min(h_rot, y1_rot + marker_size + 1)
        x_tl = max(0, x1_rot - marker_size)
        x_tl_end = min(w_rot, x1_rot + marker_size + 1)
        corner_img[y_tl:y_tl_end, x_tl:x_tl_end] = 1
        
        # Top-right corner (value 2)
        y_tr = max(0, y1_rot - marker_size)
        y_tr_end = min(h_rot, y1_rot + marker_size + 1)
        x_tr = max(0, x2_rot - marker_size)
        x_tr_end = min(w_rot, x2_rot + marker_size + 1)
        corner_img[y_tr:y_tr_end, x_tr:x_tr_end] = 2
        
        # Bottom-left corner (value 3)
        y_bl = max(0, y2_rot - marker_size)
        y_bl_end = min(h_rot, y2_rot + marker_size + 1)
        x_bl = max(0, x1_rot - marker_size)
        x_bl_end = min(w_rot, x1_rot + marker_size + 1)
        corner_img[y_bl:y_bl_end, x_bl:x_bl_end] = 3
        
        # Bottom-right corner (value 4)
        y_br = max(0, y2_rot - marker_size)
        y_br_end = min(h_rot, y2_rot + marker_size + 1)
        x_br = max(0, x2_rot - marker_size)
        x_br_end = min(w_rot, x2_rot + marker_size + 1)
        corner_img[y_br:y_br_end, x_br:x_br_end] = 4
        
        # Rotate the corner marker image using the backward rotation
        corner_img_rotated = cv2.rotate(corner_img, backward_rotation)
        
        # Find where each corner marker ended up (use centroid of marker region)
        corners_found = {}
        for marker_val in [1, 2, 3, 4]:
            y_coords, x_coords = np.where(corner_img_rotated == marker_val)
            if len(x_coords) > 0:
                # Use centroid of the marker region
                corners_found[marker_val] = (int(np.mean(x_coords)), int(np.mean(y_coords)))
        
        # Reconstruct bbox from found corners
        if len(corners_found) == 4:
            # Get all x and y coordinates from all 4 corners
            all_x = [corners_found[i][0] for i in [1, 2, 3, 4]]
            all_y = [corners_found[i][1] for i in [1, 2, 3, 4]]
            unrotated_bboxes[i, 0] = min(all_x)  # x1
            unrotated_bboxes[i, 1] = min(all_y)  # y1
            unrotated_bboxes[i, 2] = max(all_x)  # x2
            unrotated_bboxes[i, 3] = max(all_y)  # y2
        else:
            # Fallback: if markers not found (shouldn't happen), use the bbox as-is
            unrotated_bboxes[i] = bboxes[i]
    
    return unrotated_bboxes

def create_mask_from_bbox_center(
    bbox: np.ndarray,
    image_shape: Tuple[int, int],
    mask_radius: int
) -> np.ndarray:
    """
    Create a boolean mask from a bounding box center.
    
    Creates a small region around the bbox center to sample depth values.
    This is the "bbox center sampling" strategy.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
        image_shape: (height, width) of the image
        mask_radius: Radius around bbox center to create mask
        
    Returns:
        Boolean numpy array (H, W) marking the sampled region
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=bool)
    
    # Calculate bbox center
    center_u = int((bbox[0] + bbox[2]) / 2)
    center_v = int((bbox[1] + bbox[3]) / 2)
    
    # Create region around center
    v_start = max(0, center_v - mask_radius)
    v_end = min(H, center_v + mask_radius + 1)
    u_start = max(0, center_u - mask_radius)
    u_end = min(W, center_u + mask_radius + 1)
    
    # Mark the region
    mask[v_start:v_end, u_start:u_end] = True
    
    return mask

def create_mask_from_bbox(
    bbox: np.ndarray,
    image_shape: Tuple[int, int],
    mask_radius: int = None
) -> np.ndarray:
    """
    Create a boolean mask from a bounding box covering the entire bbox area.
    
    Creates a mask covering the full bbox region, clamped to image bounds.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
        image_shape: (height, width) of the image
        mask_radius: Not used, kept for API compatibility
        
    Returns:
        Boolean numpy array (H, W) marking the bbox region
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=bool)
    
    # Extract bbox coordinates and clamp to image bounds
    x1, y1, x2, y2 = bbox
    u_start = max(0, int(x1))
    u_end = min(W, int(x2))
    v_start = max(0, int(y1))
    v_end = min(H, int(y2))
    
    # Mark the entire bbox region
    if u_start < u_end and v_start < v_end:
        mask[v_start:v_end, u_start:u_end] = True
    
    return mask

def bbox_area(bbox: np.ndarray) -> float:
    """
    Calculate the area of a bounding box.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def bbox_center_pixel(bbox: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the center pixel of a bounding box.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
    """
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

def bbox_width(bbox: np.ndarray) -> float:
    """
    Calculate the width of a bounding box.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
    """
    return bbox[2] - bbox[0]

def bbox_height(bbox: np.ndarray) -> float:

    """
    Calculate the height of a bounding box.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
    """
    return bbox[3] - bbox[1]
