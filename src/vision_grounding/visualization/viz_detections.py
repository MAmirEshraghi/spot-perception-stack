#!/usr/bin/env python3
"""
Visualization script for debugging object detection results.

Loads image data dumps with masks and creates debug visualizations showing:
- Original RGB image
- Green overlay for FastSAM masks
- Red bounding boxes from YOLO

Saves visualizations to offline_outputs/debug_visualizations/

Usage:
    python visualize_detections.py

Author: Robin
Date: 2024-12-16
"""

import pickle
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

# Configuration
OFFLINE_OUTPUTS_DIR = Path(__file__).parent.parent.parent / "logs" / "current_run_outputs" / "offline_outputs"
IMAGE_DATA_DUMPS_DIR = OFFLINE_OUTPUTS_DIR / "image_data_dumps"
OUTPUT_VIZ_DIR = OFFLINE_OUTPUTS_DIR / "debug_visualizations"

# Visualization colors
MASK_COLOR = (0, 255, 0)  # Green for masks
BBOX_COLOR = (0, 0, 255)  # Red for bounding boxes
BBOX_THICKNESS = 2
MASK_ALPHA = 0.4  # Transparency for mask overlay


def load_image_data_dump(pkl_path: Path) -> Dict:
    """Load a single image data dump pickle file."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def draw_bbox(image: np.ndarray, bbox_xyxy: List[float], color: tuple, thickness: int = 2, label: str = None) -> np.ndarray:
    """Draw a bounding box on an image."""
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        # Add label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(image, (x1, y1 - text_height - 6), (x1 + text_width + 4, y1), color, -1)
        
        # Draw text
        cv2.putText(image, label, (x1 + 2, y1 - 4), font, font_scale, (255, 255, 255), font_thickness)
    
    return image


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.4) -> np.ndarray:
    """Overlay a binary mask on an image with transparency."""
    # Create colored overlay
    overlay = image.copy()
    overlay[mask.astype(bool)] = color
    
    # Blend with original image
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    return result


def create_visualization(image_data: Dict, image_id: str) -> np.ndarray:
    """
    Create a visualization with masks and bounding boxes for a single image.
    
    Args:
        image_data: Dictionary containing rgb_image, rotated_rgb_image, and yolo_object_dict
        image_id: Identifier for the image
        
    Returns:
        Annotated image with masks and bounding boxes
    """
    # Use rotated RGB image if available, otherwise use original
    rgb_image = image_data.get('rotated_rgb_image')
    if rgb_image is None:
        rgb_image = image_data.get('rgb_image')
    
    if rgb_image is None:
        print(f"  ⚠ Warning: No RGB image found for {image_id}")
        return None
    
    # Create working copy
    result_image = rgb_image.copy()
    
    # Get object detections
    yolo_objects = image_data.get('yolo_object_dict', {})
    
    if not yolo_objects:
        print(f"  ⚠ No objects detected in {image_id}")
        return result_image
    
    # Process each detected object
    for obj_id, obj_data in yolo_objects.items():
        label = obj_data.get('label', 'unknown')
        confidence = obj_data.get('confidence', 0.0)
        bbox_xyxy = obj_data.get('rotated_bbox_xyxy')  # Use rotated bbox to match rotated image
        
        # Check if bbox is available
        if bbox_xyxy is None:
            bbox_xyxy = obj_data.get('bbox_xyxy')  # Fallback to non-rotated
        
        if bbox_xyxy is None:
            continue
        
        # Overlay mask if available
        if 'scaled_bbox_masks' in obj_data:
            mask = obj_data['scaled_bbox_masks']
            is_fastsam = obj_data.get('mask_is_fastsam', False)
            
            # Resize mask to match RGB image if needed
            if mask.shape[:2] != result_image.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (result_image.shape[1], result_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            
            # Convert to bool if it's uint8
            mask_bool = mask.astype(bool)
            
            # Different color for FastSAM vs fallback masks
            mask_color = MASK_COLOR if is_fastsam else (255, 255, 0)  # Cyan for fallback
            result_image = overlay_mask(result_image, mask_bool, mask_color, MASK_ALPHA)
        
        # Draw bounding box
        label_text = f"{label} ({confidence:.2f})"
        result_image = draw_bbox(result_image, bbox_xyxy, BBOX_COLOR, BBOX_THICKNESS, label_text)
    
    return result_image


def process_all_dumps(input_dir: Path, output_dir: Path):
    """
    Process all image data dumps and create visualizations.
    
    Args:
        input_dir: Directory containing pickle dumps
        output_dir: Directory to save visualizations
    """
    print(f"\n{'='*70}")
    print(f"  Creating Debug Visualizations")
    print(f"{'='*70}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all pickle files
    pkl_files = sorted(input_dir.glob("image_data_*.pkl"))
    
    if not pkl_files:
        print(f"\n  ⚠ No image data dumps found in {input_dir}")
        return
    
    print(f"\nFound {len(pkl_files)} dump files to process\n")
    
    # Statistics
    total_images = 0
    total_objects = 0
    failed_images = 0
    
    # Process each dump file
    for pkl_idx, pkl_path in enumerate(pkl_files, 1):
        print(f"[{pkl_idx}/{len(pkl_files)}] Processing {pkl_path.name}...")
        
        try:
            # Load dump
            viz_data = load_image_data_dump(pkl_path)
            
            # Process each image in the dump
            for image_id, image_data in viz_data.items():
                total_images += 1
                
                # Count objects
                num_objects = len(image_data.get('yolo_object_dict', {}))
                total_objects += num_objects
                
                # Create visualization
                result_image = create_visualization(image_data, image_id)
                
                if result_image is None:
                    failed_images += 1
                    continue
                
                # Create safe filename from image_id
                safe_image_id = image_id.replace('/', '_')
                output_path = output_dir / f"{safe_image_id}.jpg"
                
                # Save visualization
                cv2.imwrite(str(output_path), result_image)
                print(f"  ✓ Saved {safe_image_id} ({num_objects} objects)")
        
        except Exception as e:
            print(f"  ✗ Error processing {pkl_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    print(f"Total images processed:  {total_images}")
    print(f"Total objects visualized: {total_objects}")
    print(f"Failed images:           {failed_images}")
    print(f"Output directory:        {output_dir}")
    print(f"\n✓ Visualization complete!")


def create_index_html(output_dir: Path):
    """Create an HTML index file for easy viewing of all visualizations."""
    viz_files = sorted(output_dir.glob("*.jpg"))
    
    if not viz_files:
        return
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Object Detection Debug Visualizations</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        h1 { color: #333; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }
        .item { background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        img { width: 100%; height: auto; border-radius: 4px; }
        .filename { font-size: 12px; color: #666; margin-top: 8px; word-break: break-all; }
    </style>
</head>
<body>
    <h1>Object Detection Debug Visualizations</h1>
    <p>Green overlay: FastSAM masks | Red boxes: YOLO detections</p>
    <div class="grid">
"""
    
    for viz_file in viz_files:
        rel_path = viz_file.name
        html_content += f"""
        <div class="item">
            <img src="{rel_path}" alt="{rel_path}">
            <div class="filename">{rel_path}</div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    index_path = output_dir / "index.html"
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"\n  ✓ Created index.html for easy viewing")
    print(f"  Open: {index_path}")


def main():
    """Main entry point."""
    # Check if input directory exists
    if not IMAGE_DATA_DUMPS_DIR.exists():
        print(f"Error: Input directory not found: {IMAGE_DATA_DUMPS_DIR}")
        print(f"Please run the object detection pipeline first to generate image data dumps.")
        return
    
    # Process all dumps
    process_all_dumps(IMAGE_DATA_DUMPS_DIR, OUTPUT_VIZ_DIR)
    
    # Create HTML index
    create_index_html(OUTPUT_VIZ_DIR)


if __name__ == "__main__":
    main()

