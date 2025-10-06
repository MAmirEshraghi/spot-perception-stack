# src/perception_pipeline/visualization.py
import numpy as np
import cv2
from PIL import Image
import os



def draw_detection_on_image(image_to_draw_on, mask_np, bbox, label_text, color_rgb):
    """Draws a single detection (mask, bbox, label) on an image."""
    overlay_color = np.array(color_rgb, dtype=np.uint8)
    masked_area = image_to_draw_on[mask_np]
    blended_pixels = (masked_area * 0.5 + overlay_color * 0.5).astype(np.uint8)
    image_to_draw_on[mask_np] = blended_pixels

    x, y, w, h = [int(c) for c in bbox]
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.rectangle(image_to_draw_on, (x, y), (x + w, y + h), color_bgr, 2)
    
    text_position = (x, y - 10 if y > 10 else y + 10)
    cv2.putText(img=image_to_draw_on, text=label_text, org=text_position, 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                color=(255, 255, 255), thickness=2)

def create_object_visualization(cropped_image_np, label_text):
    """Creates a standardized canvas showing a cropped object with its label."""
    padding, text_area_height, font_scale, font_thickness = 20, 50, 0.6, 2
    obj_h, obj_w, _ = cropped_image_np.shape
    (text_w, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    canvas_w = max(obj_w + (2 * padding), text_w + (2 * padding))
    canvas_h = obj_h + (2 * padding) + text_area_height
    canvas = np.full((canvas_h, canvas_w, 3), 240, dtype=np.uint8)
    
    paste_x = (canvas_w - obj_w) // 2
    paste_y = padding + text_area_height
    canvas[paste_y : paste_y + obj_h, paste_x : paste_x + obj_w] = cropped_image_np
    
    text_pos = (padding, padding + 20) 
    cv2.putText(img=canvas, text=label_text, org=text_pos, 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, 
                color=(0, 0, 0), thickness=font_thickness)
    return canvas

def create_visualization(rgb_image_np, mask_np, bbox, description_text):
    """Creates a detailed visualization with a masked object and wrapped text."""
    FIXED_IMAGE_HEIGHT = 256
    FINAL_CANVAS_HEIGHT = 430
    
    viz_image = rgb_image_np.copy()
    green_color = np.array([0, 255, 0], dtype=np.uint8)
    masked_area = viz_image[mask_np]
    blended_pixels = (masked_area * 0.5 + green_color * 0.5).astype(np.uint8)
    viz_image[mask_np] = blended_pixels
    x, y, w, h = [int(c) for c in bbox]
    cv2.rectangle(viz_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    orig_h, orig_w, _ = viz_image.shape
    aspect_ratio = orig_w / orig_h
    new_w = int(FIXED_IMAGE_HEIGHT * aspect_ratio)
    resized_image = cv2.resize(viz_image, (new_w, FIXED_IMAGE_HEIGHT))

    final_canvas = np.full((FINAL_CANVAS_HEIGHT, new_w, 3), 255, dtype=np.uint8)
    final_canvas[0:FIXED_IMAGE_HEIGHT, 0:new_w] = resized_image

    # Text wrapping logic
    font, font_scale, font_thickness, line_height = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1, 25
    padding_left_right = 10
    words = description_text.split(' ')
    wrapped_lines, current_line = [], ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)
        if text_width > (new_w - 2 * padding_left_right):
            wrapped_lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    wrapped_lines.append(current_line)

    text_y_start = FIXED_IMAGE_HEIGHT + padding_left_right + 15
    for i, line in enumerate(wrapped_lines):
        text_position = (padding_left_right, text_y_start + i * line_height)
        cv2.putText(final_canvas, line, text_position, font, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)

    return final_canvas


def save_merge_tracking_visualization(object_id, image_id, mask_index, rgb_np, bbox, scan_dir):
    """Saves a visualization for a single object instance to the merge_tracking folder."""
    merge_folder_path = os.path.join(scan_dir, "merge_tracking", object_id)
    os.makedirs(merge_folder_path, exist_ok=True)
    
    x, y, w, h = [int(c) for c in bbox]
    cropped_image = rgb_np[y:y+h, x:x+w]
    
    instance_label = f"Source: {image_id}_mask_{mask_index}"
    instance_viz = create_object_visualization(cropped_image, instance_label)
    
    instance_path = os.path.join(merge_folder_path, f"{instance_label}.png")
    Image.fromarray(instance_viz).save(instance_path)

def save_object_frame_visualization(object_id, description, best_instance_info, perception_log):
    """Saves the best full-frame view of a unique object with its mask and description."""
    parent_image_id = best_instance_info["image_id"]
    parent_image_path = perception_log.data["images"][parent_image_id]["rgb_path"]
    rgb_for_viz = np.array(Image.open(parent_image_path))

    # Find the instance_id to load the correct mask
    for inst_id, inst_data in perception_log.data["object_instances"].items():
        if inst_data["parent_image_id"] == parent_image_id and inst_data["bounding_box"] == best_instance_info["bbox"]:
            mask_np = np.load(inst_data["mask_path"])
            viz_image = create_visualization(
                rgb_for_viz, mask_np, best_instance_info["bbox"], f"{object_id}: {description}"
            )
            viz_path = os.path.join(perception_log.scan_dir, "obj_frame_visualizations", f"{object_id}.png")
            Image.fromarray(viz_image).save(viz_path)
            perception_log.add_visualization_path_to_object(object_id, viz_path)
            return # Stop after finding the matching instance

def save_full_frame_visualization(image_id, detections, perception_log):
    """Saves a full frame with all its detected object masks and labels drawn on."""
    frame_rgb_path = perception_log.data["images"][image_id]["rgb_path"]
    frame_viz_image = np.array(Image.open(frame_rgb_path))
    
    for det in detections:
        obj_id = det["object_id"]
        desc = perception_log.data["unique_objects"][obj_id].get("vlm_description", "N/A")
        label = f"{obj_id}: {desc}"
        color = (0, 255, 0) if det["is_new"] else (255, 0, 0)
        draw_detection_on_image(frame_viz_image, det["mask"], det["bbox"], label, color)
        
    frame_viz_path = os.path.join(perception_log.scan_dir, "frame_visualizations", f"{image_id}.png")
    Image.fromarray(frame_viz_image).save(frame_viz_path)

def save_best_object_instance_visualization(object_id, best_instance_data, perception_log):
    """Saves the best cropped view of a unique object to the object_visualization folder."""
    parent_image_path = perception_log.data["images"][best_instance_data["image_id"]]["rgb_path"]
    rgb_np = np.array(Image.open(parent_image_path))
    
    x, y, w, h = [int(c) for c in best_instance_data["bbox"]]
    cropped_image = rgb_np[y:y+h, x:x+w]
    
    desc = perception_log.data["unique_objects"][object_id]["vlm_description"]
    obj_label = f"{object_id}: {desc}"
    
    object_viz_image = create_object_visualization(cropped_image, obj_label)
    obj_viz_path = os.path.join(perception_log.scan_dir, "object_visualization", f"{object_id}.png")
    Image.fromarray(object_viz_image).save(obj_viz_path)