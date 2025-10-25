#!/usr/bin/env python3
"""
Script to visualize the combined scene point cloud AND 3D object bboxes.

This script uses the 'VisualizerWithKeyCallback' method.

Key controls:
  A: Toggle scene point cloud visibility (on/off)
  B: Toggle scene color (original color / light gray)
  
  D: Toggle ALL 3D bbox center points (on/off)
  
  E: Toggle Label Inspection Mode (on/off)
  ->: (In inspection mode) Next label
  <-: (In inspection mode) Previous label
  F: (In inspection mode) Toggle 3D bboxes for current label (on/off)
  
  q: Quit the visualizer
"""

import pickle
import open3d as o3d
import numpy as np
import sys
import os
import json
import colorsys  # Used for generating distinct colors
from typing import Union, List, Dict, Any, Tuple
from collections import defaultdict

try:
    from src_perception.obs_data_buffer import ObsDataBuffer, ObsDataEntry
except ImportError:
    print("Error: Could not import 'obs_data_buffer'.")
    sys.exit(1)

# --- Configuration ---
BUFFER_FILE_PATH = "data/obs_buffer.pkl"
BBOX_FILE_PATH = "data/bbox_3d_positions_v3.json"       #check the directory 
VOXEL_SIZE = 0.05
GRAY_COLOR = [0.7, 0.7, 0.7]
VIS_CONF_THRESHOLD = 0.50  # Only show detections with score > 0.5

# --- Buffer Loading ---

def load_buffer(filepath: str) -> Union['ObsDataBuffer', None]:
    """Loads the pickled ObsDataBuffer from the given path."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    print(f"Loading data buffer from {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            data_buffer = pickle.load(f)
        if not hasattr(data_buffer, 'entries') or not hasattr(data_buffer, 'static_transforms'):
            print(f"Error: Loaded object doesn't have expected ObsDataBuffer attributes. Got: {type(data_buffer)}")
            return None
        print(f"Successfully loaded buffer with {len(data_buffer.entries)} entries.")
        print(f"Buffer status: {data_buffer.get_buffer_status()}")
        return data_buffer
    except Exception as e:
        print(f"Error during unpickling: {e}")
        return None

def create_scene_pointcloud(data_buffer: 'ObsDataBuffer') -> Union[o3d.geometry.PointCloud, None]:
    """Iterates through all entries and combines them into a single scene."""
    if not data_buffer.is_tf_static_ready():
        print("Error: Static transforms are not complete in the loaded buffer.")
        return None
    whole_scene_pcd = o3d.geometry.PointCloud()
    processed_count = 0
    print(f"Combining point clouds from {len(data_buffer.entries)} entries...")
    for header_stamp, entry in data_buffer.entries.items():
        if entry.is_frame_full():
            try:
                pcds_list = entry.get_pointcloud(data_buffer.static_transforms)
                for pcd in pcds_list:
                    if len(pcd.points) > 0:
                        whole_scene_pcd += pcd
                processed_count += 1
            except Exception as e:
                print(f"  Error processing entry {header_stamp}: {e}")
        else:
            print(f"  Skipping incomplete entry: {header_stamp}")
    if processed_count == 0:
        print("No complete entries were found to process.")
        return None
    print(f"\nSuccessfully combined {processed_count} entries into one point cloud.")
    return whole_scene_pcd


def generate_distinct_colors(n: int) -> List[List[float]]:
    """Generates n visually distinct RGB colors."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.85
        value = 0.85
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(list(rgb))
    return colors

def load_bbox_points(filepath: str) -> Tuple[Union[o3d.geometry.PointCloud, None], Union[Dict, None]]:
    """
    Loads 3D bbox positions from JSON.
    Returns:
        1. A single PointCloud (bbox_pcd) with all points.
        2. A dictionary (label_data) with items grouped by label.
    """
    if not os.path.exists(filepath):
        print(f"Error: BBox file not found at {filepath}")
        return None, None
    
    print(f"Loading bbox data from {filepath}...")
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None, None

    if isinstance(data, dict):
        items = data.values()
    elif isinstance(data, list):
        items = data
    else:
        print("Error: Unknown JSON structure. Expected list or dict.")
        return None, None

    if not items:
        print("BBox file is empty.")
        return None, None

    # --- 1. Get unique labels and assign colors ---
    try:
        # Filter items by score *before* finding unique labels
        filtered_items = [item for item in items if item.get('score', 0.0) >= VIS_CONF_THRESHOLD]
        
        if not filtered_items:
            print(f"No detections found with score >= {VIS_CONF_THRESHOLD}")
            return o3d.geometry.PointCloud(), {"labels": [], "colors": {}, "items_by_label": {}}
            
        unique_labels = sorted(list(set(item['label'] for item in filtered_items)))
        color_list = generate_distinct_colors(len(unique_labels))
        label_to_color = {label: color for label, color in zip(unique_labels, color_list)}
        print(f"  Found {len(unique_labels)} unique labels.")
        print(f"  Filtered out {len(items) - len(filtered_items)} detections below score {VIS_CONF_THRESHOLD}")
        
    except KeyError:
        print("Error: JSON items are missing 'label' key.")
        return None, None

    # --- 2. Build both data structures ---
    all_points = []
    all_colors = []
    items_by_label = defaultdict(list)
    
    try:
        # Iterate over the PRE-FILTERED items
        for item in filtered_items:
            label = item['label']
            pos = item['position_3d']
            color = label_to_color[label]
            
            all_points.append(pos)
            all_colors.append(color)
            
            items_by_label[label].append({
                'position_3d': pos,
                'bbox_3d_corners': item.get('bbox_3d_corners'),
                'score': item.get('score', 0.0)
            })

    except KeyError:
        print("Error: JSON items are missing 'position_3d' or 'label' key.")
        return None, None
    
    # --- 3. Create 'bbox_pcd' (for 'D' toggle) ---
    bbox_pcd = o3d.geometry.PointCloud()
    bbox_pcd.points = o3d.utility.Vector3dVector(all_points)
    bbox_pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # --- 4. Create 'label_data' (for 'E' toggle) ---
    label_data = {
        "labels": unique_labels,
        "colors": label_to_color,
        "items_by_label": dict(items_by_label)
    }
    
    print(f"Successfully loaded {len(all_points)} bbox center points (post-filtering).")
    return bbox_pcd, label_data


class SceneVisualizer:
    """
    Manages the Open3D visualizer with key callbacks for multiple geometries.
    """
    def __init__(self, scene_pcd: o3d.geometry.PointCloud, 
                 bbox_pcd: o3d.geometry.PointCloud, 
                 label_data: Dict):
        
        # --- 1. Store state variables ---
        self.scene_pcd = scene_pcd
        self.bbox_pcd = bbox_pcd
        self.label_data = label_data
        
        
        # Scene state
        self.original_scene_colors = np.asarray(scene_pcd.colors).copy()
        self.is_scene_visible = True
        self.is_scene_colored = True
        
        # "All BBoxes" state (Key 'D')
        self.is_bbox_visible = True
        
        # "Inspection Mode" state (Key 'E')
        self.inspection_mode = False
        self.inspection_show_bboxes = False
        self.current_label_index = 0
        self.label_list = self.label_data.get("labels", [])
        self.current_label_pcd = o3d.geometry.PointCloud()
        self.current_label_bboxes = o3d.geometry.LineSet()
        #self.current_label_text = None 
        
        # --- 2. Create the visualizer and window ---
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Scene + BBox Viewer", width=1280, height=720)

        # --- 3. Register the callback functions ---
        self.vis.register_key_callback(65, self._toggle_scene) # 'A'
        self.vis.register_key_callback(66, self._toggle_scene_color) # 'B'
        self.vis.register_key_callback(68, self._toggle_all_bboxes) # 'D'
        self.vis.register_key_callback(69, self._toggle_inspection_mode) # 'E'
        self.vis.register_key_callback(70, self._toggle_inspection_bboxes) # 'F'
        self.vis.register_key_callback(262, self._next_label) # '->' (Right Arrow)
        self.vis.register_key_callback(263, self._prev_label) # '<-' (Left Arrow)

        # --- 4. Add initial geometries ---
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        self.vis.add_geometry(coord_frame)
        self.vis.add_geometry(self.scene_pcd)
        self.vis.add_geometry(self.bbox_pcd)

    def _toggle_scene(self, vis):
        """Callback for 'A' key: toggles scene point cloud visibility."""
        if self.is_scene_visible:
            print("[Callback] Hiding scene (press 'A' to show)")
            vis.remove_geometry(self.scene_pcd, reset_bounding_box=False)
        else:
            print("[Callback] Showing scene (press 'A' to hide)")
            vis.add_geometry(self.scene_pcd, reset_bounding_box=False)
        self.is_scene_visible = not self.is_scene_visible
        return True

    def _toggle_scene_color(self, vis):
        """Callback for 'B' key: toggles scene point cloud color."""
        if self.is_scene_colored:
            print("[Callback] Disabling scene color (gray mode)")
            self.scene_pcd.paint_uniform_color(GRAY_COLOR)
        else:
            print("[Callback] Enabling scene color (original mode)")
            self.scene_pcd.colors = o3d.utility.Vector3dVector(self.original_scene_colors)
        self.is_scene_colored = not self.is_scene_colored
        vis.update_geometry(self.scene_pcd)
        return True

    def _toggle_all_bboxes(self, vis):
        """Callback for 'D' key: toggles all bbox center points visibility."""
        
        if self.inspection_mode:
            print("[Callback] Inspection Mode disabled to show all bboxes")
            self.inspection_mode = False
            vis.remove_geometry(self.current_label_pcd, reset_bounding_box=False)
            vis.remove_geometry(self.current_label_bboxes, reset_bounding_box=False)
            

        if self.is_bbox_visible:
            print("[Callback] Hiding ALL bbox points (press 'D' to show)")
            vis.remove_geometry(self.bbox_pcd, reset_bounding_box=False)
        else:
            print("[Callback] Showing ALL bbox points (press 'D' to hide)")
            vis.add_geometry(self.bbox_pcd, reset_bounding_box=False)
        self.is_bbox_visible = not self.is_bbox_visible
        return True

    def _toggle_inspection_mode(self, vis):
        """Callback for 'E' key: toggles label inspection mode."""
        self.inspection_mode = not self.inspection_mode
        
        if self.inspection_mode:
            print("[Callback] Inspection Mode ON")
            if self.is_bbox_visible:
                print("  > Hiding all bboxes.")
                vis.remove_geometry(self.bbox_pcd, reset_bounding_box=False)
            
            self._update_label_pcd(vis) # This now also calls the text update
            if self.inspection_show_bboxes:
                self._update_label_bboxes(vis)
        
        else:
            print("[Callback] Inspection Mode OFF")
            # Hide the current label points AND boxes
            vis.remove_geometry(self.current_label_pcd, reset_bounding_box=False)
            vis.remove_geometry(self.current_label_bboxes, reset_bounding_box=False) 
            
            
            
            if self.is_bbox_visible:
                print("  > Restoring all bboxes.")
                vis.add_geometry(self.bbox_pcd, reset_bounding_box=False)
        return True

    def _toggle_inspection_bboxes(self, vis):
        """Callback for 'F' key: toggles 3D bboxes in inspection mode."""
        if not self.inspection_mode:
            print("[Callback] Press 'E' to enter inspection mode first.")
            return False
            
        self.inspection_show_bboxes = not self.inspection_show_bboxes
        
        if self.inspection_show_bboxes:
            print("[Callback] Showing inspection bboxes")
            self._update_label_bboxes(vis)
        else:
            print("[Callback] Hiding inspection bboxes")
            vis.remove_geometry(self.current_label_bboxes, reset_bounding_box=False)
        return True

    def _next_label(self, vis):
        """Callback for '->' key: shows next label in inspection mode."""
        if not self.inspection_mode or not self.label_list:
            return False
        
        self.current_label_index = (self.current_label_index + 1) % len(self.label_list)
        self._update_label_pcd(vis) # This now also calls the text update
        if self.inspection_show_bboxes:
            self._update_label_bboxes(vis)
        return True
        
    def _prev_label(self, vis):
        """Callback for '<-' key: shows previous label in inspection mode."""
        if not self.inspection_mode or not self.label_list:
            return False
            
        self.current_label_index = (self.current_label_index - 1) % len(self.label_list)
        self._update_label_pcd(vis) # This now also calls the text update
        if self.inspection_show_bboxes:
            self._update_label_bboxes(vis)
        return True
        
    def _update_label_pcd(self, vis):
        """Helper function to show the currently selected label's POINTS."""
        if not self.label_list:
            print("No labels to inspect.")
            return

        vis.remove_geometry(self.current_label_pcd, reset_bounding_box=False)
        
        label_name = self.label_list[self.current_label_index]
        items = self.label_data["items_by_label"].get(label_name, [])
        points = [item['position_3d'] for item in items if item['position_3d']]
        color = self.label_data["colors"].get(label_name, [1,0,0])
        
        # --- Update log and prepare text for 3D label ---
        avg_score = 0.0
        if items:
            scores = [item['score'] for item in items if 'score' in item]
            if scores:
                avg_score = np.mean(scores)
        
        print(f"  [Inspect] Showing {self.current_label_index + 1}/{len(self.label_list)}: {label_name} ({len(points)} points, Avg Score: {avg_score:.2f})")
        
    
        
        self.current_label_pcd = o3d.geometry.PointCloud()
        if points:
            self.current_label_pcd.points = o3d.utility.Vector3dVector(points)
            self.current_label_pcd.paint_uniform_color(color)
        
        vis.add_geometry(self.current_label_pcd, reset_bounding_box=False)

    def _update_label_bboxes(self, vis):
        """Helper function to show the currently selected label's BBOXES."""
        if not self.label_list:
            return

        vis.remove_geometry(self.current_label_bboxes, reset_bounding_box=False)
        
        label_name = self.label_list[self.current_label_index]
        items = self.label_data["items_by_label"].get(label_name, [])
        color = self.label_data["colors"].get(label_name, [1,0,0])
        
        all_points = []
        all_lines = []
        point_index = 0
        
        for item in items:
            corners = item.get('bbox_3d_corners')
            if not corners:
                continue
            
            tl = corners.get('top_left')
            tr = corners.get('top_right')
            bl = corners.get('bottom_left')
            br = corners.get('bottom_right')
            
            if not all([tl, tr, bl, br]):
                continue
                
            points = [tl, tr, br, bl]
            all_points.extend(points)
            
            idx0 = point_index
            idx1 = point_index + 1
            idx2 = point_index + 2
            idx3 = point_index + 3
            all_lines.extend([
                [idx0, idx1], [idx1, idx2],
                [idx2, idx3], [idx3, idx0]
            ])
            
            point_index += 4

        self.current_label_bboxes = o3d.geometry.LineSet()
        if all_points:
            self.current_label_bboxes.points = o3d.utility.Vector3dVector(all_points)
            self.current_label_bboxes.lines = o3d.utility.Vector2iVector(all_lines)
            self.current_label_bboxes.paint_uniform_color(color)
        
        vis.add_geometry(self.current_label_bboxes, reset_bounding_box=False)


    def run(self):
        """Starts the visualization loop."""
        print("\n" + "*"*60)
        print("      Interactive Scene Viewer Initialized ")
        print("*"*60)
        print("Controls:")
        print("  - !!! CLICK INSIDE THE WINDOW TO FOCUS IT !!!")
        print("  - [A] : Toggle scene visibility (on/off)")
        print("  - [B] : Toggle scene color (original/gray)")
        print("  - [D] : Toggle ALL bbox points (on/off)")
        print("  - [E] : Toggle Label Inspection Mode (on/off)")
        print("  - [->] / [<-] : Next/Previous label in inspection mode")
        print("  - [F] : (In inspection mode) Toggle 3D bboxes")
        print("  - [q] : Close the window")
        print("*"*60 + "\n")
        
        self.vis.run()
        self.vis.destroy_window()
        print("Visualizer closed.")

# --- Main Execution ---

def main():
    # --- 1. Load Scene Point Cloud ---
    print("\n--- Loading Scene Point Cloud ---")
    data_buffer = load_buffer(BUFFER_FILE_PATH)
    if data_buffer is None:
        print("Exiting.")
        return

    scene_pcd = create_scene_pointcloud(data_buffer)
    if scene_pcd is None or len(scene_pcd.points) == 0:
        print("Failed to create a scene point cloud or the cloud is empty.")
        return
        
    print(f"\nOriginal scene has {len(scene_pcd.points):,} points.")
    print(f"Downsampling scene with voxel size {VOXEL_SIZE}...")
    downsampled_scene_pcd = scene_pcd.voxel_down_sample(VOXEL_SIZE)
    print(f"Downsampled scene has {len(downsampled_scene_pcd.points):,} points.")

    if not downsampled_scene_pcd.has_colors():
        print("Warning: Scene point cloud has no colors.")
        downsampled_scene_pcd.paint_uniform_color(GRAY_COLOR)
        
    # --- 2. Load BBox Center Points ---
    print("\n--- Loading BBox Points ---")
    bbox_pcd, label_data = load_bbox_points(BBOX_FILE_PATH)
    
    if bbox_pcd is None:
        print("Warning: Could not load bbox points. Creating empty placeholders.")
        bbox_pcd = o3d.geometry.PointCloud()
        label_data = {"labels": [], "colors": {}, "items_by_label": {}}

    # --- 3. Run Visualizer ---
    app = SceneVisualizer(downsampled_scene_pcd, bbox_pcd, label_data)
    app.run()

if __name__ == "__main__":
    main()