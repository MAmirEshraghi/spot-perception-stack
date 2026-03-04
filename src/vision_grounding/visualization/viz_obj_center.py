#!/usr/bin/env python3
"""
Visualize 3D object centers from object_list.json
Shows each detected object as a colored sphere at its 3D position
Also shows the environment point cloud in gray

Keyboard Controls:
  Right Arrow (->): Next label
  Left Arrow (<-): Previous label
  'A': Show all labels
  'Q': Quit
"""

import json
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import hashlib
from collections import defaultdict

def generate_color_for_label(label: str) -> Tuple[float, float, float]:
    """
    Generate a consistent color for a given label.
    Same label always gets the same color.
    """
    color_hash = int(hashlib.md5(label.encode()).hexdigest()[:6], 16)
    r = ((color_hash >> 16) & 0xFF) / 255.0
    g = ((color_hash >> 8) & 0xFF) / 255.0
    b = (color_hash & 0xFF) / 255.0
    
    # Make colors more vibrant
    max_val = max(r, g, b)
    if max_val < 0.3:
        r, g, b = r * 1.5, g * 1.5, b * 1.5
    if max_val > 0.9:
        r, g, b = r * 0.8, g * 0.8, b * 0.8
    
    return (r, g, b)


def visualize_object_centers(
    object_list_path: Path, 
    sphere_radius: float = 0.1,
    pointcloud_path: Optional[Path] = None,
    show_environment: bool = True
):
    """
    Load object_list.json and visualize 3D positions as colored spheres.
    Supports keyboard navigation to filter by label.
    """
    if not object_list_path.exists():
        print(f"Error: File not found: {object_list_path}")
        return
    
    # Load object list
    print(f"Loading object list from: {object_list_path}")
    with open(object_list_path, 'r') as f:
        data = json.load(f)
    
    objects = data.get('objects', [])
    metadata = data.get('metadata', {})
    
    print(f"\n{'='*60}")
    print(f"Object List Information")
    print(f"{'='*60}")
    print(f"  Total objects: {len(objects)}")
    print(f"  Processed frames: {metadata.get('processed_frames', 'N/A')}")
    print(f"  Session ID: {metadata.get('session_id', 'N/A')}")
    
    # Group objects by label and create spheres
    label_spheres: Dict[str, List[o3d.geometry.TriangleMesh]] = defaultdict(list)
    all_spheres: List[o3d.geometry.TriangleMesh] = []
    valid_count = 0
    invalid_count = 0
    label_counts = {}
    
    for obj in objects:
        spatial = obj.get('spatial_metadata', {})
        semantic = obj.get('semantic_metadata', {})
        detection = obj.get('detection_metadata', {})
        
        pos_3d = spatial.get('position_3d')
        is_valid = spatial.get('is_valid_depth', False)
        label = semantic.get('label', 'unknown')
        
        if pos_3d is None:
            invalid_count += 1
            continue
        
        if not isinstance(pos_3d, list) or len(pos_3d) != 3:
            invalid_count += 1
            continue
        
        if not is_valid:
            invalid_count += 1
            continue
        
        # Count labels
        label_counts[label] = label_counts.get(label, 0) + 1
        
        # Generate color for label
        color = generate_color_for_label(label)
        
        # Create sphere at position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(pos_3d)
        sphere.paint_uniform_color(color)
        
        # Add to both all_spheres and label-specific list
        all_spheres.append(sphere)
        label_spheres[label].append(sphere)
        valid_count += 1
    
    # Get sorted list of labels
    sorted_labels = sorted(label_counts.keys())
    
    print(f"\n{'='*60}")
    print(f"Label Summary:")
    for label in sorted_labels:
        print(f"  {label:20s}: {label_counts[label]:3d} objects")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Valid objects with 3D positions: {valid_count}")
    print(f"  Invalid/missing positions: {invalid_count}")
    print(f"  Unique labels: {len(sorted_labels)}")
    print(f"{'='*60}")
    
    # Load environment point cloud if requested
    env_pcd = None
    if show_environment:
        if pointcloud_path is None:
            possible_paths = [
                object_list_path.parent / "objects_segmented_pc.ply",
                object_list_path.parent / "all_points.ply",
                object_list_path.parent.parent / "objects_segmented_pc.ply",
            ]
            
            pointcloud_path = None
            for path in possible_paths:
                if path.exists():
                    pointcloud_path = path
                    break
        
        if pointcloud_path is not None and pointcloud_path.exists():
            print(f"\n{'='*60}")
            print(f"Loading environment point cloud...")
            print(f"  Path: {pointcloud_path}")
            try:
                env_pcd = o3d.io.read_point_cloud(str(pointcloud_path))
                
                if len(env_pcd.points) > 0:
                    gray_color = [0.5, 0.5, 0.5]
                    env_pcd.paint_uniform_color(gray_color)
                    
                    # Always downsample with constant voxel size
                    original_count = len(env_pcd.points)
                    print(f"  Downsampling point cloud ({original_count:,} points)...")
                    env_pcd = env_pcd.voxel_down_sample(voxel_size=0.1)
                    print(f"  After downsampling: {len(env_pcd.points):,} points (voxel_size=0.1m)")
                    
                    print(f"  ✓ Loaded {len(env_pcd.points):,} environment points (gray)")
                else:
                    print(f"  ⚠ Point cloud file is empty")
                    env_pcd = None
            except Exception as e:
                print(f"  ⚠ Failed to load point cloud: {e}")
                env_pcd = None
    
    if not all_spheres:
        print("\n❌ No valid 3D positions found to visualize!")
        return
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    # Initialize visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="3D Object Centers - Label Navigation", width=1024, height=768)
    
    # Add all geometries initially
    if env_pcd is not None:
        vis.add_geometry(env_pcd)
    for sphere in all_spheres:
        vis.add_geometry(sphere)
    vis.add_geometry(coord_frame)
    
    # State tracking
    current_label_index = -1  # -1 means "show all"
    view_controller = vis.get_view_control()
    
    def update_visualization():
        """Update visualization based on current label index."""
        nonlocal current_label_index
        
        # Clear all spheres
        vis.clear_geometries()
        
        # Add environment and coordinate frame (always visible)
        if env_pcd is not None:
            vis.add_geometry(env_pcd)
        vis.add_geometry(coord_frame)
        
        # Add spheres based on current mode
        if current_label_index == -1:
            # Show all
            for sphere in all_spheres:
                vis.add_geometry(sphere)
            count = len(all_spheres)
        else:
            # Show specific label
            if 0 <= current_label_index < len(sorted_labels):
                label = sorted_labels[current_label_index]
                for sphere in label_spheres[label]:
                    vis.add_geometry(sphere)
                count = len(label_spheres[label])
            else:
                # Fallback to all
                for sphere in all_spheres:
                    vis.add_geometry(sphere)
                count = len(all_spheres)
                current_label_index = -1
        
        # Print status (window name can't be updated dynamically in Open3D)
        if current_label_index == -1:
            print(f"\n📊 Showing ALL labels ({count} objects)")
        else:
            label = sorted_labels[current_label_index]
            print(f"\n📊 Showing label: {label} ({count} objects) [{current_label_index + 1}/{len(sorted_labels)}]")
    
    def key_callback_next(vis):
        """Right arrow: Next label"""
        nonlocal current_label_index
        if len(sorted_labels) == 0:
            return False
        
        if current_label_index < len(sorted_labels) - 1:
            current_label_index += 1
        else:
            current_label_index = -1  # Wrap to "all"
        
        update_visualization()
        return False
    
    def key_callback_prev(vis):
        """Left arrow: Previous label"""
        nonlocal current_label_index
        if len(sorted_labels) == 0:
            return False
        
        if current_label_index > -1:
            current_label_index -= 1
        else:
            current_label_index = len(sorted_labels) - 1  # Wrap from "all" to last
        
        update_visualization()
        return False
    
    def key_callback_all(vis):
        """'A' key: Show all labels"""
        nonlocal current_label_index
        current_label_index = -1
        update_visualization()
        return False
    
    def key_callback_quit(vis):
        """'Q' key: Quit"""
        return True
    
    # Register keyboard callbacks
    # Right arrow: 262, Left arrow: 263 (Open3D key codes)
    vis.register_key_callback(262, key_callback_next)  # Right arrow
    vis.register_key_callback(263, key_callback_prev)  # Left arrow
    vis.register_key_callback(ord('A'), key_callback_all)  # 'A' key
    vis.register_key_callback(ord('a'), key_callback_all)  # 'a' key
    vis.register_key_callback(ord('Q'), key_callback_quit)  # 'Q' key
    vis.register_key_callback(ord('q'), key_callback_quit)  # 'q' key
    
    # Initial update
    update_visualization()
    
    print(f"\n🎨 Opening visualization window...")
    print(f"  Controls:")
    print(f"    - Mouse drag: Rotate view")
    print(f"    - Shift + Mouse drag: Pan")
    print(f"    - Scroll: Zoom in/out")
    print(f"    - Right Arrow (->): Next label")
    print(f"    - Left Arrow (<-): Previous label")
    print(f"    - 'A': Show all labels")
    print(f"    - 'Q': Quit")
    print(f"\n  Starting view: ALL labels ({len(all_spheres)} objects)")
    print(f"  Available labels: {len(sorted_labels)}")
    
    # Run visualization
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize 3D object centers from object_list.json')
    parser.add_argument(
        '--json-path',
        type=str,
        default='logs/current_run_outputs/object_list.json',
        help='Path to object_list.json file'
    )
    parser.add_argument(
        '--pointcloud-path',
        type=str,
        default=None,
        help='Path to point cloud PLY file (auto-detected if not provided)'
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=0.1,
        help='Radius of spheres in meters (default: 0.1m = 10cm)'
    )
    parser.add_argument(
        '--no-environment',
        action='store_true',
        help='Disable environment point cloud visualization'
    )
    
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    pointcloud_path = Path(args.pointcloud_path) if args.pointcloud_path else None
    
    visualize_object_centers(
        json_path, 
        sphere_radius=args.radius,
        pointcloud_path=pointcloud_path,
        show_environment=not args.no_environment
    )
