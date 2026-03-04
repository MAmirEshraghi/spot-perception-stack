#!/usr/bin/env python3
"""
Visualize Color-Encoded Segmented Point Cloud with Label Filtering

This script loads and visualizes the objects_segmented_pc.ply file
where each detected object has a unique distinct color. It also loads
the corresponding object_list.json to map colors to labels.

Features:
- Press RIGHT arrow: Toggle through labels (show only one label at a time)
- Press 'A': Show all objects
- Press 'R': Reset view
- Downsampling support for faster processing of large point clouds

Usage:
    python viz_segment_pc.py
    python viz_segment_pc.py --file path/to/custom.ply --json path/to/custom.json
    python viz_segment_pc.py --downsample 2  # Keep every 2nd point for faster processing
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import argparse
import json
from collections import defaultdict, Counter


class LabelToggleVisualizer:
    """Interactive visualizer with label filtering via keyboard."""
    
    def __init__(self, ply_path: Path, json_path: Path, show_stats: bool = True, downsample_factor: int = 1):
        self.ply_path = ply_path
        self.json_path = json_path
        self.show_stats = show_stats
        self.downsample_factor = downsample_factor
        
        # Load data
        self.pcd_full = None
        self.points_full = None
        self.colors_full = None
        self.color_to_label = {}
        self.label_list = []
        self.current_label_idx = -1  # -1 means show all
        
        self.vis = None
        self.pcd_display = None
        
    def load_data(self):
        """Load PLY and JSON, create color-to-label mapping."""
        print(f"\n{'='*70}")
        print(f"  Loading Segmented Point Cloud with Labels")
        print(f"{'='*70}")
        print(f"PLY file: {self.ply_path}")
        print(f"JSON file: {self.json_path}")
        
        # Check files exist
        if not self.ply_path.exists():
            print(f"  ✗ Error: PLY file not found at {self.ply_path}")
            return False
        
        if not self.json_path.exists():
            print(f"  ✗ Error: JSON file not found at {self.json_path}")
            return False
        
        # Load point cloud
        print(f"\n  Loading point cloud...")
        self.pcd_full = o3d.io.read_point_cloud(str(self.ply_path))
        
        if len(self.pcd_full.points) == 0:
            print(f"  ✗ Error: Point cloud is empty")
            return False
        
        self.points_full = np.asarray(self.pcd_full.points)
        self.colors_full = (np.asarray(self.pcd_full.colors) * 255).astype(np.uint8)
        
        print(f"  ✓ Loaded {len(self.points_full):,} points")
        
        # Apply downsampling if requested
        if self.downsample_factor > 1:
            print(f"\n  Downsampling by factor {self.downsample_factor}...")
            points_before = len(self.points_full)
            # Use uniform downsampling (keep every Nth point)
            indices = np.arange(0, len(self.points_full), self.downsample_factor)
            self.points_full = self.points_full[indices]
            self.colors_full = self.colors_full[indices]
            points_after = len(self.points_full)
            print(f"    Points before: {points_before:,} | After: {points_after:,} ({100*points_after/points_before:.1f}%)")
        
        # Load JSON
        print(f"\n  Loading object labels...")
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        objects = data.get("objects", [])
        print(f"  ✓ Loaded {len(objects)} object records")
        
        # Build color-to-label mapping
        print(f"\n  Building color-to-label mapping...")
        label_counts = defaultdict(int)
        label_to_colors = defaultdict(list)  # Track which colors belong to which label
        
        for obj in objects:
            color_rgb = obj.get("detection_metadata", {}).get("distinct_color_rgb")
            label = obj.get("semantic_metadata", {}).get("label", "unknown")
            
            if color_rgb:
                color_key = tuple(color_rgb)
                self.color_to_label[color_key] = label
                label_counts[label] += 1
                if color_key not in label_to_colors[label]:
                    label_to_colors[label].append(color_key)
        
        # Filter out points with colors NOT in JSON (invalid objects)
        print(f"\n  Filtering out invalid objects...")
        points_before = len(self.points_full)
        print(f"    Points before filtering: {points_before:,}")
        
        # Build mask for points that have colors in JSON (OPTIMIZED - set-based lookup)
        print(f"    Building color lookup set...")
        valid_colors_set = set(self.color_to_label.keys())
        print(f"    Checking {len(self.points_full):,} points against {len(valid_colors_set)} valid colors...")
        # Vectorized membership check - much faster than nested loops
        # Convert each point's color to tuple and check set membership
        valid_mask = np.array([tuple(c) in valid_colors_set for c in self.colors_full], dtype=bool)
        
        # Apply filter
        self.points_full = self.points_full[valid_mask]
        self.colors_full = self.colors_full[valid_mask]
        points_after = len(self.points_full)
        points_removed = points_before - points_after
        
        print(f"    Points after filtering:  {points_after:,}")
        print(f"    Invalid points removed:  {points_removed:,} ({100*points_removed/points_before:.1f}%)")
        
        # Store label statistics for display during toggle
        self.label_counts = label_counts  # Objects per label in JSON
        self.label_to_colors = label_to_colors  # Unique colors per label
        
        # Get unique labels sorted by count
        self.label_list = sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True)
        
        print(f"  ✓ Mapped {len(self.color_to_label)} colors to labels")
        print(f"  ✓ Found {len(self.label_list)} unique labels")
        
        if self.show_stats:
            self.print_statistics(label_counts)
            self.print_color_matching_details(label_to_colors, label_counts)
        
        return True
    
    def print_statistics(self, label_counts):
        """Print detailed statistics."""
        print(f"\n  Point Cloud Statistics:")
        print(f"    Total points:     {len(self.points_full):,}")
        print(f"    Total objects:    {len(self.color_to_label)}")
        print(f"    Unique labels:    {len(self.label_list)}")
        
        # Spatial bounds
        min_bounds = self.points_full.min(axis=0)
        max_bounds = self.points_full.max(axis=0)
        extent = max_bounds - min_bounds
        
        print(f"\n  Spatial Extent:")
        print(f"    X: [{min_bounds[0]:.2f}, {max_bounds[0]:.2f}] ({extent[0]:.2f}m)")
        print(f"    Y: [{min_bounds[1]:.2f}, {max_bounds[1]:.2f}] ({extent[1]:.2f}m)")
        print(f"    Z: [{min_bounds[2]:.2f}, {max_bounds[2]:.2f}] ({extent[2]:.2f}m)")
        
        print(f"\n  Object Counts by Label (top 15):")
        for i, (label, count) in enumerate(sorted(label_counts.items(), 
                                                   key=lambda x: x[1], 
                                                   reverse=True)[:15], 1):
            print(f"    {i:2d}. {label:25s} {count:4d} objects")
        
        if len(self.label_list) > 15:
            print(f"    ... and {len(self.label_list) - 15} more labels")
    
    def print_color_matching_details(self, label_to_colors, label_counts):
        """Print detailed color matching information between JSON and PLY."""
        print(f"\n{'='*70}")
        print(f"  Color Matching Details (JSON vs PLY)")
        print(f"{'='*70}")
        
        # Pre-compute color counts for all colors (OPTIMIZED - single pass)
        print(f"  Pre-computing color counts...")
        color_tuples = [tuple(c) for c in self.colors_full]
        color_counts = Counter(color_tuples)
        
        # Show top 10 labels
        top_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for i, (label, obj_count) in enumerate(top_labels, 1):
            colors_for_label = label_to_colors[label]
            print(f"\n  [{i}] Label: '{label}' ({obj_count} objects, {len(colors_for_label)} unique colors)")
            print(f"      {'-'*66}")
            
            # For each color assigned to this label
            total_ply_points = 0
            for j, color_key in enumerate(colors_for_label[:5], 1):  # Show first 5 colors
                # Look up count from pre-computed dictionary (O(1) lookup)
                num_ply_points = color_counts.get(color_key, 0)
                total_ply_points += num_ply_points
                
                # Status indicator
                if num_ply_points > 0:
                    status = "✓"
                else:
                    status = "✗"
                
                print(f"      {status} Object {j}: JSON RGB{color_key}")
                print(f"         → PLY matches: {num_ply_points:,} points")
                
                if num_ply_points > 0:
                    print(f"         → Exact match: RGB{color_key}")
            
            if len(colors_for_label) > 5:
                print(f"      ... and {len(colors_for_label) - 5} more colors for this label")
            
            print(f"      → Total PLY points for '{label}': {total_ply_points:,}")
        
        if len(label_counts) > 10:
            print(f"\n  ... and {len(label_counts) - 10} more labels")
        print(f"{'='*70}\n")
    
    def filter_by_label(self, label=None):
        """Filter point cloud by label. If label is None, show all."""
        if label is None:
            # Show all
            mask = np.ones(len(self.points_full), dtype=bool)
            title = "All Objects"
        else:
            # Filter by label (OPTIMIZED - set-based lookup)
            # Get all colors for this label
            label_colors_set = set()
            for color_key, obj_label in self.color_to_label.items():
                if obj_label == label:
                    label_colors_set.add(color_key)
            
            # Vectorized membership check - much faster than nested loops
            mask = np.array([tuple(c) in label_colors_set for c in self.colors_full], dtype=bool)
            
            num_points = np.sum(mask)
            title = f"Label: {label} ({num_points:,} points)"
        
        # Create filtered point cloud
        filtered_points = self.points_full[mask]
        filtered_colors = self.colors_full[mask] / 255.0  # Back to [0,1] for Open3D
        
        self.pcd_display = o3d.geometry.PointCloud()
        self.pcd_display.points = o3d.utility.Vector3dVector(filtered_points)
        self.pcd_display.colors = o3d.utility.Vector3dVector(filtered_colors)
        
        return title
    
    def key_callback(self, vis, action, mods):
        """Handle keyboard events."""
        if action == 1:  # Key press (not release)
            return False
        
        # Get the key code
        return False
    
    def toggle_next_label(self, vis):
        """Toggle to next label."""
        self.current_label_idx = (self.current_label_idx + 1) % (len(self.label_list) + 1)
        
        if self.current_label_idx == len(self.label_list):
            # Show all
            title = self.filter_by_label(None)
            print(f"\n  [TOGGLE] Showing all objects")
        else:
            # Show specific label
            label = self.label_list[self.current_label_idx]
            title = self.filter_by_label(label)
            
            # Get statistics for this label
            json_obj_count = self.label_counts.get(label, 0)
            ply_color_count = len(self.label_to_colors.get(label, []))
            num_points = np.sum(np.ones(len(self.pcd_display.points), dtype=bool))
            
            print(f"\n  [TOGGLE] {title} | JSON: {json_obj_count} objs | PLY: {ply_color_count} colors")
        
        # Update visualization
        vis.clear_geometries()
        vis.add_geometry(self.pcd_display, reset_bounding_box=False)
        vis.update_renderer()
        
        return False
    
    def show_all(self, vis):
        """Show all objects."""
        self.current_label_idx = -1
        title = self.filter_by_label(None)
        print(f"\n  [ALL] Showing all objects")
        
        vis.clear_geometries()
        vis.add_geometry(self.pcd_display, reset_bounding_box=False)
        vis.update_renderer()
        
        return False
    
    def reset_view(self, vis):
        """Reset camera view."""
        print(f"\n  [RESET] Resetting view")
        vis.reset_view_point(True)
        return False
    
    def visualize(self):
        """Run interactive visualization."""
        # Initialize with all objects
        self.filter_by_label(None)
        
        print(f"\n{'='*70}")
        print(f"  Launching Interactive Visualization")
        print(f"{'='*70}")
        print(f"\n  Keyboard Controls:")
        print(f"    - RIGHT Arrow:  Toggle to next label")
        print(f"    - 'A':          Show all objects")
        print(f"    - 'R':          Reset view")
        print(f"    - 'H':          Show all Open3D controls")
        print(f"    - 'Q' or ESC:   Quit")
        print(f"\n  Mouse Controls:")
        print(f"    - Left Button:  Rotate")
        print(f"    - Right Button: Pan")
        print(f"    - Wheel:        Zoom")
        print(f"\n  Labels available: {len(self.label_list)}")
        print(f"{'='*70}\n")
        
        # Create visualization with key callbacks
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Segmented Point Cloud - Label Filter", 
                               width=1280, height=720)
        
        # Register key callbacks
        self.vis.register_key_callback(262, self.toggle_next_label)  # RIGHT arrow
        self.vis.register_key_callback(65, self.show_all)  # 'A' key
        self.vis.register_key_callback(82, self.reset_view)  # 'R' key
        
        # Add geometry
        self.vis.add_geometry(self.pcd_display)
        
        # Set view options
        opt = self.vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.1, 0.1, 0.1])
        
        # Reset view to see entire point cloud
        self.vis.reset_view_point(True)
        
        # Run
        self.vis.run()
        self.vis.destroy_window()
        
        print(f"\n  Visualization closed.")


def visualize_segmented_pointcloud(ply_path: Path, json_path: Path, show_stats: bool = True, downsample_factor: int = 1):
    """
    Load and visualize the color-encoded segmented point cloud with label filtering.
    
    Args:
        ply_path: Path to the PLY file
        json_path: Path to the JSON file with object labels
        show_stats: Whether to print statistics about the point cloud
        downsample_factor: Downsampling factor (e.g., 2 = keep every 2nd point, 3 = keep every 3rd point). 
                          Default: 1 (no downsampling)
    """
    visualizer = LabelToggleVisualizer(ply_path, json_path, show_stats, downsample_factor)
    
    if not visualizer.load_data():
        return
    
    visualizer.visualize()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Visualize color-encoded segmented point cloud with label filtering"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to PLY file (default: <project_root>/logs/current_run_outputs/offline_outputs/objects_segmented_pc.ply)"
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to JSON file with labels (default: <project_root>/logs/current_run_outputs/offline_outputs/object_list.json)"
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip printing statistics"
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Downsampling factor (e.g., 2 = keep every 2nd point, 3 = keep every 3rd point). Default: 1 (no downsampling). Higher values speed up processing."
    )
    
    args = parser.parse_args()
    
    # Determine file paths
    script_dir = Path(__file__).resolve().parent
    # Navigate to project root (tiamatl_eval_mvp/) - 2 levels up from vision_grounding folder
    project_root = script_dir.parent.parent
    
    if args.file:
        ply_path = Path(args.file)
    else:
        # Default path at project root - check multiple possible locations
        # First try offline_outputs folder (from z_sensor_object_map_node.py)
        ply_path = project_root / "logs" / "current_run_outputs" / "offline_outputs" / "objects_segmented_pc.ply"
        
        # Fallback to direct current_run_outputs if offline version doesn't exist
        if not ply_path.exists():
            ply_path = project_root / "logs" / "current_run_outputs" / "objects_segmented_pc.ply"
        
        # Also check in the vision_grounding/logs folder (for standalone pipeline runs)
        if not ply_path.exists():
            ply_path = script_dir / "logs" / "current_run_outputs" / "objects_segmented_pc.ply"
    
    if args.json:
        json_path = Path(args.json)
    else:
        # Default path at project root - check multiple possible locations
        # First try offline_outputs folder (from z_sensor_object_map_node.py)
        json_path = project_root / "logs" / "current_run_outputs" / "offline_outputs" / "object_list.json"
        
        # Fallback to direct current_run_outputs if offline version doesn't exist
        if not json_path.exists():
            json_path = project_root / "logs" / "current_run_outputs" / "object_list.json"
        
        # Also check in the vision_grounding/logs folder (for standalone pipeline runs)
        if not json_path.exists():
            json_path = script_dir / "logs" / "current_run_outputs" / "object_list.json"
    
    # Validate downsample factor
    if args.downsample < 1:
        print(f"  ✗ Error: Downsample factor must be >= 1, got {args.downsample}")
        return
    
    # Visualize
    visualize_segmented_pointcloud(ply_path, json_path, show_stats=not args.no_stats, downsample_factor=args.downsample)


if __name__ == "__main__":
    main()

