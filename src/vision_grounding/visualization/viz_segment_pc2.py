#!/usr/bin/env python3
"""
Simple Segmented Point Cloud Visualizer - Color-based Toggling

This script loads a PLY point cloud and allows cycling through individual
color segments using keyboard controls.

Features:
- Press RIGHT arrow: Toggle to next color segment
- Press 'A': Show all segments
- Press 'R': Reset view

Usage:
    python viz_segment_pc2.py
    python viz_segment_pc2.py --file path/to/custom.ply
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict


class ColorSegmentVisualizer:
    """Interactive visualizer for cycling through color segments."""
    
    def __init__(self, ply_path: Path, show_stats: bool = True):
        self.ply_path = ply_path
        self.show_stats = show_stats
        
        # Load data
        self.pcd_full = None
        self.points_full = None
        self.colors_full = None
        self.unique_colors = []
        self.color_point_counts = {}
        self.current_color_idx = -1  # -1 means show all
        
        self.vis = None
        self.pcd_display = None
        
    def load_data(self):
        """Load PLY and extract unique colors."""
        print(f"\n{'='*70}")
        print(f"  Loading Segmented Point Cloud")
        print(f"{'='*70}")
        print(f"PLY file: {self.ply_path}")
        
        # Check file exists
        if not self.ply_path.exists():
            print(f"  ✗ Error: PLY file not found at {self.ply_path}")
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
        
        # Extract unique colors
        print(f"\n  Extracting unique colors...")
        unique_colors_set = set()
        for color in self.colors_full:
            unique_colors_set.add(tuple(color))
        
        # Sort colors by number of points (descending)
        color_counts = defaultdict(int)
        for color in self.colors_full:
            color_counts[tuple(color)] += 1
        
        self.unique_colors = sorted(color_counts.keys(), key=lambda c: color_counts[c], reverse=True)
        self.color_point_counts = dict(color_counts)
        
        print(f"  ✓ Found {len(self.unique_colors)} unique color segments")
        
        if self.show_stats:
            self.print_statistics()
        
        return True
    
    def print_statistics(self):
        """Print detailed statistics."""
        print(f"\n  Point Cloud Statistics:")
        print(f"    Total points:        {len(self.points_full):,}")
        print(f"    Unique color segments: {len(self.unique_colors)}")
        
        # Spatial bounds
        min_bounds = self.points_full.min(axis=0)
        max_bounds = self.points_full.max(axis=0)
        extent = max_bounds - min_bounds
        
        print(f"\n  Spatial Extent:")
        print(f"    X: [{min_bounds[0]:.2f}, {max_bounds[0]:.2f}] ({extent[0]:.2f}m)")
        print(f"    Y: [{min_bounds[1]:.2f}, {max_bounds[1]:.2f}] ({extent[1]:.2f}m)")
        print(f"    Z: [{min_bounds[2]:.2f}, {max_bounds[2]:.2f}] ({extent[2]:.2f}m)")
        
        print(f"\n  Top 15 Color Segments by Point Count:")
        for i, color in enumerate(self.unique_colors[:15], 1):
            count = self.color_point_counts[color]
            percentage = (count / len(self.points_full)) * 100
            print(f"    {i:2d}. RGB{color} → {count:6,} points ({percentage:5.2f}%)")
        
        if len(self.unique_colors) > 15:
            print(f"    ... and {len(self.unique_colors) - 15} more color segments")
    
    def filter_by_color(self, color=None):
        """Filter point cloud by color. If color is None, show all."""
        if color is None:
            # Show all
            mask = np.ones(len(self.points_full), dtype=bool)
            title = "All Segments"
        else:
            # Filter by color - use tolerant matching (±1 RGB value)
            color_array = np.array(color)
            mask = np.all(np.abs(self.colors_full.astype(int) - color_array) <= 1, axis=1)
            
            num_points = np.sum(mask)
            percentage = (num_points / len(self.points_full)) * 100
            
            # Calculate center of mass for this segment
            segment_points = self.points_full[mask]
            if len(segment_points) > 0:
                center = segment_points.mean(axis=0)
                title = (f"Segment {self.current_color_idx + 1}/{len(self.unique_colors)}: "
                        f"RGB{color} | {num_points:,} points ({percentage:.2f}%) | "
                        f"Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
            else:
                title = f"Segment {self.current_color_idx + 1}/{len(self.unique_colors)}: RGB{color} | No points"
        
        # Create filtered point cloud
        filtered_points = self.points_full[mask]
        filtered_colors = self.colors_full[mask] / 255.0  # Back to [0,1] for Open3D
        
        self.pcd_display = o3d.geometry.PointCloud()
        self.pcd_display.points = o3d.utility.Vector3dVector(filtered_points)
        self.pcd_display.colors = o3d.utility.Vector3dVector(filtered_colors)
        
        return title
    
    def toggle_next_color(self, vis):
        """Toggle to next color segment."""
        self.current_color_idx = (self.current_color_idx + 1) % (len(self.unique_colors) + 1)
        
        if self.current_color_idx == len(self.unique_colors):
            # Show all
            title = self.filter_by_color(None)
            print(f"\n  [TOGGLE] Showing all segments")
        else:
            # Show specific color
            color = self.unique_colors[self.current_color_idx]
            title = self.filter_by_color(color)
            print(f"\n  [TOGGLE] {title}")
        
        # Update visualization
        vis.clear_geometries()
        vis.add_geometry(self.pcd_display, reset_bounding_box=False)
        vis.update_renderer()
        
        return False
    
    def show_all(self, vis):
        """Show all segments."""
        self.current_color_idx = -1
        title = self.filter_by_color(None)
        print(f"\n  [ALL] Showing all segments")
        
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
        # Initialize with all segments
        self.filter_by_color(None)
        
        print(f"\n{'='*70}")
        print(f"  Launching Interactive Visualization")
        print(f"{'='*70}")
        print(f"\n  Keyboard Controls:")
        print(f"    - RIGHT Arrow:  Toggle to next color segment")
        print(f"    - 'A':          Show all segments")
        print(f"    - 'R':          Reset view")
        print(f"    - 'H':          Show all Open3D controls")
        print(f"    - 'Q' or ESC:   Quit")
        print(f"\n  Mouse Controls:")
        print(f"    - Left Button:  Rotate")
        print(f"    - Right Button: Pan")
        print(f"    - Wheel:        Zoom")
        print(f"\n  Color segments available: {len(self.unique_colors)}")
        print(f"{'='*70}\n")
        
        # Create visualization with key callbacks
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Segmented Point Cloud - Color Toggle", 
                               width=1280, height=720)
        
        # Register key callbacks
        self.vis.register_key_callback(262, self.toggle_next_color)  # RIGHT arrow
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


def visualize_segmented_pointcloud(ply_path: Path, show_stats: bool = True):
    """
    Load and visualize the color-encoded segmented point cloud.
    
    Args:
        ply_path: Path to the PLY file
        show_stats: Whether to print statistics about the point cloud
    """
    visualizer = ColorSegmentVisualizer(ply_path, show_stats)
    
    if not visualizer.load_data():
        return
    
    visualizer.visualize()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Visualize color-encoded segmented point cloud by cycling through colors"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to PLY file (default: <project_root>/logs/current_run_outputs/offline_outputs/objects_segmented_pc.ply)"
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip printing statistics"
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
    
    # Visualize
    visualize_segmented_pointcloud(ply_path, show_stats=not args.no_stats)


if __name__ == "__main__":
    main()

