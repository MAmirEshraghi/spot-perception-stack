#!/usr/bin/env python3
"""
Visualize Individual Object Point Clouds with ID-Based Label Filtering

This script loads individual PLY files (one per object) from a directory
and visualizes them with label filtering. It uses object_id from filenames
to match with JSON records, bypassing color matching entirely.

Features:
- Press RIGHT arrow: Toggle through labels (show only one label at a time)
- Press 'A': Show all objects
- Press 'R': Reset view
- Downsampling support for faster processing of large point clouds
- ID-based matching (more accurate than color matching)

Usage:
    python viz_segment_pc_individual.py
    python viz_segment_pc_individual.py --dir path/to/individual_objects --json path/to/object_list.json
    python viz_segment_pc_individual.py --downsample 2  # Keep every 2nd point for faster processing
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import argparse
import json
from collections import defaultdict


class LabelToggleVisualizer:
    """Interactive visualizer with ID-based label filtering via keyboard."""
    
    def __init__(self, ply_dir: Path, json_path: Path, show_stats: bool = True, downsample_factor: int = 1):
        self.ply_dir = ply_dir
        self.json_path = json_path
        self.show_stats = show_stats
        self.downsample_factor = downsample_factor
        
        # Load data
        self.object_id_to_label = {}  # object_id -> label mapping
        self.label_to_pcds = defaultdict(list)  # label -> list of point clouds
        self.all_pcds = []  # All point clouds for "show all"
        self.label_list = []
        self.current_label_idx = -1  # -1 means show all
        
        # Statistics
        self.label_counts = defaultdict(int)  # label -> count of objects
        self.object_id_to_pcd = {}  # object_id -> point cloud (for stats)
        
        self.vis = None
        self.pcd_display = None
        
    def load_data(self):
        """Load individual PLY files and JSON, create ID-to-label mapping."""
        print(f"\n{'='*70}")
        print(f"  Loading Individual Object Point Clouds with Labels")
        print(f"{'='*70}")
        print(f"PLY directory: {self.ply_dir}")
        print(f"JSON file: {self.json_path}")
        
        # Check directory exists
        if not self.ply_dir.exists():
            print(f"  ✗ Error: PLY directory not found at {self.ply_dir}")
            return False
        
        if not self.json_path.exists():
            print(f"  ✗ Error: JSON file not found at {self.json_path}")
            return False
        
        # Load JSON first to build object_id -> label mapping
        print(f"\n  Loading object labels from JSON...")
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        objects = data.get("objects", [])
        print(f"  ✓ Loaded {len(objects)} object records")
        
        # Build object_id -> label mapping based on frame_id and per-frame index
        print(f"\n  Building object_id to label mapping...")

        # Track how many objects we've seen per frame_id to reconstruct local indices
        per_frame_counts = defaultdict(int)

        for obj in objects:
            frame_meta = obj.get("frame_metadata", {})
            frame_id = frame_meta.get("frame_id")  # e.g. "1765453374.821996404/head_rgb_right"
            label = obj.get("semantic_metadata", {}).get("label", "unknown")

            if frame_id is None:
                continue

            # Local index within this frame (0, 1, 2, ...) in insertion order
            local_idx = per_frame_counts[frame_id]
            per_frame_counts[frame_id] += 1

            # Reconstruct the same ID pattern used when saving PLYs:
            # full_object_id = f"{image_id}_bbox_{local_object_id}"
            # where local_object_id was "object_{idx}"
            full_object_id = f"{frame_id}_bbox_object_{local_idx}"

            # Sanitize exactly like save_individual_object_pointclouds
            sanitized_object_id = (
                full_object_id
                .replace('/', '_')
                .replace(':', '_')
                .replace('\\', '_')
            )

            self.object_id_to_label[sanitized_object_id] = label
            self.label_counts[label] += 1
        
        print(f"  ✓ Mapped {len(self.object_id_to_label)} object IDs to labels")
        print(f"  ✓ Found {len(self.label_counts)} unique labels")
        
        # Load all PLY files from directory
        print(f"\n  Loading individual PLY files from directory...")
        ply_files = sorted(self.ply_dir.glob("*.ply"))
        
        if len(ply_files) == 0:
            print(f"  ✗ Error: No PLY files found in {self.ply_dir}")
            return False
        
        print(f"  Found {len(ply_files)} PLY files")
        
        loaded_count = 0
        matched_count = 0
        unmatched_count = 0
        
        for ply_file in ply_files:
            # Extract object_id from filename (remove .ply extension)
            object_id = ply_file.stem  # Gets filename without extension
            
            # Load point cloud
            try:
                pcd = o3d.io.read_point_cloud(str(ply_file))
                
                if len(pcd.points) == 0:
                    unmatched_count += 1
                    continue
                
                # Apply downsampling if requested
                if self.downsample_factor > 1:
                    points = np.asarray(pcd.points)
                    indices = np.arange(0, len(points), self.downsample_factor)
                    pcd = pcd.select_by_index(indices)
                
                # Check if object_id exists in JSON
                if object_id in self.object_id_to_label:
                    label = self.object_id_to_label[object_id]
                    self.label_to_pcds[label].append(pcd)
                    self.object_id_to_pcd[object_id] = pcd
                    self.all_pcds.append(pcd)
                    matched_count += 1
                else:
                    # Object ID not found in JSON - still load but mark as unmatched
                    print(f"    ⚠ Warning: {object_id} not found in JSON, skipping")
                    unmatched_count += 1
                
                loaded_count += 1
                
            except Exception as e:
                print(f"    ✗ Error loading {ply_file.name}: {e}")
                unmatched_count += 1
                continue
        
        print(f"  ✓ Loaded {loaded_count} PLY files")
        print(f"  ✓ Matched {matched_count} objects to labels")
        if unmatched_count > 0:
            print(f"  ⚠ {unmatched_count} files could not be matched or loaded")
        
        # Get unique labels sorted by count
        self.label_list = sorted(self.label_counts.keys(), key=lambda x: self.label_counts[x], reverse=True)
        
        # Calculate total points across all objects
        total_points = sum(len(pcd.points) for pcd in self.all_pcds)
        print(f"  ✓ Total points across all objects: {total_points:,}")
        
        if self.show_stats:
            self.print_statistics()
        
        return True
    
    def print_statistics(self):
        """Print detailed statistics."""
        total_points = sum(len(pcd.points) for pcd in self.all_pcds)
        
        print(f"\n  Point Cloud Statistics:")
        print(f"    Total points:     {total_points:,}")
        print(f"    Total objects:    {len(self.all_pcds)}")
        print(f"    Unique labels:    {len(self.label_list)}")
        
        # Calculate spatial bounds from all point clouds
        if self.all_pcds:
            all_points = np.vstack([np.asarray(pcd.points) for pcd in self.all_pcds])
            min_bounds = all_points.min(axis=0)
            max_bounds = all_points.max(axis=0)
            extent = max_bounds - min_bounds
            
            print(f"\n  Spatial Extent:")
            print(f"    X: [{min_bounds[0]:.2f}, {max_bounds[0]:.2f}] ({extent[0]:.2f}m)")
            print(f"    Y: [{min_bounds[1]:.2f}, {max_bounds[1]:.2f}] ({extent[1]:.2f}m)")
            print(f"    Z: [{min_bounds[2]:.2f}, {max_bounds[2]:.2f}] ({extent[2]:.2f}m)")
        
        print(f"\n  Object Counts by Label (top 15):")
        for i, (label, count) in enumerate(sorted(self.label_counts.items(), 
                                                   key=lambda x: x[1], 
                                                   reverse=True)[:15], 1):
            # Count points for this label
            label_pcds = self.label_to_pcds[label]
            label_points = sum(len(pcd.points) for pcd in label_pcds)
            print(f"    {i:2d}. {label:25s} {count:4d} objects ({label_points:,} points)")
        
        if len(self.label_list) > 15:
            print(f"    ... and {len(self.label_list) - 15} more labels")
    
    def filter_by_label(self, label=None):
        """Filter point clouds by label. If label is None, show all."""
        if label is None:
            # Show all - combine all point clouds
            if not self.all_pcds:
                self.pcd_display = o3d.geometry.PointCloud()
                title = "All Objects (0 objects)"
            else:
                combined = o3d.geometry.PointCloud()
                for pcd in self.all_pcds:
                    combined += pcd
                self.pcd_display = combined
                total_points = sum(len(pcd.points) for pcd in self.all_pcds)
                title = f"All Objects ({len(self.all_pcds)} objects, {total_points:,} points)"
        else:
            # Show specific label - combine point clouds for this label
            label_pcds = self.label_to_pcds.get(label, [])
            
            if not label_pcds:
                self.pcd_display = o3d.geometry.PointCloud()
                title = f"Label: {label} (0 objects)"
            else:
                combined = o3d.geometry.PointCloud()
                for pcd in label_pcds:
                    combined += pcd
                self.pcd_display = combined
                total_points = sum(len(pcd.points) for pcd in label_pcds)
                title = f"Label: {label} ({len(label_pcds)} objects, {total_points:,} points)"
        
        return title
    
    def toggle_next_label(self, vis):
        """Toggle to next label."""
        if len(self.label_list) == 0:
            return False
        
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
            num_objects = len(self.label_to_pcds.get(label, []))
            num_points = len(self.pcd_display.points) if self.pcd_display else 0
            
            print(f"\n  [TOGGLE] {title} | JSON: {json_obj_count} objs | PLY: {num_objects} files")
        
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
        self.vis.create_window(window_name="Individual Object Point Clouds - ID-Based Label Filter", 
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


def visualize_individual_pointclouds(ply_dir: Path, json_path: Path, show_stats: bool = True, downsample_factor: int = 1):
    """
    Load and visualize individual object point clouds with ID-based label filtering.
    
    Args:
        ply_dir: Path to directory containing individual PLY files (named by object_id)
        json_path: Path to JSON file with object labels
        show_stats: Whether to print statistics about the point clouds
        downsample_factor: Downsampling factor (e.g., 2 = keep every 2nd point, 3 = keep every 3rd point). 
                          Default: 1 (no downsampling)
    """
    visualizer = LabelToggleVisualizer(ply_dir, json_path, show_stats, downsample_factor)
    
    if not visualizer.load_data():
        return
    
    visualizer.visualize()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Visualize individual object point clouds with ID-based label filtering"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Path to directory with individual PLY files (default: <project_root>/logs/current_run_outputs/offline_outputs/individual_objects)"
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
    
    if args.dir:
        ply_dir = Path(args.dir)
    else:
        # Default path at project root - check multiple possible locations
        # First try offline_outputs/individual_objects folder
        ply_dir = project_root / "logs" / "current_run_outputs" / "offline_outputs" / "individual_objects"
        
        # Fallback to direct individual_objects if offline_outputs doesn't exist
        if not ply_dir.exists():
            ply_dir = project_root / "logs" / "current_run_outputs" / "offline_outputs" / "individual_objects"
        
        # Also check in the vision_grounding/logs folder (for standalone pipeline runs)
        if not ply_dir.exists():
            ply_dir = script_dir / "logs" / "current_run_outputs" / "offline_outputs" / "individual_objects"
    
    if args.json:
        json_path = Path(args.json)
    else:
        # Default path at project root - check multiple possible locations
        # First try offline_outputs folder
        json_path = project_root / "logs" / "current_run_outputs" / "offline_outputs" / "object_list.json"
        
        # Fallback to direct current_run_outputs if offline version doesn't exist
        if not json_path.exists():
            json_path = project_root / "logs" / "current_run_outputs" / "offline_outputs" / "object_list.json"
        
        # Also check in the vision_grounding/logs folder (for standalone pipeline runs)
        if not json_path.exists():
            json_path = script_dir / "logs" / "current_run_outputs" / "offline_outputs" /  "object_list.json"
    
    # Validate downsample factor
    if args.downsample < 1:
        print(f"  ✗ Error: Downsample factor must be >= 1, got {args.downsample}")
        return
    
    # Visualize
    visualize_individual_pointclouds(ply_dir, json_path, show_stats=not args.no_stats, downsample_factor=args.downsample)


if __name__ == "__main__":
    main()

