#!/usr/bin/env python3
"""
Interactive Visualization for Deduplicated Objects

This script loads and visualizes deduplicated object point clouds from the
deduplication output directory. It allows interactive navigation through labels.

Features:
- Shows all merged objects initially
- Press RIGHT arrow: Navigate to next label (forward)
- Press LEFT arrow: Navigate to previous label (backward)
- Press 'A': Toggle full RGB scene (all_points.ply) on/off
- Press 'R': Reset view

Usage:
    python viz_deduplicated_objects.py
    python viz_deduplicated_objects.py --dir path/to/deduplicated_objects
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import argparse
import json
from collections import defaultdict
from typing import Dict, List, Optional


class DeduplicatedObjectsVisualizer:
    """Interactive visualizer for deduplicated objects with label navigation."""
    
    def __init__(self, dedup_dir: Path, show_stats: bool = True, show_all: bool = False, 
                 scene_path: Optional[Path] = None, scene_downsample: int = 1):
        self.dedup_dir = dedup_dir
        self.show_stats = show_stats
        self.show_all = show_all  # If True, show all objects; if False, only merged objects
        self.scene_path = scene_path  # Optional path to full RGB scene point cloud
        self.scene_downsample = scene_downsample  # Downsampling factor for scene (1 = no downsampling)
        
        # Data storage
        self.all_pcds = []  # List of all deduplicated point clouds
        self.label_to_pcds = {}  # Dict[label] -> List[point clouds]
        self.label_list = []  # Sorted list of labels
        self.current_label_idx = -1  # -1 means show all
        self.summary = None  # Store summary for statistics
        
        # Individual (pre-deduplication) object point clouds (optional overlay)
        self.indiv_dir: Optional[Path] = None
        self.indiv_pcds: List[o3d.geometry.PointCloud] = []
        self.indiv_label_to_pcds: Dict[str, List[o3d.geometry.PointCloud]] = {}
        
        # Full-scene RGB point cloud (optional)
        self.scene_pcd: Optional[o3d.geometry.PointCloud] = None
        self.scene_visible: bool = False
        self.indiv_visible: bool = False
        self.dedup_visible: bool = True
        
        self.vis = None
        self.pcd_display = None
        
    def load_data(self):
        """Load all deduplicated PLY files and summary JSON."""
        print(f"\n{'='*70}")
        print(f"  Loading Deduplicated Objects")
        print(f"{'='*70}")
        print(f"Directory: {self.dedup_dir}")
        
        if not self.dedup_dir.exists():
            print(f"  ✗ Error: Directory not found at {self.dedup_dir}")
            return False
        
        # Load summary JSON
        summary_path = self.dedup_dir / "deduplication_summary.json"
        if not summary_path.exists():
            print(f"  ✗ Error: Summary file not found at {summary_path}")
            return False
        
        print(f"\n  Loading summary...")
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        self.summary = summary  # Store for later use
        
        objects_by_label = summary.get("objects_by_label", {})
        files = summary.get("files", [])
        
        # Filter files based on show_all flag
        if not self.show_all:
            # Only show merged objects (num_merged >= 2)
            filtered_files = [f for f in files if f.get("num_merged", 1) >= 2]
            print(f"  ✓ Found {len(files)} total objects, {len(filtered_files)} merged objects (filtering non-merged)")
        else:
            # Show all objects
            filtered_files = files
            print(f"  ✓ Found {len(files)} objects across {len(objects_by_label)} labels (showing all)")
        
        # Load filtered PLY files
        print(f"\n  Loading point cloud files...")
        label_to_pcds = defaultdict(list)
        all_pcds = []
        
        merged_count = 0
        non_merged_count = 0
        
        for file_info in filtered_files:
            filename = file_info.get("filename")
            label = file_info.get("label")
            filepath = self.dedup_dir / filename
            
            if not filepath.exists():
                print(f"    ⚠ Warning: File not found: {filename}")
                continue
            
            try:
                pcd = o3d.io.read_point_cloud(str(filepath))
                if len(pcd.points) == 0:
                    print(f"    ⚠ Warning: Empty point cloud: {filename}")
                    continue
                
                label_to_pcds[label].append(pcd)
                all_pcds.append(pcd)
                
                num_points = len(pcd.points)
                num_merged = file_info.get("num_merged", 1)
                
                if num_merged >= 3:
                    merged_count += 1
                    print(f"    ✓ {filename}: {num_points:,} points (merged from {num_merged} objects)")
                else:
                    non_merged_count += 1
                    print(f"    ✓ {filename}: {num_points:,} points (single object, not merged)")
                
            except Exception as e:
                print(f"    ✗ Error loading {filename}: {e}")
                continue
        
        if not self.show_all:
            print(f"\n  Filtered: {merged_count} merged objects, {non_merged_count} non-merged objects excluded")
        
        self.label_to_pcds = dict(label_to_pcds)
        self.all_pcds = all_pcds
        
        # Sort labels by count (most objects first)
        self.label_list = sorted(
            self.label_to_pcds.keys(),
            key=lambda x: len(self.label_to_pcds[x]),
            reverse=True
        )
        
        print(f"\n  ✓ Loaded {len(all_pcds)} point clouds")
        print(f"  ✓ Organized into {len(self.label_list)} labels")
        
        # Load individual (non-deduplicated) object point clouds, grouped by label, if available
        # Default location: sibling folder of deduplicated_objects -> individual_objects
        self.indiv_dir = self.dedup_dir.parent / "individual_objects"
        object_list_path = self.dedup_dir.parent / "object_list.json"
        object_id_to_label: Dict[str, str] = {}

        # Build object_id -> label mapping from object_list.json (same logic as individual saver)
        if object_list_path.exists():
            try:
                with open(object_list_path, "r") as f:
                    obj_data = json.load(f)
                objects = obj_data.get("objects", [])
                from collections import defaultdict as _dd
                per_frame_counts = _dd(int)
                for obj in objects:
                    frame_meta = obj.get("frame_metadata", {})
                    frame_id = frame_meta.get("frame_id")
                    label = obj.get("semantic_metadata", {}).get("label", "unknown")
                    if frame_id is None:
                        continue
                    local_idx = per_frame_counts[frame_id]
                    per_frame_counts[frame_id] += 1
                    full_object_id = f"{frame_id}_bbox_object_{local_idx}"
                    sanitized_object_id = (
                        full_object_id
                        .replace("/", "_")
                        .replace(":", "_")
                        .replace("\\", "_")
                    )
                    object_id_to_label[sanitized_object_id] = label
                print(f"  ✓ Built mapping for {len(object_id_to_label)} individual object IDs from {object_list_path}")
            except Exception as e:
                print(f"  ⚠ Failed to load object_list.json for individual labels: {e}")

        self.indiv_label_to_pcds = defaultdict(list)

        if self.indiv_dir.exists():
            print(f"\n  Loading individual object point clouds from: {self.indiv_dir}")
            indiv_files = sorted(self.indiv_dir.glob("*.ply"))
            if indiv_files:
                for ply_file in indiv_files:
                    obj_id = ply_file.stem
                    label = object_id_to_label.get(obj_id, "unknown")
                    try:
                        pcd = o3d.io.read_point_cloud(str(ply_file))
                        if len(pcd.points) == 0:
                            continue
                        self.indiv_pcds.append(pcd)
                        self.indiv_label_to_pcds[label].append(pcd)
                    except Exception as e:
                        print(f"    ⚠ Warning: Failed to load individual PLY {ply_file.name}: {e}")
                print(f"  ✓ Loaded {len(self.indiv_pcds)} individual object point clouds across {len(self.indiv_label_to_pcds)} labels")
            else:
                print(f"  ⚠ No individual PLY files found in {self.indiv_dir}")
        else:
            print(f"  ⚠ Individual objects directory not found at {self.indiv_dir} (skipping)")
        
        # Optionally load full-scene RGB point cloud
        if self.scene_path is not None:
            print(f"\n  Loading full-scene RGB point cloud (all_points)...")
            if self.scene_path.exists():
                try:
                    self.scene_pcd = o3d.io.read_point_cloud(str(self.scene_path))
                    points_before = len(self.scene_pcd.points)
                    
                    # Apply downsampling if requested (preserves RGB colors)
                    if self.scene_downsample > 1:
                        print(f"  Downsampling scene by factor {self.scene_downsample}...")
                        points = np.asarray(self.scene_pcd.points)
                        colors = np.asarray(self.scene_pcd.colors)
                        # Uniform downsampling (keep every Nth point)
                        indices = np.arange(0, len(points), self.scene_downsample)
                        self.scene_pcd.points = o3d.utility.Vector3dVector(points[indices])
                        self.scene_pcd.colors = o3d.utility.Vector3dVector(colors[indices])
                        points_after = len(self.scene_pcd.points)
                        print(f"    Points before: {points_before:,} | After: {points_after:,} ({100*points_after/points_before:.1f}%)")
                    
                    # Verify RGB colors are present
                    if len(self.scene_pcd.colors) > 0:
                        print(f"  ✓ Loaded full RGB scene from: {self.scene_path} ({len(self.scene_pcd.points):,} points, RGB colors preserved)")
                    else:
                        print(f"  ⚠ Warning: Scene point cloud has no colors (RGB)")
                        print(f"  ✓ Loaded full scene from: {self.scene_path} ({len(self.scene_pcd.points):,} points)")
                except Exception as e:
                    print(f"  ✗ Error loading scene point cloud '{self.scene_path}': {e}")
                    self.scene_pcd = None
            else:
                print(f"  ⚠ Scene point cloud not found at {self.scene_path} (skipping)")
        
        if self.show_stats:
            self.print_statistics(summary)
        
        return True
    
    def print_statistics(self, summary: Dict):
        """Print detailed statistics."""
        print(f"\n{'='*70}")
        print(f"  Statistics")
        print(f"{'='*70}")
        
        stats = summary.get("statistics", {})
        print(f"\n  Deduplication Results:")
        print(f"    Objects before: {stats.get('total_objects_before', 0)}")
        print(f"    Objects after:  {stats.get('total_objects_after', 0)}")
        print(f"    Total merges:   {stats.get('total_merges', 0)}")
        
        # Count merged vs non-merged
        all_files = summary.get("files", [])
        merged_files = [f for f in all_files if f.get("num_merged", 1) >= 2]
        non_merged_files = [f for f in all_files if f.get("num_merged", 1) == 1]
        
        print(f"\n  Object Status:")
        print(f"    Merged objects (2+):    {len(merged_files)}")
        print(f"    Non-merged objects (1): {len(non_merged_files)}")
        
        if not self.show_all:
            print(f"\n  ⚠ Filtering: Only showing merged objects (use --all to show all)")
        
        print(f"\n  Objects by Label (showing {len(self.label_to_pcds)} labels):")
        for i, (label, pcd_list) in enumerate(
            sorted(self.label_to_pcds.items(), key=lambda x: len(x[1]), reverse=True)[:15],
            1
        ):
            count = len(pcd_list)
            total_points = sum(len(pcd.points) for pcd in pcd_list)
            print(f"    {i:2d}. {label:30s}: {count:3d} objects ({total_points:,} points)")
        
        if len(self.label_list) > 15:
            print(f"    ... and {len(self.label_list) - 15} more labels")
    
    def create_combined_pcd(self, pcds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
        """Combine multiple point clouds into one."""
        if not pcds:
            return o3d.geometry.PointCloud()
        
        combined = o3d.geometry.PointCloud()
        for pcd in pcds:
            combined += pcd
        
        return combined
    
    def build_current_view(self) -> o3d.geometry.PointCloud:
        """
        Build the current view based on:
        - deduplicated objects (by current label or all) if enabled
        - individual objects overlay if enabled
        - full-scene RGB overlay if enabled
        """
        combined = o3d.geometry.PointCloud()
        
        # Deduplicated objects (base)
        if self.dedup_visible:
            if self.current_label_idx == -1:
                base_pcds = self.all_pcds
            else:
                if 0 <= self.current_label_idx < len(self.label_list):
                    label = self.label_list[self.current_label_idx]
                    base_pcds = self.label_to_pcds.get(label, [])
                else:
                    base_pcds = self.all_pcds
            base_pcd = self.create_combined_pcd(base_pcds)
            combined += base_pcd
        
        # Individual objects overlay (label-synchronized)
        if self.indiv_visible and self.indiv_pcds:
            if self.current_label_idx == -1:
                indiv_pcds = self.indiv_pcds
            else:
                if 0 <= self.current_label_idx < len(self.label_list):
                    label = self.label_list[self.current_label_idx]
                    indiv_pcds = self.indiv_label_to_pcds.get(label, [])
                else:
                    indiv_pcds = self.indiv_pcds
            indiv_pcd = self.create_combined_pcd(indiv_pcds)
            combined += indiv_pcd
        
        # Full-scene overlay
        if self.scene_visible and self.scene_pcd is not None:
            combined += self.scene_pcd
        
        return combined
    
    def show_label(self, label: Optional[str] = None):
        """Show point clouds for a specific label or all labels."""
        if label is None:
            # Show all
            title = f"All Objects ({len(self.all_pcds)} objects)"
        else:
            # Show specific label
            if label not in self.label_to_pcds:
                print(f"  ⚠ Warning: Label '{label}' not found")
                return None
            
            label_pcds = self.label_to_pcds[label]
            total_points = sum(len(pcd.points) for pcd in label_pcds)
            title = f"Label: {label} ({len(label_pcds)} objects, {total_points:,} points)"
        
        # Rebuild view with current toggles (dedup/scene/individual)
        self.pcd_display = self.build_current_view()
        
        return title
    
    def navigate_next(self, vis):
        """Right arrow: Navigate to next label."""
        if len(self.label_list) == 0:
            return False
        
        self.current_label_idx = (self.current_label_idx + 1) % (len(self.label_list) + 1)
        
        if self.current_label_idx == len(self.label_list):
            # Wrap to "all"
            label = None
            title = self.show_label(None)
            print(f"\n  [NAVIGATE] Showing all objects")
        else:
            # Show specific label
            label = self.label_list[self.current_label_idx]
            title = self.show_label(label)
            print(f"\n  [NAVIGATE] {title} [{self.current_label_idx + 1}/{len(self.label_list)}]")
        
        # Update visualization
        vis.clear_geometries()
        vis.add_geometry(self.pcd_display, reset_bounding_box=False)
        vis.update_renderer()
        
        return False
    
    def navigate_prev(self, vis):
        """Left arrow: Navigate to previous label."""
        if len(self.label_list) == 0:
            return False
        
        if self.current_label_idx == -1:
            # Wrap from "all" to last label
            self.current_label_idx = len(self.label_list) - 1
        else:
            self.current_label_idx -= 1
        
        if self.current_label_idx < 0:
            # Wrap to "all"
            label = None
            title = self.show_label(None)
            print(f"\n  [NAVIGATE] Showing all objects")
        else:
            # Show specific label
            label = self.label_list[self.current_label_idx]
            title = self.show_label(label)
            print(f"\n  [NAVIGATE] {title} [{self.current_label_idx + 1}/{len(self.label_list)}]")
        
        # Update visualization
        vis.clear_geometries()
        vis.add_geometry(self.pcd_display, reset_bounding_box=False)
        vis.update_renderer()
        
        return False
    
    def show_all_objects(self, vis):
        """'A' key: Show all objects."""
        self.current_label_idx = -1
        title = self.show_label(None)
        print(f"\n  [ALL] Showing all objects")
        
        vis.clear_geometries()
        vis.add_geometry(self.pcd_display, reset_bounding_box=False)
        vis.update_renderer()
        
        return False

    def toggle_individual(self, vis):
        """'S' key: Toggle individual (pre-dedup) object point clouds on/off."""
        if not self.indiv_pcds:
            print("\n  ⚠ No individual object point clouds loaded; cannot toggle.")
            return False
        
        self.indiv_visible = not self.indiv_visible
        if self.indiv_visible:
            print("\n  [INDIV] Showing individual objects overlay")
        else:
            print("\n  [INDIV] Hiding individual objects overlay")
        
        self.pcd_display = self.build_current_view()
        vis.clear_geometries()
        vis.add_geometry(self.pcd_display, reset_bounding_box=False)
        vis.update_renderer()
        
        return False

    def toggle_deduplicated(self, vis):
        """'D' key: Toggle deduplicated objects on/off."""
        # If turning off dedup and there is no other layer, warn
        if self.dedup_visible and not self.scene_visible and not self.indiv_visible:
            print("\n  ⚠ Cannot hide deduplicated objects when no other layer is visible.")
            return False
        
        self.dedup_visible = not self.dedup_visible
        if self.dedup_visible:
            print("\n  [DEDUP] Showing deduplicated objects")
        else:
            print("\n  [DEDUP] Hiding deduplicated objects")
        
        self.pcd_display = self.build_current_view()
        vis.clear_geometries()
        vis.add_geometry(self.pcd_display, reset_bounding_box=False)
        vis.update_renderer()
        
        return False
    
    def reset_view(self, vis):
        """'R' key: Reset camera view."""
        print(f"\n  [RESET] Resetting view")
        vis.reset_view_point(True)
        return False
    
    def toggle_scene(self, vis):
        """'A' key: Toggle full-scene RGB point cloud on/off (overlaid with current objects)."""
        if self.scene_pcd is None:
            print("\n  ⚠ Full-scene point cloud not loaded; cannot toggle. (Set --scene or use default all_points.ply)")
            return False
        
        # Toggle state
        self.scene_visible = not self.scene_visible
        if self.scene_visible:
            print("\n  [SCENE] Showing full RGB scene overlaid with current view")
        else:
            print("\n  [SCENE] Hiding full RGB scene")
        
        # Rebuild combined view
        self.pcd_display = self.build_current_view()
        
        vis.clear_geometries()
        vis.add_geometry(self.pcd_display, reset_bounding_box=False)
        vis.update_renderer()
        
        return False
    
    def visualize(self):
        """Run interactive visualization."""
        # Initialize with all objects
        title = self.show_label(None)
        
        print(f"\n{'='*70}")
        print(f"  Launching Interactive Visualization")
        print(f"{'='*70}")
        print(f"\n  Keyboard Controls:")
        print(f"    - RIGHT Arrow (->):  Next label (forward)")
        print(f"    - LEFT Arrow  (<-):  Previous label (backward)")
        print(f"    - 'A':               Toggle full RGB scene (all_points.ply) on/off")
        print(f"    - 'S':               Toggle individual objects overlay (from individual_objects)")
        print(f"    - 'D':               Toggle deduplicated objects on/off")
        print(f"    - 'R':               Reset view")
        print(f"    - 'H':               Show all Open3D controls")
        print(f"    - 'Q' or ESC:        Quit")
        print(f"\n  Mouse Controls:")
        print(f"    - Left Button:       Rotate")
        print(f"    - Right Button:      Pan")
        print(f"    - Wheel:             Zoom")
        print(f"\n  Labels available: {len(self.label_list)}")
        print(f"  Current view: {title}")
        print(f"{'='*70}\n")
        
        # Create visualization with key callbacks
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name="Deduplicated Objects - Label Navigation",
            width=1280,
            height=720
        )
        
        # Register key callbacks
        self.vis.register_key_callback(262, self.navigate_next)   # RIGHT arrow
        self.vis.register_key_callback(263, self.navigate_prev)   # LEFT arrow
        self.vis.register_key_callback(65, self.toggle_scene)     # 'A' key: toggle full scene
        self.vis.register_key_callback(83, self.toggle_individual) # 'S' key: toggle individual overlay
        self.vis.register_key_callback(68, self.toggle_deduplicated) # 'D' key: toggle dedup layer
        self.vis.register_key_callback(82, self.reset_view)       # 'R' key
        
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


def visualize_deduplicated_objects(dedup_dir: Path, show_stats: bool = True, show_all: bool = False, 
                                   scene_path: Optional[Path] = None, scene_downsample: int = 1):
    """
    Load and visualize deduplicated object point clouds with label navigation.
    
    Args:
        dedup_dir: Path to directory containing deduplicated PLY files and summary JSON
        show_stats: Whether to print statistics about the objects
        show_all: If True, show all objects; if False, only show merged objects (num_merged >= 2)
        scene_path: Optional path to full-scene RGB point cloud (all_points.ply)
        scene_downsample: Downsampling factor for scene (e.g., 2 = keep every 2nd point). Default: 1 (no downsampling)
    """
    visualizer = DeduplicatedObjectsVisualizer(dedup_dir, show_stats, show_all, scene_path, scene_downsample)
    
    if not visualizer.load_data():
        return
    
    visualizer.visualize()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Interactive visualization for deduplicated objects"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Path to deduplicated objects directory (default: <project_root>/logs/current_run_outputs/offline_outputs/deduplicated_objects)"
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip printing statistics"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all objects (default: only show merged objects with num_merged >= 2)"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Path to full-scene RGB point cloud (default: <project_root>/tiamat_agent/vision_grounding/data/all_points.ply)"
    )
    parser.add_argument(
        "--scene-downsample",
        type=int,
        default=1,
        help="Downsampling factor for scene point cloud (e.g., 2 = keep every 2nd point, 3 = keep every 3rd point). Default: 1 (no downsampling). Higher values speed up scene rendering."
    )
    
    args = parser.parse_args()
    
    # Determine directory path
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    if args.dir:
        dedup_dir = Path(args.dir)
    else:
        # Default path
        dedup_dir = project_root / "logs" / "current_run_outputs" / "offline_outputs" / "deduplicated_objects"
    
    # Determine scene point cloud path
    if args.scene:
        scene_path = Path(args.scene)
    else:
        # Default: all_points.ply produced by object_detection_pipeline2.py
        scene_path = script_dir / "data" / "all_points.ply"
    
    # Validate downsample factor
    if args.scene_downsample < 1:
        print(f"  ✗ Error: Scene downsample factor must be >= 1, got {args.scene_downsample}")
        return
    
    # Visualize
    visualize_deduplicated_objects(dedup_dir, show_stats=not args.no_stats, show_all=args.all, 
                                   scene_path=scene_path, scene_downsample=args.scene_downsample)


if __name__ == "__main__":
    main()

