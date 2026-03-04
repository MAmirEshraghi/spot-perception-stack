#!/usr/bin/env python3
"""
Deduplicate objects by label using point cloud overlap.

Process:
1. Load objects_segmented_pc.ply and object_list.json
2. Extract individual object point clouds (by color matching)
3. Group objects by label
4. For each label, deduplicate objects with >90% overlap
5. Save merged point clouds

Usage:
    python deduplicate_objects_by_label.py
    python deduplicate_objects_by_label.py --ply path/to/objects_segmented_pc.ply --json path/to/object_list.json
"""

import sys
from pathlib import Path

# Add project root to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import open3d as o3d
import numpy as np
import torch
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Import coverage functions
from src.vision_grounding.pcd_coverage import (
    mega_optimized_query_batch_coverage
)

# Configuration
COVERAGE_THRESHOLD = 0.9  # Merge if overlap > 90%
VOXEL_SIZE = 0.1  # For coverage calculation


def load_ply_and_json(ply_path: Path, json_path: Path):
    """Load PLY and JSON files."""
    print(f"\n{'='*70}")
    print(f"  Loading Data")
    print(f"{'='*70}")
    print(f"PLY file: {ply_path}")
    print(f"JSON file: {json_path}")
    
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    print(f"\n  Loading point cloud...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    if len(pcd.points) == 0:
        raise ValueError(f"Point cloud is empty")
    
    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    
    print(f"  ✓ Loaded {len(points):,} points")
    
    print(f"\n  Loading JSON...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    objects = data.get("objects", [])
    
    print(f"  ✓ Loaded {len(objects)} object records")
    
    return points, colors, objects


def extract_object_point_clouds(points: np.ndarray, colors: np.ndarray, 
                                objects: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract point cloud for each object by color matching."""
    print(f"\n{'='*70}")
    print(f"  Extracting Object Point Clouds")
    print(f"{'='*70}")
    print(f"Extracting point clouds for {len(objects)} objects...")
    
    object_pcds = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for obj in objects:
        obj_id = obj.get("object_id", "unknown")
        color_rgb = obj.get("detection_metadata", {}).get("distinct_color_rgb")
        
        if not color_rgb:
            continue
        
        # Match colors (exact match - no tolerance)
        color_key = np.array(color_rgb)
        color_mask = np.all(colors == color_key, axis=1)
        object_points = points[color_mask]
        
        if len(object_points) > 0:
            object_pcds[obj_id] = object_points
            label = obj.get("semantic_metadata", {}).get("label", "unknown")
            print(f"  {obj_id}: {label} - {len(object_points):,} points")
    
    print(f"\n  ✓ Extracted {len(object_pcds)} object point clouds")
    return object_pcds


def group_objects_by_label(objects: List[Dict]) -> Dict[str, List[Dict]]:
    """Group objects by label."""
    grouped = defaultdict(list)
    for obj in objects:
        label = obj.get("semantic_metadata", {}).get("label", "unknown")
        grouped[label].append(obj)
    return dict(grouped)


def deduplicate_label_group(objects: List[Dict], object_pcds: Dict[str, np.ndarray],
                           threshold: float = 0.9, voxel_size: float = 0.1) -> List[Dict]:
    """
    Deduplicate objects within a label group using coverage calculation.
    
    Returns list of merged objects (duplicates removed).
    """
    if len(objects) < 2:
        # Handle single object case
        if len(objects) == 1:
            obj = objects[0]
            obj_id = obj.get("object_id")
            if obj_id in object_pcds:
                merged_obj = obj.copy()
                merged_obj["merged_point_cloud"] = object_pcds[obj_id]
                merged_obj["merged_from"] = [obj_id]
                merged_obj["num_merged"] = 1
                return [merged_obj]
        return objects
    
    # Get point clouds for this label group
    pcd_list = []
    valid_objects = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for obj in objects:
        obj_id = obj.get("object_id")
        if obj_id in object_pcds:
            pcd_array = object_pcds[obj_id]
            if len(pcd_array) > 0:
                # Convert to torch tensor and move to device
                pcd_tensor = torch.from_numpy(pcd_array).float().to(device)
                pcd_list.append(pcd_tensor)
                valid_objects.append(obj)
    
    if len(pcd_list) < 2:
        return valid_objects
    
    print(f"    Computing coverage matrix for {len(pcd_list)} objects...")
    
    # Compute all-vs-all coverage
    coverages = mega_optimized_query_batch_coverage(
        pcd_list, pcd_list, voxel_size=voxel_size
    )
    
    # Set diagonal to 0 (self-comparison)
    coverages.fill_diagonal_(0)
    
    # Find pairs to merge (coverage > threshold)
    merged_indices = set()
    merged_objects = []
    
    for i in range(len(valid_objects)):
        if i in merged_indices:
            continue
        
        # Find all objects that overlap with this one
        overlapping = [j for j in range(len(valid_objects)) 
                      if coverages[i, j].item() > threshold and j not in merged_indices]
        
        if overlapping:
            # Merge: combine point clouds
            merged_pcd = pcd_list[i].cpu().numpy()
            merged_from_ids = [valid_objects[i]["object_id"]]
            
            for j in overlapping:
                merged_pcd = np.vstack([merged_pcd, pcd_list[j].cpu().numpy()])
                merged_indices.add(j)
                merged_from_ids.append(valid_objects[j]["object_id"])
            
            # Create merged object (use first object as base)
            merged_obj = valid_objects[i].copy()
            merged_obj["merged_from"] = merged_from_ids
            merged_obj["merged_point_cloud"] = merged_pcd
            merged_obj["num_merged"] = len(merged_from_ids)
            merged_objects.append(merged_obj)
            
            print(f"      Merged {len(merged_from_ids)} objects: {merged_from_ids[0]} + {len(merged_from_ids)-1} others")
        else:
            # No overlap, keep as-is
            merged_obj = valid_objects[i].copy()
            merged_obj["merged_point_cloud"] = pcd_list[i].cpu().numpy()
            merged_obj["merged_from"] = [valid_objects[i]["object_id"]]
            merged_obj["num_merged"] = 1
            merged_objects.append(merged_obj)
    
    print(f"    Merged {len(valid_objects)} → {len(merged_objects)} unique objects")
    return merged_objects


def save_deduplicated_objects(merged_by_label: Dict[str, List[Dict]], output_dir: Path):
    """Save merged point clouds as PLY files."""
    print(f"\n{'='*70}")
    print(f"  Saving Deduplicated Objects")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "total_labels": len(merged_by_label),
        "objects_by_label": {},
        "files": [],
        "statistics": {
            "total_objects_before": 0,
            "total_objects_after": 0,
            "total_merges": 0
        }
    }
    
    for label, objects in merged_by_label.items():
        label_count = len(objects)
        summary["objects_by_label"][label] = label_count
        summary["statistics"]["total_objects_after"] += label_count
        
        # Sanitize label for filename (remove special characters)
        safe_label = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in label)
        
        for idx, obj in enumerate(objects):
            pcd_array = obj["merged_point_cloud"]
            num_merged = obj.get("num_merged", 1)
            
            # Count objects before (from merged_from list)
            summary["statistics"]["total_objects_before"] += num_merged
            if num_merged > 1:
                summary["statistics"]["total_merges"] += (num_merged - 1)
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_array)
            
            # Assign color based on original object color
            color = obj.get("detection_metadata", {}).get("distinct_color_rgb", [128, 128, 128])
            pcd.paint_uniform_color([c/255.0 for c in color])
            
            # Save
            filename = f"{safe_label}_{idx:03d}.ply"
            filepath = output_dir / filename
            o3d.io.write_point_cloud(str(filepath), pcd)
            
            summary["files"].append({
                "label": label,
                "filename": filename,
                "num_points": len(pcd_array),
                "num_merged": num_merged,
                "merged_from": obj.get("merged_from", [obj.get("object_id")])
            })
            
            print(f"  Saved: {filename} ({len(pcd_array):,} points, merged from {num_merged} objects)")
    
    # Save summary
    summary_path = output_dir / "deduplication_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n  ✓ Saved {len(summary['files'])} merged objects to {output_dir}")
    print(f"  ✓ Summary saved to {summary_path}")
    
    # Print statistics
    print(f"\n  Deduplication Statistics:")
    print(f"    Objects before: {summary['statistics']['total_objects_before']}")
    print(f"    Objects after:  {summary['statistics']['total_objects_after']}")
    print(f"    Reduction:      {summary['statistics']['total_objects_before'] - summary['statistics']['total_objects_after']} objects ({100*(1 - summary['statistics']['total_objects_after']/summary['statistics']['total_objects_before']):.1f}%)")
    print(f"    Total merges:   {summary['statistics']['total_merges']}")


def main():
    """Main deduplication pipeline."""
    parser = argparse.ArgumentParser(
        description="Deduplicate objects by label using point cloud overlap"
    )
    parser.add_argument(
        "--ply",
        type=str,
        default=None,
        help="Path to objects_segmented_pc.ply file"
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to object_list.json file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for deduplicated objects"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Coverage threshold for merging (default: 0.9)"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.1,
        help="Voxel size for coverage calculation (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Determine file paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    if args.ply:
        ply_path = Path(args.ply)
    else:
        # Default path
        ply_path = project_root / "logs" / "current_run_outputs" / "offline_outputs" / "objects_segmented_pc.ply"
    
    if args.json:
        json_path = Path(args.json)
    else:
        # Default path
        json_path = project_root / "logs" / "current_run_outputs" / "offline_outputs" / "object_list.json"
    
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default output path
        output_dir = project_root / "logs" / "current_run_outputs" / "offline_outputs" / "deduplicated_objects"
    
    # Update global config from args
    global COVERAGE_THRESHOLD, VOXEL_SIZE
    COVERAGE_THRESHOLD = args.threshold
    VOXEL_SIZE = args.voxel_size
    
    print(f"\n{'#'*70}")
    print(f"#{'OBJECT DEDUPLICATION BY LABEL'.center(68)}#")
    print(f"{'#'*70}")
    print(f"\nConfiguration:")
    print(f"  Coverage threshold: {COVERAGE_THRESHOLD}")
    print(f"  Voxel size:         {VOXEL_SIZE}")
    print(f"  Output directory:   {output_dir}")
    
    try:
        # Load data
        points, colors, objects = load_ply_and_json(ply_path, json_path)
        
        # Extract individual object point clouds
        object_pcds = extract_object_point_clouds(points, colors, objects)
        
        if len(object_pcds) == 0:
            print("\n  ✗ Error: No object point clouds extracted. Check color matching.")
            return
        
        # Group by label
        grouped_by_label = group_objects_by_label(objects)
        print(f"\n{'='*70}")
        print(f"  Grouping by Label")
        print(f"{'='*70}")
        print(f"Grouped into {len(grouped_by_label)} labels:")
        for label, label_objects in sorted(grouped_by_label.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {label:30s}: {len(label_objects):4d} objects")
        
        # Deduplicate each label group
        print(f"\n{'='*70}")
        print(f"  Deduplicating by Label")
        print(f"{'='*70}")
        merged_by_label = {}
        
        for label, label_objects in sorted(grouped_by_label.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"\nProcessing label: '{label}' ({len(label_objects)} objects)")
            merged = deduplicate_label_group(label_objects, object_pcds, 
                                            threshold=COVERAGE_THRESHOLD,
                                            voxel_size=VOXEL_SIZE)
            merged_by_label[label] = merged
        
        # Save results
        save_deduplicated_objects(merged_by_label, output_dir)
        
        print(f"\n{'='*70}")
        print(f"✨ Deduplication Complete!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

