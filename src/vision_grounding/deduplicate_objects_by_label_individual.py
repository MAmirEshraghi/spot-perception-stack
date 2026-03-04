#!/usr/bin/env python3
"""
Deduplicate objects by label using point cloud overlap (ID-based matching).

Process:
1. Load individual PLY files from directory and object_list.json
2. Extract individual object point clouds (by ID matching - filename ↔ object_id)
3. Group objects by label
4. For each label, deduplicate objects with >90% overlap
5. Save merged point clouds

Usage:
    python deduplicate_objects_by_label_individual.py
    python deduplicate_objects_by_label_individual.py --dir path/to/individual_objects --json path/to/object_list.json
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


def load_individual_plys_and_json(ply_dir: Path, json_path: Path):
    """Load individual PLY files and JSON file."""
    print(f"\n{'='*70}")
    print(f"  Loading Data")
    print(f"{'='*70}")
    print(f"PLY directory: {ply_dir}")
    print(f"JSON file: {json_path}")
    
    if not ply_dir.exists():
        raise FileNotFoundError(f"PLY directory not found: {ply_dir}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    print(f"\n  Loading JSON...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    objects = data.get("objects", [])
    
    print(f"  ✓ Loaded {len(objects)} object records")
    
    # Build object_id -> object mapping based on frame_id and per-frame index
    print(f"\n  Building object_id to object mapping...")
    
    # Track how many objects we've seen per frame_id to reconstruct local indices
    per_frame_counts = defaultdict(int)
    object_id_to_obj = {}
    
    for obj in objects:
        frame_meta = obj.get("frame_metadata", {})
        frame_id = frame_meta.get("frame_id")  # e.g. "1765453374.821996404/head_rgb_right"
        
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
        
        object_id_to_obj[sanitized_object_id] = obj
    
    print(f"  ✓ Mapped {len(object_id_to_obj)} object IDs")
    
    # Load all PLY files from directory
    print(f"\n  Loading individual PLY files from directory...")
    ply_files = sorted(ply_dir.glob("*.ply"))
    
    if len(ply_files) == 0:
        raise ValueError(f"No PLY files found in {ply_dir}")
    
    print(f"  Found {len(ply_files)} PLY files")
    
    return ply_files, object_id_to_obj, objects


def extract_object_point_clouds(ply_files: List[Path], object_id_to_obj: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    """Extract point cloud for each object by ID matching (filename ↔ object_id)."""
    print(f"\n{'='*70}")
    print(f"  Extracting Object Point Clouds")
    print(f"{'='*70}")
    print(f"Extracting point clouds for {len(ply_files)} PLY files...")
    
    object_pcds = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
            
            points = np.asarray(pcd.points)
            
            # Check if object_id exists in JSON mapping
            if object_id in object_id_to_obj:
                obj = object_id_to_obj[object_id]
                object_pcds[object_id] = points
                label = obj.get("semantic_metadata", {}).get("label", "unknown")
                print(f"  {object_id}: {label} - {len(points):,} points")
                matched_count += 1
            else:
                # Object ID not found in JSON - skip
                print(f"    ⚠ Warning: {object_id} not found in JSON, skipping")
                unmatched_count += 1
            
            loaded_count += 1
            
        except Exception as e:
            print(f"    ✗ Error loading {ply_file.name}: {e}")
            unmatched_count += 1
            continue
    
    print(f"\n  ✓ Loaded {loaded_count} PLY files")
    print(f"  ✓ Matched {matched_count} objects to JSON records")
    if unmatched_count > 0:
        print(f"  ⚠ {unmatched_count} files could not be matched or loaded")
    
    print(f"  ✓ Extracted {len(object_pcds)} object point clouds")
    return object_pcds


def group_objects_by_label(objects: List[Dict], obj_to_object_id: Dict[int, str]) -> Tuple[Dict[str, List[Dict]], Dict[str, List[str]]]:
    """
    Group objects by label, also returning object_id for each object.
    Returns: (grouped_objects, grouped_object_ids)
    """
    grouped_objects = defaultdict(list)
    grouped_object_ids = defaultdict(list)
    
    for idx, obj in enumerate(objects):
        label = obj.get("semantic_metadata", {}).get("label", "unknown")
        grouped_objects[label].append(obj)
        if idx in obj_to_object_id:
            grouped_object_ids[label].append(obj_to_object_id[idx])
        else:
            grouped_object_ids[label].append(None)
    
    return dict(grouped_objects), dict(grouped_object_ids)


def build_object_id_to_obj_mapping(objects: List[Dict]) -> Dict[str, Dict]:
    """Build mapping from sanitized object_id to object record."""
    per_frame_counts = defaultdict(int)
    object_id_to_obj = {}
    
    for obj in objects:
        frame_meta = obj.get("frame_metadata", {})
        frame_id = frame_meta.get("frame_id")
        
        if frame_id is None:
            continue
        
        local_idx = per_frame_counts[frame_id]
        per_frame_counts[frame_id] += 1
        
        full_object_id = f"{frame_id}_bbox_object_{local_idx}"
        sanitized_object_id = (
            full_object_id
            .replace('/', '_')
            .replace(':', '_')
            .replace('\\', '_')
        )
        
        object_id_to_obj[sanitized_object_id] = obj
    
    return object_id_to_obj


def create_obj_to_object_id_mapping(objects: List[Dict]) -> Dict[int, str]:
    """
    Create mapping from object index in list to sanitized object_id.
    Uses the same logic as when building object_id_to_obj.
    """
    per_frame_counts = defaultdict(int)
    obj_to_object_id = {}
    
    for idx, obj in enumerate(objects):
        frame_meta = obj.get("frame_metadata", {})
        frame_id = frame_meta.get("frame_id")
        
        if frame_id is None:
            continue
        
        local_idx = per_frame_counts[frame_id]
        per_frame_counts[frame_id] += 1
        
        full_object_id = f"{frame_id}_bbox_object_{local_idx}"
        sanitized_object_id = (
            full_object_id
            .replace('/', '_')
            .replace(':', '_')
            .replace('\\', '_')
        )
        
        obj_to_object_id[idx] = sanitized_object_id
    
    return obj_to_object_id


def deduplicate_label_group(objects: List[Dict], object_ids: List[Optional[str]], 
                           object_pcds: Dict[str, np.ndarray],
                           threshold: float = 0.9, voxel_size: float = 0.1) -> List[Dict]:
    """
    Deduplicate objects within a label group using coverage calculation.
    
    Args:
        objects: List of object dictionaries
        object_ids: List of corresponding sanitized object_ids (same order as objects)
        object_pcds: Dictionary mapping object_id to point cloud array
    
    Returns list of merged objects (duplicates removed).
    """
    if len(objects) < 2:
        # Handle single object case
        if len(objects) == 1:
            obj = objects[0]
            obj_id = object_ids[0] if object_ids else None
            
            if obj_id and obj_id in object_pcds:
                merged_obj = obj.copy()
                merged_obj["merged_point_cloud"] = object_pcds[obj_id]
                merged_obj["merged_from"] = [obj_id]
                merged_obj["num_merged"] = 1
                return [merged_obj]
        # No valid point cloud for this label -> skip
        return []
    
    # Get point clouds for this label group
    pcd_list = []
    valid_objects = []
    valid_object_ids = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for obj, obj_id in zip(objects, object_ids):
        if obj_id and obj_id in object_pcds:
            pcd_array = object_pcds[obj_id]
            if len(pcd_array) > 0:
                # Convert to torch tensor and move to device
                pcd_tensor = torch.from_numpy(pcd_array).float().to(device)
                pcd_list.append(pcd_tensor)
                valid_objects.append(obj)
                valid_object_ids.append(obj_id)
    
    if len(pcd_list) < 2:
        # Handle single or no valid objects
        if len(pcd_list) == 1:
            merged_obj = valid_objects[0].copy()
            merged_obj["merged_point_cloud"] = pcd_list[0].cpu().numpy()
            merged_obj["merged_from"] = [valid_object_ids[0]]
            merged_obj["num_merged"] = 1
            return [merged_obj]
        return []
    
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
            merged_from_ids = [valid_object_ids[i]]
            
            for j in overlapping:
                merged_pcd = np.vstack([merged_pcd, pcd_list[j].cpu().numpy()])
                merged_indices.add(j)
                merged_from_ids.append(valid_object_ids[j])
            
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
            merged_obj["merged_from"] = [valid_object_ids[i]]
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
            
            # Assign color based on original object color (if available)
            color = obj.get("detection_metadata", {}).get("distinct_color_rgb", [128, 128, 128])
            if isinstance(color, list) and len(color) == 3:
                pcd.paint_uniform_color([c/255.0 for c in color])
            else:
                pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Default gray
            
            # Save
            filename = f"{safe_label}_{idx:03d}.ply"
            filepath = output_dir / filename
            o3d.io.write_point_cloud(str(filepath), pcd)
            
            summary["files"].append({
                "label": label,
                "filename": filename,
                "num_points": len(pcd_array),
                "num_merged": num_merged,
                "merged_from": obj.get("merged_from", [])
            })
            
            print(f"  Saved: {filename} ({len(pcd_array):,} points, merged from {num_merged} objects)")
    
    # Save summary
    summary_path = output_dir / "deduplication_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n  ✓ Saved {len(summary['files'])} merged objects to {output_dir}")
    print(f"  ✓ Summary saved to {summary_path}")
    
    # Print statistics
    if summary['statistics']['total_objects_before'] > 0:
        reduction_pct = 100 * (1 - summary['statistics']['total_objects_after'] / summary['statistics']['total_objects_before'])
        print(f"\n  Deduplication Statistics:")
        print(f"    Objects before: {summary['statistics']['total_objects_before']}")
        print(f"    Objects after:  {summary['statistics']['total_objects_after']}")
        print(f"    Reduction:      {summary['statistics']['total_objects_before'] - summary['statistics']['total_objects_after']} objects ({reduction_pct:.1f}%)")
        print(f"    Total merges:   {summary['statistics']['total_merges']}")
    else:
        print(f"\n  ⚠ No objects to deduplicate")


def main():
    """Main deduplication pipeline."""
    parser = argparse.ArgumentParser(
        description="Deduplicate objects by label using point cloud overlap (ID-based matching)"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Path to directory with individual PLY files"
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
    
    if args.dir:
        ply_dir = Path(args.dir)
    else:
        # Default path
        ply_dir = project_root / "logs" / "current_run_outputs" / "offline_outputs" / "individual_objects"
    
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
    print(f"#{'OBJECT DEDUPLICATION BY LABEL (ID-BASED)'.center(68)}#")
    print(f"{'#'*70}")
    print(f"\nConfiguration:")
    print(f"  Coverage threshold: {COVERAGE_THRESHOLD}")
    print(f"  Voxel size:         {VOXEL_SIZE}")
    print(f"  PLY directory:      {ply_dir}")
    print(f"  JSON file:          {json_path}")
    print(f"  Output directory:   {output_dir}")
    
    try:
        # Load data
        ply_files, object_id_to_obj, objects = load_individual_plys_and_json(ply_dir, json_path)
        
        # Extract individual object point clouds
        object_pcds = extract_object_point_clouds(ply_files, object_id_to_obj)
        
        if len(object_pcds) == 0:
            print("\n  ✗ Error: No object point clouds extracted. Check ID matching.")
            return
        
        # Build object index to object_id mapping
        obj_to_object_id = create_obj_to_object_id_mapping(objects)
        
        # Group by label (with object_ids)
        grouped_by_label, grouped_object_ids = group_objects_by_label(objects, obj_to_object_id)
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
            label_object_ids = grouped_object_ids[label]
            print(f"\nProcessing label: '{label}' ({len(label_objects)} objects)")
            merged = deduplicate_label_group(label_objects, label_object_ids, object_pcds,
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

