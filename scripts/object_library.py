import torch
import numpy as np
import os
from collections import defaultdict
#import compactdata as cd 
import pickle
import time 

from src_perception.components.pcd_coverage import mega_optimized_query_batch_coverage
from src_perception.components.point_cloud import generate_distinct_colors as generate_distinct_colors_int

# ===============================================
# CONFIGURATION
VOXEL_SIZE = 0.1
VOXEL_SIZE_2= 0.05
COVERAGE_THRESHOLD = 0.2
SELF_DEDUP_THRESHOLD = 0.3
DATA_PATH = "data/object_pcds.pkl"
OUTPUT_PATH = "logs/object_library/unique_objects.pkl"
# ===============================================

class ObjectLibrary:
    """
    Manages a library of unique 3D objects using efficient tensor operations.
    This version is optimized to use many-vs-many batch coverage calculations.
    """
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # All points are stored in large tensors
        self.unique_object_points = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.unique_object_ids = torch.empty((0,), dtype=torch.long, device=self.device)

        self.total_merge_events = 0
        #Statistics Tracking  
        self.total_merge_events = 0

        # Helper for managing object metadata
        self.next_obj_id = 0
        self._active_ids = set()
        self._id_to_color_map = {}
        
        # Pre-generate a pool of distinct colors
        colors_int = generate_distinct_colors_int(1000)
        self._color_pool = [[c / 255.0 for c in color] for color in colors_int]
        
        print(f"ObjectLibrary initialized on device: {self.device}")

    def get_obj_pc(self, obj_id):
        """Returns the point cloud for a specific object ID."""
        if obj_id not in self._active_ids:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)
        
        mask = (self.unique_object_ids == obj_id)
        return self.unique_object_points[mask]

    def add_objects(self, candidate_pcds: list):
        """
        Adds a batch of new candidate point clouds to the library, performing
        a single, batched many-vs-many deduplication check.
        """
        if not candidate_pcds:
            return

        candidate_tensors = [torch.from_numpy(pcd).to(self.device).float() for pcd in candidate_pcds]

        if self.next_obj_id == 0:
            print("Library is empty. Adding all candidates as new objects.")
            for tensor in candidate_tensors:
                self._add_new_object(tensor)
            return

        # Prepare the list of existing unique objects for comparison
        active_id_list = sorted(list(self._active_ids))
        unique_tensors = [self.get_obj_pc(obj_id) for obj_id in active_id_list]
        
        # Perform a single many-vs-many coverage calculation.
        # returns a matrix where coverages[i, j] is the coverage of candidate[i] by unique_object[j].
        #coverages_matrix = mega_optimized_query_batch_coverage(candidate_tensors, unique_tensors, VOXEL_SIZE)
        counts_matrix, coverages_matrix = mega_optimized_query_batch_coverage(
            candidate_tensors, unique_tensors, VOXEL_SIZE, return_counts=True)
        # Find the best match for each candidate from the results matrix
        best_match_scores, best_match_indices = torch.max(coverages_matrix, dim=1)

        # Use the indices of the best scores to get the corresponding counts
        best_match_counts = torch.gather(counts_matrix, 1, best_match_indices.unsqueeze(1)).squeeze(1)
        
        # Process each candidate based on the batched results
        for i, candidate_tensor in enumerate(candidate_tensors):
            score = best_match_scores[i].item()
            count = best_match_counts[i].item()

            if score > COVERAGE_THRESHOLD: #or count > 5:
                best_unique_idx = best_match_indices[i].item()
                target_obj_id = active_id_list[best_unique_idx]
                self._merge_object(candidate_tensor, target_obj_id)
                self.total_merge_events += 1
                #print(f"  - Candidate {i}: Matched and merged with unique object {target_obj_id} (Score: {score:.2f})")
            else:
                new_id = self._add_new_object(candidate_tensor)
                #print(f"  - Candidate {i}: Added as new unique object {new_id} (Best score: {score:.2f})")

    def self_deduplication(self, iterations=3):
        """
        Post-processes the library to merge similar objects using an all-vs-all comparison.
        """
        print(f"\n{'~'*60}\nRunning self-deduplication pass...\n{'~'*60}")
        for iteration in range(iterations):
            merged_any = False
            active_ids = sorted(list(self._active_ids))
            
            if len(active_ids) < 2:
                print("  Not enough objects to compare. Stopping.")
                break

            # === EFFICIENCY UPGRADE ===
            # Prepare all unique pcds and run a single all-vs-all comparison.
            all_unique_pcds = [self.get_obj_pc(obj_id) for obj_id in active_ids]
            coverages_matrix = mega_optimized_query_batch_coverage(all_unique_pcds, all_unique_pcds, VOXEL_SIZE_2)
            
            # Avoid self-comparison by setting the diagonal to 0
            coverages_matrix.fill_diagonal_(0)

            # Find all pairs (A, B) where A is well-covered by B
            matches = torch.where(coverages_matrix > SELF_DEDUP_THRESHOLD)
            merged_in_this_pass = set()

            for query_idx, ref_idx in zip(*matches):
                obj_id_query = active_ids[query_idx]
                obj_id_ref = active_ids[ref_idx]

                if obj_id_query in merged_in_this_pass or obj_id_ref in merged_in_this_pass:
                    continue

                # Merge the query object (which is covered) into the reference object
                self.unique_object_ids[self.unique_object_ids == obj_id_query] = obj_id_ref
                self._active_ids.remove(obj_id_query)
                merged_in_this_pass.add(obj_id_query)
                
                score = coverages_matrix[query_idx, ref_idx].item()
                print(f"  ✓ Merged object {obj_id_query} into {obj_id_ref} (Score: {score:.2f})")
                merged_any = True

            if not merged_any:
                print(f"  No merges in iteration {iteration + 1}. Stopping early.")
                break
        
        # Final cleanup of the main tensor to remove points from merged objects
        if merged_any:
            active_ids_mask = torch.isin(self.unique_object_ids, torch.tensor(list(self._active_ids), device=self.device))
            self.unique_object_points = self.unique_object_points[active_ids_mask]
            self.unique_object_ids = self.unique_object_ids[active_ids_mask]

        print(f"  Final count: {len(self._active_ids)} unique objects")

    def save(self, file_path):
        """Saves the unique objects library to a file using compactdata."""
        print(f"\nSaving object library to {file_path}...")
        log_data = []
        for obj_id in sorted(list(self._active_ids)):
            points = self.get_obj_pc(obj_id).cpu().numpy()
            if points.shape[0] > 0: # Only save objects that still have points
                log_data.append({
                    "unique_object_id": obj_id,
                    "color_rgb": self._id_to_color_map[obj_id],
                    "point_cloud": points # numpy to list for saving by compacrdata
                })
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            #cd.dump(log_data, f)
            pickle.dump(log_data, f)
        print(f"  > Saved {len(log_data)} unique objects successfully.")
        
    def get_stats2(self, total_candidates=None):
        """Prints a final summary report about the library."""
        
        print_config_report()
        
        print("\n" + "="*60)
        print(" " * 20 + "FINAL SUMMARY REPORT")
        print("="*60)
        
        num_unique_objects = len(self._active_ids)
        total_points = self.unique_object_points.shape[0]

        if total_candidates is not None:
            print(f"Total Object Masks Processed : {total_candidates}")
            print(f"Final Unique Objects Found    : {num_unique_objects}")
            print(f"Total Merge Events          : {self.total_merge_events}")
            if total_candidates > 0:
                uniqueness_ratio = (num_unique_objects / total_candidates) * 100
                print(f"Uniqueness Ratio              : {uniqueness_ratio:.2f}%")
        else:
            print(f"Final Unique Objects Found    : {num_unique_objects}")

        print("-" * 60)
        
        # --- NEW: Object Size Distribution Statistics ---
        if num_unique_objects > 0:
            object_sizes = [self.get_obj_pc(obj_id).shape[0] for obj_id in self._active_ids]
            avg_points = sum(object_sizes) // num_unique_objects
            min_points = min(object_sizes)
            max_points = max(object_sizes)
            # Count how many objects are just tiny fragments (e.g., < 50 points)
            small_fragments = sum(1 for s in object_sizes if s < 50)
            
            print(f"Total Points in Library       : {total_points}")
            print(f"Average Points per Object     : {avg_points}")
            print(f"Smallest / Largest Object     : {min_points} / {max_points} points")
            print(f"Number of Small Fragments (<50 pts): {small_fragments} (Lower is better)")
            
        print("="*60)

    def get_stats(self, total_candidates=None, total_images=None, total_frames=None, timings=None):
        """Prints a final, detailed summary report for analysis."""
        
        print_config_report()
        print("\n" + "="*60)
        print(" " * 20 + "FINAL SUMMARY REPORT")
        print("="*60)
    
        # --- 3.2: Frame and Image Statistics ---
        if total_images and total_frames and total_candidates:
            print("--- Input Data ---")
            print(f"Total Images (steps)          : {total_images}")
            print(f"Total Frames                  : {total_frames}")
            print(f"Total Object Masks Processed    : {total_candidates}")
            print(f"Average Masks per Image       : {total_candidates / total_images:.2f}")
            print(f"Average Masks per Frame         : {total_candidates / total_frames:.2f}")
            print("-" * 60)

        # --- Deduplication Results ---
        print("--- Deduplication Results ---")
        num_unique_objects = len(self._active_ids)
        print(f"Final Unique Objects Found      : {num_unique_objects}")
        if total_candidates is not None:
            print(f"Total Merge Events            : {self.total_merge_events}")
            if total_candidates > 0:
                uniqueness_ratio = (num_unique_objects / total_candidates) * 100
                print(f"Uniqueness Ratio                : {uniqueness_ratio:.2f}% (Lower better: (num_unique_objects/total_candidates))")
        print("-" * 60)

        # --- 3.1: Detailed Fragment Size Distribution ---
        print("--- Object Size Distribution ---")
        if num_unique_objects > 0:
            object_sizes = [self.get_obj_pc(obj_id).shape[0] for obj_id in self._active_ids]
            total_points = sum(object_sizes)
            fragments_under_50 = sum(1 for s in object_sizes if s < 50)
            fragments_50_100 = sum(1 for s in object_sizes if 50 <= s < 100)
            fragments_100_200 = sum(1 for s in object_sizes if 100 <= s < 200)
            fragments_200_300 = sum(1 for s in object_sizes if 200 <= s < 300)
            
            print(f"Total Points in Library         : {total_points}")
            print(f"Average Points per Object       : {total_points // num_unique_objects}")
            print(f"Fragments (<50 pts)             : {fragments_under_50}")
            print(f"Fragments (50-100 pts)          : {fragments_50_100}")
            print(f"Fragments (100-200 pts)         : {fragments_100_200}")
            print(f"Fragments (200-300 pts)         : {fragments_200_300}")
        else:
            print("No unique objects found.")
        print("-" * 60)

        # --- 3.4: Performance and Timing Analysis ---
        if timings:
            print("--- Performance Analysis ---")
            total_time = sum(timings)
            avg_time = total_time / len(timings)
            print(f"Total Deduplication Time        : {total_time:.4f} seconds")
            print(f"Average Time per Image          : {avg_time:.4f} seconds")

        print("="*60)

    def visualize(self):
        """(Placeholder) For future visualization logic."""
        print("Visualization function is not implemented yet.")

    # --- Helper Methods ---
    def _add_new_object(self, points_tensor):
        new_id = self.next_obj_id
        ids_tensor = torch.full((points_tensor.shape[0],), new_id, device=self.device, dtype=torch.long)
        self.unique_object_points = torch.cat([self.unique_object_points, points_tensor], dim=0)
        self.unique_object_ids = torch.cat([self.unique_object_ids, ids_tensor], dim=0)
        
        self._id_to_color_map[new_id] = self._color_pool[new_id % len(self._color_pool)]
        self._active_ids.add(new_id)
        self.next_obj_id += 1
        return new_id

    def _merge_object(self, points_tensor, target_obj_id):
        ids_tensor = torch.full((points_tensor.shape[0],), target_obj_id, device=self.device, dtype=torch.long)
        self.unique_object_points = torch.cat([self.unique_object_points, points_tensor], dim=0)
        self.unique_object_ids = torch.cat([self.unique_object_ids, ids_tensor], dim=0)


def load_and_group_data(data_path):
    """Loads and groups point cloud data by frame/camera."""
    print(f"Loading object data from {data_path}...")
    try:
        with open(data_path, 'rb') as f:
            all_objects = cd.load(f)
    except Exception: # Fallback to pickle for original file
        import pickle
        with open(data_path, 'rb') as f:
            all_objects = pickle.load(f)

    grouped_data = defaultdict(list)
    for obj in all_objects:
        key = (obj['frame_id'], obj['camera_id'])
        grouped_data[key].append(obj['obj_point_cloud'])
    
    sorted_keys = sorted(grouped_data.keys())
    print(f"Loaded {len(all_objects)} objects from {len(sorted_keys)} images.")
    return grouped_data, sorted_keys

def print_config_report():
    """Prints a report of the key configuration parameters."""
    print("\n" + "*"*60)
    print(" " * 19 + "CONFIGURATION REPORT")
    print("*"*60)
    print(f"Data Source               : {DATA_PATH}")
    print(f"Voxel Size                : {VOXEL_SIZE}")
    print(f"Initial Coverage Threshold: {COVERAGE_THRESHOLD}")
    print(f"Self-Dedup Threshold      : {SELF_DEDUP_THRESHOLD}")
    print(f"Output File               : {OUTPUT_PATH}")
    print("*"*60)

if __name__ == "__main__":
    
    print_config_report()
    library = ObjectLibrary()
    
    step_data, step_keys = load_and_group_data(DATA_PATH)
    total_candidates_processed = 0
    deduplication_timings = [] # For timing analysis

    # 3.2: Calculate frame/image stats before the loop
    total_images = len(step_keys)
    total_frames = len(set(key[0] for key in step_keys)) if step_keys else 0


    for i, key in enumerate(step_keys):
        frame_id, camera_id = key
        candidate_pcds = step_data[key]
        total_candidates_processed += len(candidate_pcds)
        print(f"\n--- Processing Step {i + 1}/{len(step_keys)} | Frame: {frame_id} ---")
        
        # 3.4: Time the deduplication step 
        start_time = time.time()
        library.add_objects(candidate_pcds)
        end_time = time.time()
        deduplication_timings.append(end_time - start_time)

    # Pass all new stats to the report 
    library.get_stats(
        total_candidates=total_candidates_processed,
        total_images=total_images,
        total_frames=total_frames,
        timings=deduplication_timings
    )

    #library.self_deduplication()
    
    library.save(OUTPUT_PATH)