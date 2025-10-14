import open3d as o3d
import numpy as np
import torch
import pickle
import os
from collections import defaultdict

from src_perception.components.pcd_coverage import mega_optimized_batch_coverage
from src_perception.components.point_cloud import generate_distinct_colors as generate_distinct_colors_int


# CONFIGURATIONs
#==================================================================
DATA_PATH = "data/object_pcds.pkl"
LOG_DIR = "logs"
OUTPUT_LOG_PATH = os.path.join(LOG_DIR, "unique_objects_log.pkl")
VOXEL_SIZE = 0.1
COVERAGE_THRESHOLD = 0.25
#==================================================================


class InteractiveDeduplicator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # State variables
        self.unique_pcds = []
        colors_int = generate_distinct_colors_int(1000) # Generate 500 unique colors
        self.distinct_colors = [[c / 255.0 for c in color] for color in colors_int] # int(0-255) to floots (0.0-1.0) for Open3D
        self.unique_colors = [] # ?
        self.step_index = 0 # ?
        self.is_processing = False # ?
        self.state_history = []  # to store previous states

        # Load and prepare data
        self.step_data, self.step_keys = self._load_and_group_data()
        
        # Setup Visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("Interactive Deduplication Viewer | Press [->] to advance", 1280, 720)
        
        # KEY_RIGHT = 262  # ->
        self.vis.register_key_callback(262, self.process_next_step)
        # KEY_LEFT = 263  # <-
        self.vis.register_key_callback(263, self.process_previous_step) # Register the new backward function
        
        
        # Add a coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        self.vis.add_geometry(coord_frame)

        print_initial_instructions()



    def _load_and_group_data(self):
        """Loads the object PCDs and groups them by image."""
        print(f"\nLoading object data from {DATA_PATH}...")
        with open(DATA_PATH, 'rb') as f:
            all_objects = pickle.load(f)

        grouped_data = defaultdict(list)
        for obj in all_objects:
            key = (obj['frame_id'], obj['camera_id'])
            grouped_data[key].append(obj['obj_point_cloud'])
        
        # Sort by frame_id, then camera_id
        sorted_keys = sorted(grouped_data.keys())
        print(f"Loaded {len(all_objects)} objects from {len(sorted_keys)} images.")
        return grouped_data, sorted_keys

    def process_next_step(self, vis):
        """Core logic that runs on each key press."""
        
        # Use the lock to prevent the function from running twice
        if self.is_processing:
            return False
        self.is_processing = True
        
        if self.step_index >= len(self.step_keys):
            print("\n--- All images processed. No more steps. ---")
            self.is_processing = False # Release lock
            return False

        # Save the current state BEFORE processing the next step (history for left array key)
        current_state = {
            'pcds': [o3d.geometry.PointCloud(pcd) for pcd in self.unique_pcds],
            'colors': list(self.unique_colors), # Copy the list of colors
            'index': self.step_index
        }
        self.state_history.append(current_state)

        # 1. Get candidate objects for the current step
        current_key = self.step_keys[self.step_index]
        frame_id, camera_id = current_key
        candidate_pcds_np = self.step_data[current_key]
        
        print(f"\n{'='*60}\nSTEP {self.step_index + 1}/{len(self.step_keys)} | Frame: {frame_id}, Camera: {camera_id}\n{'='*60}")
        print(f"  > Found {len(candidate_pcds_np)} candidate objects in this image.")
        
        # 2. Check if this is the first step (initialization)
        if self.step_index == 0:
            print("  -> First image: Initializing scene. All objects will be added as unique.")
            for i, candidate_np in enumerate(candidate_pcds_np):
                # Directly add each candidate as a new unique object
                new_pcd = o3d.geometry.PointCloud()
                new_pcd.points = o3d.utility.Vector3dVector(candidate_np)
                self.unique_pcds.append(new_pcd)
                
                # Assign a new persistent color
                new_color = self.distinct_colors[len(self.unique_pcds) - 1]
                self.unique_colors.append(new_color)
                
                print(f"    - Candidate {i}: Added as new unique object {len(self.unique_pcds) - 1}")

        # 3. Perform deduplication for each candidate
        else:  
            # For all subsequent images, do the standard deduplication logic
            candidate_tensors = [torch.from_numpy(pcd).to(self.device).float() for pcd in candidate_pcds_np]
            unique_tensors = [torch.from_numpy(np.asarray(pcd.points)).to(self.device).float() for pcd in self.unique_pcds]
            
            
            for i, candidate_tensor in enumerate(candidate_tensors):
                # self.unique_pcds will always have items here because of the first step
                counts, coverages = mega_optimized_batch_coverage(candidate_tensor, unique_tensors, VOXEL_SIZE, return_counts=True)
                best_match_percentage = torch.max(coverages).item() if len(coverages) > 0 else 0.0
                best_match_count = torch.max(counts).item() if len(coverages) > 0 else 0.0

                if best_match_percentage > COVERAGE_THRESHOLD or best_match_count>5:
                    best_match_idx = torch.argmax(coverages).item()
                    
                    # Merge the point clouds
                    candidate_o3d = o3d.geometry.PointCloud()
                    candidate_o3d.points = o3d.utility.Vector3dVector(candidate_tensor.cpu().numpy())
                    self.unique_pcds[best_match_idx] += candidate_o3d  #merging by add the point to existing obj
                    
                    print(f"    - Candidate {i}: Matched and merged with unique object {best_match_idx} (Score: {best_match_percentage:.2f})")
                else:
                    # Add as a new unique object
                    new_pcd = o3d.geometry.PointCloud()
                    new_pcd.points = o3d.utility.Vector3dVector(candidate_tensor.cpu().numpy())
                    self.unique_pcds.append(new_pcd)
                    
                    new_color = self.distinct_colors[len(self.unique_pcds) - 1]
                    self.unique_colors.append(new_color)
                    
                    print(f"    - Candidate {i}: Added as new unique object {len(self.unique_pcds) - 1}")


        # 4. Update visualization and save state
        self._update_visualizer(vis)
        self._save_state()
        self.step_index += 1

        self.is_processing = False 

        return True
        
    def process_previous_step(self, vis):
        """Reverts the scene to the previously saved state."""
        
        if self.is_processing:
            return False
        self.is_processing = True
        
        # Check if there is any history to go back to
        if not self.state_history:
            print("\n--- At the beginning. Cannot go back further. ---")
            return False
            
        print(f"\n{'<--'*20}\nGoing back to the previous step...\n{'<--'*20}")
        
        # Pop the last state from the history
        last_state = self.state_history.pop()
        
        # Restore the state variables
        self.unique_pcds = last_state['pcds']
        self.unique_colors = last_state['colors']
        self.step_index = last_state['index']
        
        # Update the visualizer to show the restored scene
        self._update_visualizer(vis)
        
        # You can optionally re-save the log file here if you want it to reflect the backward step
        self._save_state() 
        self.is_processing = False

        return True

    def _update_visualizer(self, vis):
        """Clears and redraws all unique objects with their persistent colors."""
        
        vis.clear_geometries()
        
        # Add a coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(coord_frame, reset_bounding_box=False)

        for i, pcd in enumerate(self.unique_pcds):
            # Downsample for faster rendering
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
            pcd_downsampled.paint_uniform_color(self.unique_colors[i])
            vis.add_geometry(pcd_downsampled, reset_bounding_box=False)
        
        # reset for first time
        if self.step_index == 0:
            vis.reset_view_point(True)

        vis.poll_events()
        vis.update_renderer()
        
    def _save_state(self):
        """Saves the current list of unique PCDs and their metadata."""
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
            
        log_data = []
        for i, pcd in enumerate(self.unique_pcds):
            log_data.append({
                "unique_object_id": i,
                "color_rgb": self.unique_colors[i],
                "point_cloud": np.asarray(pcd.points)
            })
            
        with open(OUTPUT_LOG_PATH, 'wb') as f:
            pickle.dump(log_data, f)
        
        print(f"\n  > Report: Total unique objects = {len(self.unique_pcds)}. State saved to {OUTPUT_LOG_PATH}")

    def run(self):
        """Starts the visualization loop."""
        self.vis.run()
        self.vis.destroy_window()

def print_initial_instructions():
    print("\n" + "*"*60)
    print("      Interactive Deduplication Viewer Initialized ")
    print("*"*60)
    print("Controls:")
    print("  - Focus the Open3D window.")
    print("  - Press the [->] (Right Arrow) key to process the next image.")
    print("  - Press the [<-] (Left Arrow) key to show the previous image.")
    print("  - Pan/Rotate/Zoom with the mouse to inspect the scene.")
    print("  - Press [Q] to quit.")
    print("*"*60 + "\n")


if __name__ == "__main__":
    viewer = InteractiveDeduplicator()
    viewer.run()