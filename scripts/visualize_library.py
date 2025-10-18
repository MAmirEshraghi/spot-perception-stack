import open3d as o3d
import numpy as np
import os
import pickle

# --- Configuration ---
# Path to the original, un-merged data file, used for the 'C' key raw view.
RAW_DATA_PATH = "data/object_pcds.pkl"
# Path to the processed, unique object library file created by the previous script.
FILE_TO_VISUALIZE = "logs/object_library/unique_objects.pkl"

class PointCloudViewer:
    """
    A simple viewer to inspect saved point cloud objects one by one, with multiple
    background and viewing modes controlled by keyboard shortcuts.
    """

    def __init__(self, processed_file_path):
        """Initializes the viewer and all its state variables."""
        # --- Core Properties ---
        self.processed_file_path = processed_file_path # Path to the main data file
        self.objects = []                              # List to hold the loaded unique objects
        self.current_index = 0                         # Index of the unique object currently being viewed
        self.vis = None                                # The Open3D visualizer object

        # --- Scene Point Clouds (pre-built for performance) ---
        self.scene_pcd = None          # Gray background of all unique objects
        self.colorful_scene_pcd = None # Colorful background of all unique objects
        self.raw_scene_pcd = None      # Colorful scene of ALL original objects

        # --- State Management for Views ---
        # A single variable controls which background is active.
        self.background_mode = "NONE" 
        # Remembers the mode before 'C' was pressed, for toggle-back functionality.
        self.previous_background_mode = "NONE"
        
        # --- Load data and set up the window ---
        self._load_data()
        if self.objects: # Only set up the visualizer if data was loaded successfully
            self._setup_visualizer()

    def _load_data(self):
        """Loads both processed and raw data files to build all scene variants."""
        
        # --- Section 1: Load the PROCESSED unique objects from the library file ---
        if not os.path.exists(self.processed_file_path):
            print(f"❌ Error: Processed file not found at '{self.processed_file_path}'")
            return

        print(f"Loading processed library from '{self.processed_file_path}'...")
        try:
            # Attempt to load the file using pickle
            with open(self.processed_file_path, 'rb') as f:
                self.objects = pickle.load(f)
        except Exception as e:
            print(f"❌ Error: Failed to load processed file. Error: {e}")
            return
        
        # Validate that the loaded data is in the expected format
        if not isinstance(self.objects, list) or not all('point_cloud' in d for d in self.objects):
            print("❌ Error: Processed data is not in the expected format (a list of dicts).")
            self.objects = []
            return

        print(f"✅ Successfully loaded {len(self.objects)} unique objects.")
        # After loading, build the background scenes for the 'A' and 'B' views
        self._build_processed_scene_pcds()

        # --- Section 2: Load the RAW original objects for the 'C' key view ---
        if not os.path.exists(RAW_DATA_PATH):
            print(f"⚠️ Warning: Raw data file not found at '{RAW_DATA_PATH}'. 'C' key view disabled.")
            return
            
        print(f"Loading raw source data from '{RAW_DATA_PATH}'...")
        try:
            # Load the original, un-merged point clouds
            with open(RAW_DATA_PATH, 'rb') as f: raw_objects_data = pickle.load(f)
        except Exception as e:
            print(f"❌ Error loading raw data: {e}"); return

        # Build the single point cloud for the raw view
        print("Building raw scene point cloud...")
        all_raw_points, all_raw_colors = [], []
        # Generate random distinct colors for each original mask
        raw_colors = np.random.rand(len(raw_objects_data), 3) # Colors from 0.0-1.0

        for i, obj_data in enumerate(raw_objects_data):
            points, color = obj_data['obj_point_cloud'], raw_colors[i]
            all_raw_points.append(points)
            # Assign the same color to all points in this mask
            all_raw_colors.append(np.tile(color, (points.shape[0], 1)))
        
        # Create the final Open3D object for the raw scene
        self.raw_scene_pcd = o3d.geometry.PointCloud()
        self.raw_scene_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_raw_points))
        self.raw_scene_pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_raw_colors))
        print("  > Raw scene built successfully.")

    def _build_processed_scene_pcds(self):
        """Helper to build backgrounds from the processed unique objects."""
        print("Building processed scene backgrounds for 'A' and 'B' views...")
        all_points_list, all_colors_list = [], []
        # Iterate through the unique objects to gather their points and colors
        for obj in self.objects:
            points, color = obj['point_cloud'], obj.get('color_rgb', [0.5, 0.5, 0.5])
            all_points_list.append(points)
            all_colors_list.append(np.tile(color, (points.shape[0], 1)))

        # Combine all points into a single NumPy array
        all_points = np.vstack(all_points_list)

        # Create the gray background scene
        self.scene_pcd = o3d.geometry.PointCloud()
        self.scene_pcd.points = o3d.utility.Vector3dVector(all_points)
        self.scene_pcd.paint_uniform_color([0.8, 0.8, 0.8]) # Light gray

        # Create the colorful background scene
        self.colorful_scene_pcd = o3d.geometry.PointCloud()
        self.colorful_scene_pcd.points = o3d.utility.Vector3dVector(all_points)
        self.colorful_scene_pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors_list))
        print("  > Processed backgrounds built successfully.")

    def _setup_visualizer(self):
        """Initializes the Open3D visualizer and registers key callbacks."""
        # Create the main window
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("Object Library Viewer | Use Arrow Keys to Navigate", 1280, 720)

        # Register keyboard shortcuts to their corresponding handler functions
        self.vis.register_key_callback(262, self._next_object)      # Right arrow
        self.vis.register_key_callback(263, self._previous_object)  # Left arrow
        self.vis.register_key_callback(65, self._handle_a_key)       # 'A' key
        self.vis.register_key_callback(66, self._handle_b_key)       # 'B' key
        self.vis.register_key_callback(67, self._handle_c_key)       # 'C' key
        
        # Print instructions to the console for the user
        print("\n--- Controls ---")
        print("  [->] / [<-] : Cycle through unique objects.")
        print("  [A]          : Set background to GRAY UNIQUE objects (or turn off).")
        print("  [B]          : Set background to COLORFUL UNIQUE objects.")
        print("  [C]          : Set background to ALL RAW objects (press again to revert).")
        print("----------------\n")

    def _update_view(self):
        """
        Renders the scene based on the current `self.background_mode` state.
        This is the main drawing function called after any state change.
        """
        # Start fresh by clearing all geometries from the scene
        self.vis.clear_geometries()
        self.vis.get_render_option().point_size = 5.0

        # Always add a coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.vis.add_geometry(coord_frame, reset_bounding_box=False)

        # A single, clean block to decide which background to draw.
        if self.background_mode == "GRAY_UNIQUE":
            self.vis.add_geometry(self.scene_pcd, reset_bounding_box=False)
        elif self.background_mode == "COLORFUL_UNIQUE":
            self.vis.add_geometry(self.colorful_scene_pcd, reset_bounding_box=False)
        elif self.background_mode == "RAW_ALL":
            if self.raw_scene_pcd:
                self.vis.add_geometry(self.raw_scene_pcd, reset_bounding_box=False)
        
        # This part always runs, drawing the main focused object on top of any background.
        if not self.objects: return # Safety check
        current_obj = self.objects[self.current_index]
        obj_id, points, color = current_obj.get("unique_object_id", self.current_index), current_obj['point_cloud'], current_obj.get('color_rgb', [0.5, 0.5, 0.5])
        print(f"Displaying Unique Object ID: {obj_id} ({self.current_index + 1}/{len(self.objects)}) | Background: {self.background_mode}")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)
        self.vis.add_geometry(pcd, reset_bounding_box=False)

    def _handle_c_key(self, vis):
        """Handles the 'C' key, which toggles the raw data background on or off."""
        if not self.raw_scene_pcd:
            print("Raw scene data not loaded, cannot switch view.")
            return
        
        if self.background_mode == "RAW_ALL":
            # If C is already active, revert to the previously saved mode.
            self.background_mode = self.previous_background_mode
        else:
            # If C is not active, save the current mode and switch to C.
            self.previous_background_mode = self.background_mode
            self.background_mode = "RAW_ALL"
        self._update_view()

    def _handle_b_key(self, vis):
        """Handles the 'B' key. Sets background to colorful unique objects."""
        if self.background_mode == "COLORFUL_UNIQUE":
             # If already in B mode, toggle back to A mode as a sensible default.
            self.background_mode = "GRAY_UNIQUE"
        else:
            self.background_mode = "COLORFUL_UNIQUE"
        self._update_view()

    def _handle_a_key(self, vis):
        """Handles the 'A' key. Toggles the gray unique background on or off."""
        if self.background_mode == "GRAY_UNIQUE":
            # If already in A mode, turn the background off.
            self.background_mode = "NONE"
        else:
            self.background_mode = "GRAY_UNIQUE"
        self._update_view()

    def _next_object(self, vis):
        """Advances to the next object, regardless of background mode."""
        # Use modulo arithmetic to loop back to the start of the list.
        self.current_index = (self.current_index + 1) % len(self.objects)
        self._update_view()

    def _previous_object(self, vis):
        """Goes to the previous object, regardless of background mode."""
        # Use modulo arithmetic to loop back to the end of the list.
        self.current_index = (self.current_index - 1 + len(self.objects)) % len(self.objects)
        self._update_view()

    def run(self):
        """Starts the main visualization event loop."""
        # Safety check before starting
        if not self.objects or not self.vis:
            print("Nothing to visualize. Exiting.")
            return
        
        # Draw the initial view before the loop begins
        self._update_view()
        # Set the camera view once at the very beginning
        self.vis.reset_view_point(True)
        # Start the interactive event loop (this blocks until the window is closed)
        self.vis.run()
        # Clean up the window when the loop is exited
        self.vis.destroy_window()

if __name__ == "__main__":
    # This block runs when the script is executed directly from the command line.
    
    # Create an instance of our viewer class, passing the path to the data file.
    viewer = PointCloudViewer(FILE_TO_VISUALIZE)
    # Start the application.
    viewer.run()