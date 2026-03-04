#!/usr/bin/env python3
"""Visualize all RGB and depth images from an ObsDataEntry in a collage format."""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import math
from typing import Tuple, Optional
from tiamat_agent.vision_grounding.obs_data_buffer import ObsDataEntry
import matplotlib.colors as mcolors



# Visualize Occupancy grid. 
def overlay_robot_position(ax, world_map, robot_world_coords, robot_yaw, force_color = None, robot_label = "Current Robot Pose", use_optimistic = False):
    is_collision, mask_info = world_map.is_pose_collision(x = robot_world_coords[0], y = robot_world_coords[1], yaw = robot_yaw, use_optimistic=use_optimistic, return_info=True)
    if mask_info is not None:
        world_map.occupancy_overlay(*mask_info)  # Overlay the robot's collision mask on the occupancy grid.

    agent_color = force_color if force_color is not None else ('green' if is_collision else 'red')

    # Add Robot Center Dot
    ax.scatter(robot_world_coords[0], robot_world_coords[1], color=agent_color, s=25)

    # Add robot yaw arrow
    arrow_length = 0.5
    dx, dy = arrow_length * np.cos(robot_yaw), arrow_length * np.sin(robot_yaw)
    ax.arrow(robot_world_coords[0], robot_world_coords[1], dx, dy, 
            head_width=0.1, head_length=0.1, fc=agent_color, ec=agent_color)
    # Add robot position text
    ax.annotate(text=f"{robot_label}: \n (x,y)=({robot_world_coords[0]:.2f},\
        {robot_world_coords[1]:.2f})\n Yaw: {np.degrees(robot_yaw):.1f}° \n Collision: {is_collision}",
            xy=(robot_world_coords[0], robot_world_coords[1]),
            color=agent_color,
            fontsize=12,
            xytext=(3, 3), textcoords='offset points',
            ha='left', va='bottom'
        )
    return ax

def plot_goal_pose_arrow(ax, world_map, goal_pose, force_color=None, goal_label="Goal Pose", use_optimistic=False, text_annotation=True):
    """
    Overlay a goal pose on the plot with symbol and arrow showing yaw.
    Also overlays the robot's collision mask at that pose.
    
    Args:
        ax: matplotlib axis
        world_map: WorldMap object
        goal_pose: tuple (x, y, yaw) in world coordinates
        force_color: color to use (default: green if collision-free, red if collision)
        goal_label: label for the goal
    
    Returns:
        ax: matplotlib axis
    """
    if goal_pose is None:
        return ax
    
    goal_x, goal_y, goal_yaw = goal_pose
    
    # # Check collision and get mask info
    is_collision, mask_info = world_map.is_pose_collision(
        x=goal_x, y=goal_y, yaw=goal_yaw, 
        use_optimistic=use_optimistic, return_info=True
    )
    # world_map.occupancy_overlay(*mask_info)  # Overlay the robot's collision mask
    
    # Choose color based on collision status
    goal_color = force_color if force_color is not None else ('red' if is_collision else 'lime')
    
    # Plot goal position with a star marker
    ax.scatter(goal_x, goal_y, c=goal_color, s=300, marker='*',
              edgecolors='black', linewidth=2, zorder=10, label=goal_label)
    
    # Draw arrow showing yaw direction
    arrow_length = 0.6
    dx = arrow_length * np.cos(goal_yaw)
    dy = arrow_length * np.sin(goal_yaw)
    ax.arrow(goal_x, goal_y, dx, dy,
            head_width=0.2, head_length=0.15, 
            fc=goal_color, ec='black', linewidth=2, zorder=11)
    
    # Add text annotation
    if text_annotation:
        collision_status = "COLLISION!" if is_collision else "Collision-Free"
        ax.annotate(
            text=f"{goal_label}:\n(x,y)=({goal_x:.2f},{goal_y:.2f})\nYaw: {np.degrees(goal_yaw):.1f}°\n{collision_status}",
            xy=(goal_x, goal_y),
            color=goal_color,
            fontsize=10,
            xytext=(5, 5), textcoords='offset points',
            ha='left', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    return ax

def plot_occupancy_map(ax, world_map):
    # Plot the occupancy grid. ############################################################################################
    occgrid = world_map.occupancy_map
    # Updated colormap to include unknown regions
    # Values: -1=unknown, 0=free, 1=occupied
    cmap = mcolors.ListedColormap(["gray", "white", "black"])  # unknown, free, occupied
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    
    # Adjust extent for rotation
    # Originally: x = world_map.x_coords, y = world_map.y_coords
    # After 90 deg CCW: x -> reversed y, y -> x
    extent = [world_map.x_coords[0], world_map.x_coords[-1],    # left, right
              world_map.y_coords[-1], world_map.y_coords[0]]    # bottom, top
    
    ax.imshow(
        occgrid,
        cmap=cmap, norm=norm, origin='upper',
        extent=extent, interpolation='nearest', alpha=0.7
    )
    ax.set_title("Occupancy Grid (Grey=Unknown, White=Free, Black=Occupied) with Frontiers")
    ax.set_xlabel("x (world coords)")  
    ax.set_ylabel("y (world coords)") 

    # Add a test point at the center to verify coordinate system
    center_x = (world_map.x_coords[0] + world_map.x_coords[-1]) / 2
    center_y = (world_map.y_coords[0] + world_map.y_coords[-1]) / 2
    ax.scatter(x=[center_x], y=[center_y], c='red', s=100, marker='x', label='Map Center')
    ############################################################################################################################
    return ax

def plot_height_map(ax, world_map):
    heightmap = world_map.heightmap

    min_height = np.min(heightmap[heightmap > -100000]) # -100000 is a placeholder for -inf
    max_height = np.max(heightmap[heightmap < 100000])  # Get max height for legend
    heightmap[heightmap<min_height] = min_height - 1  
    extent = [world_map.x_coords[0], world_map.x_coords[-1],    # left, right
              world_map.y_coords[-1], world_map.y_coords[0]]    # bottom, top

    im = ax.imshow(heightmap, cmap='viridis', origin='upper', extent=extent)
    ax.set_title("Height Map")
    ax.set_xlabel("x (world coords)")
    ax.set_ylabel("y (world coords)")
    
    # Create custom legend with 5 color patches showing heights
    # Get 5 evenly spaced height values
    height_values = np.linspace(min_height, max_height, 5)
    # Normalize heights to [0, 1] for colormap
    norm = mcolors.Normalize(vmin=min_height, vmax=max_height)
    cmap = plt.cm.viridis
    
    # Create legend patches
    legend_elements = []
    for height_val in height_values:
        color = cmap(norm(height_val))
        patch = Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5)
        legend_elements.append((patch, f'{height_val:.2f}'))
    
    # Create legend and position it on the left side of the plot
    legend = ax.legend([elem[0] for elem in legend_elements], 
                       [elem[1] for elem in legend_elements],
                       title='Height',
                       loc='center left',
                       bbox_to_anchor=(0.02, 0.5),
                       frameon=True,
                       fancybox=True,
                       shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    ax.add_artist(legend)
    
    return ax

def plot_cluster_of_points(ax, world_map, cluster, color='blue', s=10, alpha=0.9, edgecolors='black', linewidth=1.0, label=None):
    convert_grid_to_world_coordinates = lambda cluster: [world_map.map_to_world(row, col) for row, col in cluster]
    world_x_coords, world_y_coords = zip(*convert_grid_to_world_coordinates(cluster))
    ax.scatter( x=world_x_coords, y=world_y_coords,  # (x, y) coordinates
                c=[color], 
                s=s,  # Make points large enough to be visible
                alpha=alpha,
                edgecolors=edgecolors,
                linewidth=linewidth,
                label=label
            )
    return ax

def plot_frontiers(ax, world_map, frontier_clusters, best_frontier_cluster = None, create_legend=True):
    print(f"Plotting {len(frontier_clusters)} frontier clusters")
    
    to_plot_frontiers = frontier_clusters + ([best_frontier_cluster] if best_frontier_cluster is not None else []) 
    best_frontier_index = len(frontier_clusters) if best_frontier_cluster is not None else None
    best_frontier_color = 'green'

    colors = plt.cm.tab20(np.linspace(0, 1, len(to_plot_frontiers)))
    for cluster_idx, cluster in enumerate(to_plot_frontiers):
        if len(cluster) == 0: continue

        is_best_frontier = cluster_idx == best_frontier_index
        size_of_point = 25 if not is_best_frontier else 100
        color = best_frontier_color if is_best_frontier else colors[cluster_idx]
        label = f'Frontier {cluster_idx} ({len(cluster)} points)'
        plot_cluster_of_points(ax, world_map, cluster, color=color, s=size_of_point, alpha=0.9, edgecolors='black', linewidth=1.0, label=label)
        
    # Add legend if there are frontiers - positioned on left side outside plot
    if create_legend and len(frontier_clusters) > 0:
        ax.legend(bbox_to_anchor=(0.0, 0.5), loc='center right', fontsize=8, framealpha=0.9)
        
    return ax

def plot_path(ax, path):
    if path is None or len(path) == 0:
        return  # Skip plotting if path is None or empty
    
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]

    # Plot path line
    ax.plot(path_x, path_y, '-b', linewidth=3, label='A* Path')

    # Plot arrows for yaw direction (skip first and last points to avoid clutter)
    arrow_scale = 0.3
    for i in range(1, len(path) - 1):  # Skip first and last
        x, y, yaw = path[i]
        dx = arrow_scale * np.cos(yaw)
        dy = arrow_scale * np.sin(yaw)
        ax.arrow(x, y, dx, dy, 
                head_width=0.08, head_length=0.08, 
                fc='blue', ec='blue', alpha=0.7, linewidth=1.5)
    return ax

def plot_robot_cams(ax, robot_world_coords, robot_yaw, flash_length=0.5, camera_name = None, color='blue'):
    """
    Plot camera directions as blue flashes/rays from robot position.
    Shows current robot camera viewing directions.
    
    Camera positions relative to robot yaw (12:00 = forward):
    - head_right: 10:30 (-30°)
    - head_left: 13:30 (+30°)
    - left: 15:00 (+90°)
    - right: 9:00 (-90°)
    - rear: 18:00 (180°)
    """
    camera_angles = {
        'head_right': -np.pi / 6,   # 10:30, -30°
        'head_left': np.pi / 6,      # 13:30, +30°
        'left': np.pi / 2,           # 15:00, +90°
        'right': -np.pi / 2,         # 9:00, -90°
        'rear': np.pi                # 18:00, 180°
    }
    
    for camera_name, angle_offset in camera_angles.items():
        if camera_name is not None and camera_name != camera_name: 
            continue
        camera_yaw = robot_yaw + angle_offset
        dx = flash_length * np.cos(camera_yaw)
        dy = flash_length * np.sin(camera_yaw)
        
        ax.arrow(robot_world_coords[0], robot_world_coords[1], 
                dx, dy, 
                head_width=0.08, head_length=0.1, 
                fc=color, ec=color, alpha=0.7, linewidth=1.5)

    return ax


def plot_detection_batches(ax, detection_data_cache):
    for batch_idx, batch in enumerate(detection_data_cache):
        # Most recent batch (last one) is red, all others are purple
        for pair in batch["pairs"]:
            robot_pos = pair['robot_position']
            robot_yaw = pair['robot_yaw']
            camera_name = pair['camera_name']
            ax = plot_robot_cams(ax, robot_pos, robot_yaw, camera_name=camera_name, color="purple", flash_length=0.1)
    return ax

# -- plotters -- #
def plot_state(ax, world_map, robot_world_coords = None, robot_yaw = None, 
frontier_clusters = None, best_frontier_cluster = None, best_frontier_goal_pose = None, 
path_to_goal = None, goal_pose = None, detection_data_cache = None, object_records = None, blacklist_labels = [], show_heightmap = False):
    if ax is None:    
        fig, ax = plt.subplots(figsize=(12, 10))

    # Get robot mask and overlay it on the occupancy grid.
    if robot_world_coords is not None and robot_yaw is not None:
        ax = overlay_robot_position(ax, world_map, robot_world_coords, robot_yaw, robot_label="Current Robot Pose") 
        ax = plot_robot_cams(ax, robot_world_coords, robot_yaw)
    if show_heightmap:
        ax = plot_height_map(ax, world_map)
    else:   
        ax = plot_occupancy_map(ax, world_map)
    if frontier_clusters is not None:
        ax = plot_frontiers(ax, world_map, frontier_clusters, best_frontier_cluster)
    if best_frontier_cluster is not None:
        ax = plot_cluster_of_points(ax, world_map, best_frontier_cluster, color='green', s=25)
    if best_frontier_goal_pose is not None:
        ax = plot_goal_pose_arrow(ax, world_map, best_frontier_goal_pose, force_color='Green', goal_label='Selected Frontier')
    if goal_pose is not None:
        ax = plot_goal_pose_arrow(ax, world_map, goal_pose, force_color='orange', goal_label='Goal Pose')
    if path_to_goal is not None:
        ax = plot_path(ax, path_to_goal)
    if detection_data_cache is not None:
        ax = plot_detection_batches(ax, detection_data_cache)
    if object_records is not None:
        x_coords, y_coords, heights, labels , unique_robot_positions = parse_object_records(object_records, blacklist_labels)
        ax = plot_object_points(ax, x_coords, y_coords, heights, labels)
    return ax



def plot_robot_rectangle(ax, x, y, yaw, length, width, color='green', alpha=0.5, label=None):
    """Plot a rectangle representing the robot's footprint - aligned with WorldMap collision checking."""
    # Create rectangle centered at origin (same as WorldMap collision checking)
    rect = Rectangle((-length/2, -width/2), length, width,
                    facecolor=color, edgecolor='darkgreen',
                    linewidth=1.5, alpha=alpha, label=label)
    
    # Apply rotation and translation (same transformation as WorldMap)
    t = Affine2D().rotate(yaw).translate(x, y) + ax.transData
    rect.set_transform(t)
    
    # Draw front direction indicator
    front_x = x + (length/2) * math.cos(yaw)
    front_y = y + (length/2) * math.sin(yaw)
    ax.plot([x, front_x], [y, front_y], 'y-', linewidth=3, alpha=1.0)
    
    return ax

def parse_object_records(object_records, blacklist_labels = []):
       # Extract valid positions and metadata
    x_coords = []
    y_coords = []
    heights = []
    labels = []
    
    for record in object_records:
        spatial_meta = record.get("spatial_metadata", {})
        semantic_meta = record.get("semantic_metadata", {})

        if semantic_meta.get("label", "unknown") in blacklist_labels:
            continue
        
        position_3d = spatial_meta.get("position_3d")
        if position_3d is not None and len(position_3d) >= 3:
            # Check if position is valid
            if spatial_meta.get("is_valid_depth", False):
                x_coords.append(position_3d[0])
                y_coords.append(position_3d[1])
                
                # Get height (z coordinate)
                height = position_3d[2]
                heights.append(height)
                
                # Get label
                label = semantic_meta.get("label", "unknown")
                labels.append(label)

        # Extract robot position and yaw from the first object record
    unique_robot_positions = set()
    for record in object_records:
        frame_meta = record.get("frame_metadata", {})
        robot_position = frame_meta.get("robot_position")
        robot_yaw = frame_meta.get("robot_yaw")
        if robot_position is not None and robot_yaw is not None:
            unique_robot_positions.add((tuple(robot_position), robot_yaw))
    
    print(f"Unique robot positions: {len(unique_robot_positions)}")

    return x_coords, y_coords, heights, labels, unique_robot_positions

def plot_object_points(ax, x_coords, y_coords, heights, labels):
    print(f"Plotting {len(x_coords)} object points")
    # Plot points (on top of map if present)
    scatter = ax.scatter(x_coords, y_coords, c=heights, cmap='viridis', s=100, alpha=0.6, 
                        edgecolors='black', linewidths=1, zorder=2)
    
    # Add colorbar for height
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Height (m)', rotation=270, labelpad=20)
    
    # Add labels for each point
    for i, (x, y, label, height) in enumerate(zip(x_coords, y_coords, labels, heights)):
        # Create label text with object name and height
        label_text = f"{label}\n{height:.2f}m"
        ax.annotate(label_text, (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'),
                   ha='left', zorder=3)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Top-Down 2D Map of Object Detections', fontsize=14)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    # ax.legend(loc='upper right', fontsize=10)
    return ax

import pickle as pk
from tiamat_agent.planning.WorldMap import WorldMap
def load_saved_world_map():
    # Use default path similar to z_play_astar.py
    map_path = "/home/tiamat_eval/tiamatl_eval_mvp/tiamat_agent/data/saved_maps/state_log_frame_0.pkl"
    
    print(f"Loading map from: {map_path}")
    with open(map_path, "rb") as f:
        state = pk.load(f)
    
    # Extract the world map from the state and recreate it
    old_world_map = state["world_map"]
    world_map = WorldMap(
        old_world_map.occupancy_map, 
        old_world_map.heightmap, 
        old_world_map.resolution, 
        old_world_map.x_coords, 
        old_world_map.y_coords, 
        yaw_step=45
    )

    return world_map


def plot_object_records_top_down(ax, object_records, world_map=None, 
                                    show: bool = True, save_path: Optional[str] = None, 
                                     overlay_map: bool = False,
                                     plot_robot: bool = True, 
                                     blacklist_labels = []):
    """
    Visualize object records in a top-down 2D map.
    
    Plots each object's x, y position with labels showing the object name and height.
    Optionally overlays an occupancy map as background and plots robot position as a rectangle.
    
    Args:
        object_records: List of object record dictionaries
        show: If True, display the plot
        save_path: Optional path to save the figure
        overlay_map: If True, overlay the occupancy map as background
        map_path: Optional path to the saved map pickle file. If None and overlay_map=True,
                  uses default path from logs.
        plot_robot: If True, plot the robot as a rectangle with position and yaw from object records
    """
    if not object_records:
        print("No object records to visualize")
        return
    
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))

    x_coords, y_coords, heights, labels , unique_robot_positions = parse_object_records(object_records, blacklist_labels)
    ax = plot_object_points(ax, x_coords, y_coords, heights, labels)

    if overlay_map:
        world_map = load_saved_world_map() if world_map is None else world_map
        ax = plot_occupancy_map(ax, world_map)
    
    if plot_robot:
        for robot_position, robot_yaw in unique_robot_positions:
            # ax = overlay_robot_position(ax, world_map, robot_position, robot_yaw, force_color='blue', robot_label="Robot")
            ax = plot_robot_rectangle(ax, robot_position[0], robot_position[1], robot_yaw, world_map.robot_length, world_map.robot_width, color='blue', alpha=0.5, label='Robot')
    
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved top-down map to {save_path}")

    if show:
        plt.show()
    
    return ax


# Vision Plotters. 

def normalize_depth_for_visualization(depth_image: np.ndarray, depth_scale: float = 1000.0, max_depth: float = 10.0) -> Tuple[np.ndarray, float, float]:
    """Normalize depth image to 0-255 range using per-image depth range."""
    depth_meters = depth_image.astype(np.float32) / depth_scale if np.nanmax(depth_image[depth_image > 0]) > 100 else depth_image.astype(np.float32)
    valid_mask = (depth_meters > 0) & np.isfinite(depth_meters) & (depth_meters <= max_depth)
    
    if not np.any(valid_mask):
        return np.zeros((*depth_image.shape, 3), dtype=np.uint8), 0.0, max_depth
    
    img_min, img_max = np.nanmin(depth_meters[valid_mask]), min(np.nanmax(depth_meters[valid_mask]), max_depth)
    if img_max <= img_min:
        img_max = img_min + 0.1
    
    depth_normalized = np.clip(((depth_meters - img_min) / (img_max - img_min) * 255), 0, 255).astype(np.uint8)
    depth_normalized[~valid_mask] = 0
    return cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET), img_min, img_max

def create_image_collage(images: list, labels: list, max_cols: int = 3, resize: bool = True) -> np.ndarray:
    """Create a collage of images with labels.
    
    Args:
        images: List of images (numpy arrays)
        labels: List of labels for each image
        max_cols: Maximum number of columns in the collage
        resize: If True, resize all images to match first image size. If False, preserve original sizes.
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    num_cols = min(max_cols, len(images))
    num_rows = (len(images) + num_cols - 1) // num_cols
    
    # Convert images to BGR if needed
    processed_images = []
    for img in images:
        img = img.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        processed_images.append(img)
    
    if resize:
        # Original resize logic
        target_h, target_w = processed_images[0].shape[:2]
        resized_images = [cv2.resize(img, (target_w, target_h)) for img in processed_images]
        
        collage = np.ones((num_rows * target_h + (num_rows - 1) * 10, num_cols * target_w + (num_cols - 1) * 10, 3), dtype=np.uint8) * 255
        
        for idx, (img, label) in enumerate(zip(resized_images, labels)):
            y_start, x_start = (idx // num_cols) * (target_h + 10), (idx % num_cols) * (target_w + 10)
            collage[y_start:y_start + target_h, x_start:x_start + target_w] = img
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(collage, (x_start, y_start), (x_start + text_w + 10, y_start + text_h + 10), (0, 0, 0), -1)
            cv2.putText(collage, label, (x_start + 5, y_start + text_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        # Preserve original sizes
        row_heights = []
        col_widths = [0] * num_cols
        
        for idx, img in enumerate(processed_images):
            row = idx // num_cols
            col = idx % num_cols
            h, w = img.shape[:2]
            
            if row >= len(row_heights):
                row_heights.append(0)
            row_heights[row] = max(row_heights[row], h)
            col_widths[col] = max(col_widths[col], w)
        
        padding = 10
        label_height = 30
        total_h = sum(row_heights) + (num_rows - 1) * padding + num_rows * label_height
        total_w = sum(col_widths) + (num_cols - 1) * padding
        
        collage = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
        
        current_y = 0
        for row in range(num_rows):
            current_x = 0
            for col in range(num_cols):
                idx = row * num_cols + col
                if idx >= len(processed_images):
                    break
                
                img = processed_images[idx]
                label = labels[idx]
                h, w = img.shape[:2]
                
                y_offset = (row_heights[row] - h) // 2
                y_pos = current_y + label_height + y_offset
                x_pos = current_x + (col_widths[col] - w) // 2
                
                collage[y_pos:y_pos + h, x_pos:x_pos + w] = img
                
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_x = current_x + (col_widths[col] - text_w) // 2
                cv2.rectangle(collage, (text_x - 5, current_y), (text_x + text_w + 5, current_y + text_h + 5), (0, 0, 0), -1)
                cv2.putText(collage, label, (text_x, current_y + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                current_x += col_widths[col] + padding
            
            current_y += row_heights[row] + padding + label_height
    
    return collage

def visualize_entry_images(entry: ObsDataEntry, show: bool = True, save_path: Optional[str] = None, 
                           max_depth: float = 10.0, depth_scale: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Visualize all RGB and depth images from an entry in collage format. Uses per-image depth normalization."""
    pairs = entry.get_rgb_depth_pairs()
    if not pairs:
        print("No RGB-depth pairs found in entry")
        return None, None
    
    rgb_images, depth_images, rgb_labels, depth_labels = [], [], [], []
    for rgb_name, rgb_image, depth_name, depth_image in pairs:
        rgb_images.append(rgb_image)
        rgb_labels.append(rgb_name)
        depth_vis, img_min, img_max = normalize_depth_for_visualization(depth_image, depth_scale, max_depth)
        depth_images.append(depth_vis)
        depth_labels.append(f"{depth_name}\n({img_min:.2f}m-{img_max:.2f}m)")
    
    rgb_collage = create_image_collage(rgb_images, rgb_labels, max_cols=3)
    depth_collage = create_image_collage(depth_images, depth_labels, max_cols=3)
    
    if show:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        axes[0].imshow(rgb_collage)
        axes[0].set_title(f'RGB Images - Timestamp: {entry.header_stamp}', fontsize=14)
        axes[0].axis('off')
        axes[1].imshow(depth_collage)
        axes[1].set_title(f'Depth Images (Per-image normalization, blue=far, red=near) - Timestamp: {entry.header_stamp}', fontsize=14)
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()
    
    if save_path:
        cv2.imwrite(save_path, np.vstack([rgb_collage, depth_collage]))
        print(f"Saved collage to {save_path}")
    
    return rgb_collage, depth_collage

def visualize_images_and_metadata(images: list, metadata: list, show: bool = True, save_path: Optional[str] = None, max_cols: int = 3) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Visualize a list of images and list of metadata."""
    if not images or not metadata:
        print("No images or metadata found")
        return None, None
    
    if len(images) != len(metadata):
        print(f"Mismatch: {len(images)} images but {len(metadata)} metadata entries")
        return None, None
    
    collage = create_image_collage(images, metadata, max_cols=max_cols)
    
    if show:
        plt.figure(figsize=(15, 10))
        plt.imshow(collage)
        plt.title('Images with Metadata', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    if save_path:
        cv2.imwrite(save_path, collage)
        print(f"Saved collage to {save_path}")
    
    return collage, None

# visualize_images_and_metadata([d["rgb_image"] for d in image_data_by_id.values()], [", ".join(d["vlm_object_name_list"]) for d in image_data_by_id.values()])

import open3d as o3d
# image_data_by_id , visualizaitons
def visualize_all_object_pointclouds(image_data_by_id):
    """
    Concatenates all object_pcd_segment point clouds from image_data_by_id
    and visualizes the combined point cloud using Open3D with label annotations.
    Preserves original colors from point clouds and displays labels as 3D text in the viewer.
    
    Args:
        image_data_by_id: Dictionary mapping image IDs to image data containing
                         yolo_object_dict with object_pcd_segment entries
    """
    geometries = []
    label_positions = []
    
    # Collect all point clouds with original colors and labels
    for image_id, image_data in image_data_by_id.items():
        yolo_object_dict = image_data.get("yolo_object_dict", {})
        
        for object_key, object_data in yolo_object_dict.items():
            object_pcd = object_data.get("object_pcd_segment")
            if object_pcd is not None and isinstance(object_pcd, o3d.geometry.PointCloud):
                if len(object_pcd.points) > 0:
                    label = object_data.get("label", "unknown")
                    center_3d = object_data.get("center_3d")
                    
                    # Preserve original point cloud with existing colors (don't repaint)
                    geometries.append((object_pcd, label))
                    
                    # Store label position for annotation
                    if center_3d is not None and len(center_3d) == 3:
                        label_positions.append((center_3d, label))
    
    if not geometries:
        print("No point clouds found to visualize")
        return
    
    # Simple visualization preserving original colors
    all_pcds = [pcd for pcd, _ in geometries]
    
    # Add small markers at label positions (optional - to mark centroids)
    for center_3d, label in label_positions:
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        marker.translate(center_3d)
        marker.paint_uniform_color([1.0, 1.0, 1.0])  # White markers
        all_pcds.append(marker)
    
    render_option = o3d.visualization.RenderOption()
    render_option.point_size = 2.0
    
    print(f"\nVisualizing {len(geometries)} point cloud segments (preserving original colors)")
    print(f"Labels at centroids:")
    for center, label in label_positions:
        print(f"  {label:20s} at [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    print("\nNote: White spheres mark label positions. Point clouds use their original colors.")
    
    o3d.visualization.draw_geometries(all_pcds, window_name="Object Point Clouds with Labels")


def visualize_full_pointclouds(image_data_by_id):
    """
    Simple function that visualizes all full_pcd point clouds from image_data_by_id.
    
    Args:
        image_data_by_id: Dictionary mapping image IDs to image data containing full_pcd entries
    """
    all_pcds = []
    
    for image_id, image_data in image_data_by_id.items():
        full_pcd = image_data.get("full_pcd")
        if full_pcd is not None and isinstance(full_pcd, o3d.geometry.PointCloud):
            if len(full_pcd.points) > 0:
                all_pcds.append(full_pcd)
    
    if not all_pcds:
        print("No full_pcd point clouds found to visualize")
        return
    
    print(f"\nVisualizing {len(all_pcds)} full point clouds")
    o3d.visualization.draw_geometries(all_pcds, window_name="All Full Point Clouds")

def visualize_all_images_raw(image_data_by_id, max_depth: float = 10.0, depth_scale: float = 1000.0, 
                         max_cols: int = 3, resize: bool = False):
    """Visualize all RGB (normal and rotated) and depth images from image_data_by_id in a single plot.
    
    Args:
        image_data_by_id: Dictionary mapping image IDs to image data
        max_depth: Maximum depth for visualization
        depth_scale: Scale factor for depth images
        max_cols: Maximum number of columns in collage
        resize: If True, resize images to match. If False, preserve original sizes.
    """
    all_images = []
    all_labels = []
    
    for image_id, image_data in image_data_by_id.items():
        # Original RGB
        if "rgb_image" in image_data:
            all_images.append(image_data["rgb_image"])
            all_labels.append(f"{image_data.get('rgb_name', image_id)}\n(original)")
        
        # Rotated RGB
        if "rotated_rgb_image" in image_data:
            all_images.append(image_data["rotated_rgb_image"])
            all_labels.append(f"{image_data.get('rgb_name', image_id)}\n(rotated)")
        
        # Depth
        if "depth_image" in image_data:
            depth_vis, img_min, img_max = normalize_depth_for_visualization(
                image_data["depth_image"], depth_scale, max_depth
            )
            all_images.append(depth_vis)
            all_labels.append(f"{image_data.get('depth_name', image_id)}\n({img_min:.2f}m-{img_max:.2f}m)")
    
    if not all_images:
        return
    
    # Create collage using existing function with resize flag
    collage = create_image_collage(all_images, all_labels, max_cols=max_cols, resize=resize)
    
    title_suffix = "Resized" if resize else "Original Sizes Preserved"
    plt.figure(figsize=(15, 10))
    plt.imshow(collage)
    plt.title(f'All Images (RGB Original, RGB Rotated, Depth) - {title_suffix}', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_all_images_yolo_detections_in_rotated(image_data_by_id, max_cols: int = 3):
    """Visualize YOLO detections in rotated images only, preserving original sizes."""
    annotated_images = []
    labels = []
    
    for image_id, image_data in image_data_by_id.items():
        if "rotated_rgb_image" not in image_data:
            continue
        
        img = image_data["rotated_rgb_image"].copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # Draw bounding boxes if available
        bboxes = image_data.get("yolo_rotated_detections_xyxy")
        class_ids = image_data.get("yolo_detections_class_ids", [])
        confidences = image_data.get("yolo_detections_confidences", [])
        object_dict = image_data.get("yolo_object_dict", {})
        
        if bboxes is not None and len(bboxes) > 0:
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, min(x1, img.shape[1] - 1))
                y1 = max(0, min(y1, img.shape[0] - 1))
                x2 = max(0, min(x2, img.shape[1] - 1))
                y2 = max(0, min(y2, img.shape[0] - 1))
                
                if x1 < x2 and y1 < y2:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label if available
                    label = ""
                    if i < len(class_ids) and f"object_{i}" in object_dict:
                        obj_data = object_dict[f"object_{i}"]
                        label = obj_data.get("label", "")
                        if i < len(confidences):
                            label += f" {confidences[i]:.2f}"
                    elif i < len(class_ids):
                        label = f"class_{class_ids[i]}"
                    
                    if label:
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(img, (x1, y1 - text_h - 5), (x1 + text_w + 5, y1), (0, 255, 0), -1)
                        cv2.putText(img, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        annotated_images.append(img)
        labels.append(f"{image_data.get('rgb_name', image_id)}\n({len(bboxes) if bboxes is not None else 0} detections)")
    
    if not annotated_images:
        print("No rotated images with detections found")
        return
    
    collage = create_image_collage(annotated_images, labels, max_cols=max_cols, resize=False)
    plt.figure(figsize=(15, 10))
    plt.imshow(collage)
    plt.title('YOLO Detections in Rotated Images', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_all_images_yolo_detections_in_non_rotated(image_data_by_id, max_cols: int = 3):
    """Visualize YOLO detections in non-rotated (original) images only, preserving original sizes."""
    annotated_images = []
    labels = []
    
    for image_id, image_data in image_data_by_id.items():
        if "rgb_image" not in image_data:
            continue
        
        img = image_data["rgb_image"].copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # Draw bounding boxes if available
        bboxes = image_data.get("yolo_detections_xyxy")
        class_ids = image_data.get("yolo_detections_class_ids", [])
        confidences = image_data.get("yolo_detections_confidences", [])
        object_dict = image_data.get("yolo_object_dict", {})
        
        if bboxes is not None and len(bboxes) > 0:
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, min(x1, img.shape[1] - 1))
                y1 = max(0, min(y1, img.shape[0] - 1))
                x2 = max(0, min(x2, img.shape[1] - 1))
                y2 = max(0, min(y2, img.shape[0] - 1))
                
                if x1 < x2 and y1 < y2:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label if available
                    label = ""
                    if i < len(class_ids) and f"object_{i}" in object_dict:
                        obj_data = object_dict[f"object_{i}"]
                        label = obj_data.get("label", "")
                        if i < len(confidences):
                            label += f" {confidences[i]:.2f}"
                    elif i < len(class_ids):
                        label = f"class_{class_ids[i]}"
                    
                    if label:
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(img, (x1, y1 - text_h - 5), (x1 + text_w + 5, y1), (0, 255, 0), -1)
                        cv2.putText(img, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        annotated_images.append(img)
        labels.append(f"{image_data.get('rgb_name', image_id)}\n({len(bboxes) if bboxes is not None else 0} detections)")
    
    if not annotated_images:
        print("No non-rotated images with detections found")
        return
    
    collage = create_image_collage(annotated_images, labels, max_cols=max_cols, resize=False)
    plt.figure(figsize=(15, 10))
    plt.imshow(collage)
    plt.title('YOLO Detections in Non-Rotated Images', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_yolo_detections_in_depth(image_data_by_id, max_depth: float = 10.0, depth_scale: float = 1000.0, 
                                       max_cols: int = 3):
    """Visualize YOLO detections in depth images using scaled bboxes, preserving original sizes.
    
    Args:
        image_data_by_id: Dictionary mapping image IDs to image data
        max_depth: Maximum depth for visualization
        depth_scale: Scale factor for depth images
        max_cols: Maximum number of columns in collage
    """
    annotated_images = []
    labels = []
    
    for image_id, image_data in image_data_by_id.items():
        if "depth_image" not in image_data:
            continue
        
        # Normalize depth image for visualization
        depth_vis, img_min, img_max = normalize_depth_for_visualization(
            image_data["depth_image"], depth_scale, max_depth
        )
        
        # Get object dictionary with scaled bboxes
        object_dict = image_data.get("yolo_object_dict", {})
        
        # Draw scaled bounding boxes on depth image
        num_detections = 0
        for obj_key, obj_data in object_dict.items():
            scaled_bbox = obj_data.get("scaled_bbox_xyxy")
            if scaled_bbox is not None and len(scaled_bbox) == 4:
                x1, y1, x2, y2 = map(int, scaled_bbox)
                # Clamp to image bounds
                x1 = max(0, min(x1, depth_vis.shape[1] - 1))
                y1 = max(0, min(y1, depth_vis.shape[0] - 1))
                x2 = max(0, min(x2, depth_vis.shape[1] - 1))
                y2 = max(0, min(y2, depth_vis.shape[0] - 1))
                
                if x1 < x2 and y1 < y2:
                    cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label if available
                    label = obj_data.get("label", "")
                    if label:
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(depth_vis, (x1, y1 - text_h - 5), (x1 + text_w + 5, y1), (0, 255, 0), -1)
                        cv2.putText(depth_vis, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    num_detections += 1
        
        annotated_images.append(depth_vis)
        labels.append(f"{image_data.get('depth_name', image_id)}\n({num_detections} detections)\n({img_min:.2f}m-{img_max:.2f}m)")
    
    if not annotated_images:
        print("No depth images with detections found")
        return
    
    collage = create_image_collage(annotated_images, labels, max_cols=max_cols, resize=False)
    plt.figure(figsize=(15, 10))
    plt.imshow(collage)
    plt.title('YOLO Detections in Depth Images (Scaled Bboxes)', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------


def find_and_plot_frontiers(ax, world_map, robot_pos=(0, 0), free_thresh=0.3, unknown_val=-1, min_cluster_size=5):
    """
    Find frontiers in occupancy grid and plot them.
    Frontiers = free cells adjacent to unknown cells.
    
    Args:
        ax: matplotlib axis
        world_map: WorldMap object with occupancy_map attribute
        robot_pos: (x, y) world coordinates of robot
        free_thresh: threshold below which cell is free (for probability grids)
        unknown_val: value representing unknown cells (-1 typically)
        min_cluster_size: minimum points to consider a frontier cluster
    
    Returns:
        ax, frontier_clusters (list of arrays), accessible_clusters (list of arrays)
    """
    from tiamat_agent.mapping.frontiers import find_frontiers
    from scipy import ndimage
    
    import time
    start_time = time.time()
    accessible_clusters, inaccessible_clusters, frontier_mask = find_frontiers(world_map, robot_world_pos = (-10, 8), min_cluster_size=2)
    end_time = time.time()
    print(f"Time taken to find frontiers: {end_time - start_time} seconds")
    
    for idx, cluster in enumerate(accessible_clusters):
        # Convert grid coords to world coords
        world_coords = [world_map.map_to_world(row, col) for row, col in cluster]
        wx, wy = zip(*world_coords)
        ax.scatter(wx, wy, c=['green'], s=15, alpha=0.8, 
                   edgecolors='green', linewidth=0.5,
                   label=f'Accessible Frontier {idx} ({len(cluster)} pts)')
    
    for idx, cluster in enumerate(inaccessible_clusters):
        world_coords = [world_map.map_to_world(row, col) for row, col in cluster]
        wx, wy = zip(*world_coords)
        ax.scatter(wx, wy, c='red', s=10, alpha=0.5, marker='x',
                   label=f'Inaccessible Frontier ({len(cluster)} pts)' if idx == 0 else None)
    
    # Mark robot position
    ax.scatter(robot_pos[0], robot_pos[1], c='blue', s=200, marker='*', 
               zorder=10, label='Robot')
    
    # Plot centroids of accessible clusters
    for cluster in accessible_clusters:
        centroid_grid = cluster.mean(axis=0).astype(int)
        # Ensure indices are within bounds
        row = int(np.clip(centroid_grid[0], 0, world_map.occupancy_map.shape[0] - 1))
        col = int(np.clip(centroid_grid[1], 0, world_map.occupancy_map.shape[1] - 1))
        centroid_world = world_map.map_to_world(row, col)
        ax.scatter(centroid_world[0], centroid_world[1], c='green', s=100, 
                   marker='D', edgecolors='black', linewidth=2, zorder=5)
    
    ax.legend(bbox_to_anchor=(0.0, 0.5), loc='center right', fontsize=8, framealpha=0.9)
    ax.set_title(f"Frontiers: {len(accessible_clusters)} accessible, {len(inaccessible_clusters)} inaccessible")
    
    print(f"Found {len(accessible_clusters) + len(inaccessible_clusters)} total frontier clusters")
    print(f"  - {len(accessible_clusters)} accessible from robot")
    print(f"  - {len(inaccessible_clusters)} inaccessible")
    
    return ax, accessible_clusters, inaccessible_clusters



def load_saved_world_map_v2( sample = False):
    from pathlib import Path
    import glob
    import random
    
    backup_folder = Path("/home/tiamat_eval/tiamatl_eval_mvp/logs/current_run_outputs nov 25 backup/map_dumps")
    world_map_files = sorted(glob.glob(str(backup_folder / "world_map_*.pkl")))
    if not world_map_files:
        raise FileNotFoundError(f"No world map files found in {backup_folder}")
    if sample:
        map_path = random.choice(world_map_files)
    else:
        map_path = world_map_files[-1]
    
    print(f"Loading map from: {map_path}")
    with open(map_path, "rb") as f:
        world_map = pk.load(f)
    return world_map

def load_saved_world_map_v3(sample = False):
    from pathlib import Path
    import glob
    import random
    
    backup_folder = Path("/home/tiamat_eval/tiamatl_eval_mvp/logs/current_run_outputs nov 29 backup/map_dumps")
    world_map_files = sorted(glob.glob(str(backup_folder / "world_map_*.pkl")))
    if not world_map_files:
        raise FileNotFoundError(f"No world map files found in {backup_folder}")
    if sample:
        map_path = random.choice(world_map_files)
    else:
        map_path = world_map_files[-1]
    
    print(f"Loading map from: {map_path}")
    with open(map_path, "rb") as f:
        world_map = pk.load(f)
    return world_map


def viz_worldmap_with_frontier_routine():
    from tiamat_agent.mapping.frontiers import find_best_frontier_with_goal
    # world_map = load_saved_world_map_v2(sample=True)
    world_map = load_saved_world_map_v3(sample=True)
    # (robot_pos, robot_yaw, world_map) = pk.load(open(p, "rb"))
    # ... your loading code ...
    # print(f"robot_pos: {robot_pos}, robot_yaw: {robot_yaw}")
    robot_pos = (-10.0, 8.0)
    robot_yaw = 0.0
    
    # Find frontiers with goal poses
    from time import time
    start_time = time()

    frontier_result = find_best_frontier_with_goal(
        world_map,
        robot_world_pos=robot_pos,
        robot_world_yaw=robot_yaw,
        min_cluster_size=5,
        selection_method='closest',
        force_pessimistic=True
    )
    best_goal, best_cluster, accessible_clusters, goal_poses = frontier_result.best_goal_pose, frontier_result.best_cluster, frontier_result.accessible_clusters, frontier_result.goal_poses
    end_time = time()
    print(f"Time taken to find best frontier with goal: {end_time - start_time:0.5f} seconds")


    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))

    
    # Plot robot
    ax = overlay_robot_position(ax, world_map, robot_pos, robot_yaw, use_optimistic=True,
                                 force_color='blue', robot_label="Robot")
    
    # Plot all accessible frontiers using plot_frontiers function (legend created separately)
    ax = plot_frontiers(ax, world_map, accessible_clusters, best_cluster, create_legend=False)
    
    # Plot goal poses for all frontiers
    colors = plt.cm.tab10(np.linspace(0, 1, len(accessible_clusters)))
    print("goal_poses: ", goal_poses)
    for idx, goal in enumerate(goal_poses):
        # Plot goal pose if valid using the overlay function
        if goal is not None:
            # ax = overlay_robot_position(ax, world_map, (goal[0], goal[1]), goal[2], use_optimistic=True, robot_label=f'Frontier Goal {idx}')
            ax = plot_goal_pose_arrow(ax, world_map, goal, force_color=colors[idx], goal_label=f'Frontier Goal {idx}', use_optimistic=True)
    
    # Highlight best frontier and goal with special styling
    if best_goal:
        ax = plot_goal_pose_arrow(ax, world_map, best_goal, 
                               force_color='lime', 
                               goal_label='Best Goal')
    
    ax = plot_occupancy_map(ax, world_map)
    
    # Position legend on left side, outside plot area to avoid blocking the map
    ax.legend(bbox_to_anchor=(0.0, 0.5), loc='center right', fontsize=8, framealpha=0.9)
    plt.tight_layout(rect=[0.22, 0, 1, 1])  # Leave 22% left margin for legend
    plt.show()


def load_saved_object_records_from_current_run(sample=False):
    """Load object records from current_run_outputs/object_list_dumps folder."""
    from pathlib import Path
    import glob
    import random
    from tiamat_agent.utils.session_logger import LOGS_FOLDER
    
    object_list_dump_folder = LOGS_FOLDER.joinpath("current_run_outputs", "object_list_dumps")
    object_library_files = sorted(glob.glob(str(object_list_dump_folder / "object_library_*.pkl")))
    
    if not object_library_files:
        raise FileNotFoundError(f"No object library files found in {object_list_dump_folder}")
    
    if sample:
        library_path = random.choice(object_library_files)
    else:
        library_path = object_library_files[-1]  # Most recent
    
    print(f"Loading object library from: {library_path}")
    with open(library_path, "rb") as f:
        object_library = pk.load(f)
    
    # Convert ObjectLibrary to list of object records
    # ObjectLibrary.__iter__ returns iter(self.objects_by_id.values())
    object_records = list(object_library)
    
    print(f"Loaded {len(object_records)} object records")
    return object_records


def load_saved_world_map_from_current_run(sample=False):
    """Load world map from current_run_outputs/map_dumps folder."""
    from pathlib import Path
    import glob
    import random
    from tiamat_agent.utils.session_logger import LOGS_FOLDER
    
    map_dump_folder = LOGS_FOLDER.joinpath("current_run_outputs", "map_dumps")
    world_map_files = sorted(glob.glob(str(map_dump_folder / "world_map_*.pkl")))
    
    if not world_map_files:
        raise FileNotFoundError(f"No world map files found in {map_dump_folder}")
    
    if sample:
        map_path = random.choice(world_map_files)
    else:
        map_path = world_map_files[-1]  # Most recent
    
    print(f"Loading world map from: {map_path}")
    with open(map_path, "rb") as f:
        world_map = pk.load(f)
    
    return world_map


def plot_object_records_top_down_interactive(object_records, world_map=None, 
                                             overlay_map: bool = False,
                                             plot_robot: bool = True, 
                                             blacklist_labels = [],
                                             show: bool = True):
    """
    Interactive visualization of object records in a top-down 2D map using Plotly.
    Allows clicking on legend items to show/hide specific object labels.
    
    Args:
        object_records: List of object record dictionaries
        world_map: Optional WorldMap object
        overlay_map: If True, overlay the occupancy map as background
        plot_robot: If True, plot robot positions from object records
        blacklist_labels: List of labels to exclude
        show: If True, display the plot immediately
    
    Returns:
        plotly.graph_objects.Figure object
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        raise ImportError("plotly is required for interactive visualization. Install with: pip install plotly")
    
    # Parse object records grouped by label
    objects_by_label = {}
    unique_robot_positions = set()
    
    for record in object_records:
        spatial_meta = record.get("spatial_metadata", {})
        semantic_meta = record.get("semantic_metadata", {})
        frame_meta = record.get("frame_metadata", {})
        
        label = semantic_meta.get("label", "unknown")
        
        # Skip blacklisted labels
        if label in blacklist_labels:
            continue
        
        # Extract robot position
        robot_position = frame_meta.get("robot_position")
        robot_yaw = frame_meta.get("robot_yaw")
        if robot_position is not None and robot_yaw is not None:
            unique_robot_positions.add((tuple(robot_position), robot_yaw))
        
        # Extract object position
        position_3d = spatial_meta.get("position_3d")
        if position_3d is not None and len(position_3d) >= 3:
            if spatial_meta.get("is_valid_depth", False):
                if label not in objects_by_label:
                    objects_by_label[label] = {
                        'x': [], 'y': [], 'heights': [], 'records': []
                    }
                objects_by_label[label]['x'].append(position_3d[0])
                objects_by_label[label]['y'].append(position_3d[1])
                objects_by_label[label]['heights'].append(position_3d[2])
                objects_by_label[label]['records'].append(record)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add occupancy map overlay if requested
    if overlay_map and world_map is not None:
        occgrid = world_map.occupancy_map
        x_coords = world_map.x_coords
        y_coords = world_map.y_coords
        
        # Create custom colormap: -1=unknown (gray), 0=free (white), 1=occupied (black)
        # Map values: -1 -> 0, 0 -> 1, 1 -> 2 for indexing
        occgrid_normalized = occgrid.copy().astype(float)
        occgrid_normalized[occgrid == -1] = 0  # unknown -> gray
        occgrid_normalized[occgrid == 0] = 1   # free -> white
        occgrid_normalized[occgrid == 1] = 2   # occupied -> black
        
        # Create color scale: gray -> white -> black
        colorscale = [[0, 'gray'], [0.5, 'white'], [1.0, 'black']]
        
        fig.add_trace(go.Heatmap(
            z=occgrid_normalized,
            x=x_coords,
            y=y_coords,
            colorscale=colorscale,
            showscale=False,
            hoverinfo='skip',
            opacity=0.6,
            name='Occupancy Map'
        ))
    
    # Add a trace for each unique label (for interactive legend)
    # Sort by number of occurrences (most common first)
    unique_labels = sorted(objects_by_label.keys(), 
                          key=lambda label: len(objects_by_label[label]['x']), 
                          reverse=True)
    colors = px.colors.qualitative.Set3  # Color palette
    if len(unique_labels) > len(colors):
        # Extend colors if needed
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20')
        colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                  for r, g, b, _ in [cmap(i/len(unique_labels)) for i in range(len(unique_labels))]]
    
    for idx, label in enumerate(unique_labels):
        data = objects_by_label[label]
        
        # Create hover text with label, height, and description
        hover_texts = []
        for x, y, h, record in zip(data['x'], data['y'], data['heights'], data['records']):
            description = record.get("semantic_metadata", {}).get("description", "No description available")
            task_desc = record.get("task_description", "No task description available")
            hover_text = (
                f"<b>Object ID: {record.get('object_id', 'N/A')}</b><br>" +
                f"<b>{label}</b><br>" +
                f"Position: ({x:.2f}, {y:.2f})<br>" +
                f"Height: {h:.2f}m<br>" +
                f"Description: {description}<br>" +
                f"Task Description: {task_desc}"
            )
            hover_texts.append(hover_text)
        
        fig.add_trace(go.Scatter(
            x=data['x'],
            y=data['y'],
            mode='markers+text',
            name=label,
            marker=dict(
                size=10,
                color=data['heights'],
                colorscale='Viridis',
                showscale=(idx == 0),  # Show colorbar only for first trace
                colorbar=dict(
                    title="Height (m)", 
                    len=0.5, 
                    y=0.75,
                    x=-0.12,  # Move to left side (negative values place it to the left of plot)
                    xanchor='right',
                    yanchor='middle'
                ) if idx == 0 else None,
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            text=[label] * len(data['x']),  # Show label on marker
            textposition='top center',
            textfont=dict(size=12),  # Increased from 8 to 12
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_texts,
            legendgroup=label,
            showlegend=True
        ))
    
    # Add robot positions if requested
    if plot_robot and world_map is not None:
        robot_x, robot_y, robot_yaws = [], [], []
        for robot_position, robot_yaw in unique_robot_positions:
            robot_x.append(robot_position[0])
            robot_y.append(robot_position[1])
            robot_yaws.append(robot_yaw)
        
        if robot_x:
            # Robot position markers
            fig.add_trace(go.Scatter(
                x=robot_x,
                y=robot_y,
                mode='markers',
                name='Robot Position',
                marker=dict(
                    size=15,
                    symbol='square',
                    color='blue',
                    line=dict(width=2, color='darkblue')
                ),
                hovertemplate='<b>Robot</b><br>Position: (%{x:.2f}, %{y:.2f})<br>Yaw: %{customdata:.2f}°<extra></extra>',
                customdata=[np.degrees(yaw) for yaw in robot_yaws],
                legendgroup='robot',
                showlegend=True
            ))
            
            # Robot yaw arrows (simplified - just direction indicators)
            # Combine all arrows into a single trace to avoid cluttering legend
            arrow_length = 0.5
            arrow_x, arrow_y = [], []
            for x, y, yaw in zip(robot_x, robot_y, robot_yaws):
                dx = arrow_length * np.cos(yaw)
                dy = arrow_length * np.sin(yaw)
                arrow_x.extend([x, x + dx, None])  # None creates line breaks
                arrow_y.extend([y, y + dy, None])
            
            if arrow_x:
                fig.add_trace(go.Scatter(
                    x=arrow_x,
                    y=arrow_y,
                    mode='lines',
                    name='Robot Direction',
                    line=dict(color='blue', width=2),
                    hoverinfo='skip',
                    legendgroup='robot',
                    showlegend=True
                ))
    
    # Prepare button controls for select/deselect all
    # Identify which trace indices correspond to object labels (not occupancy map or robot)
    object_trace_indices = []
    non_object_trace_names = {'Occupancy Map', 'Robot Position', 'Robot Direction'}
    
    for idx, trace in enumerate(fig.data):
        trace_name = trace.name if hasattr(trace, 'name') else ''
        # Only include object label traces, explicitly exclude occupancy map and robot
        if trace_name in unique_labels and trace_name not in non_object_trace_names:
            object_trace_indices.append(idx)
    
    # Create visibility lists - only affect object traces, keep everything else visible
    def create_visibility_for_object_traces(show_objects):
        """Create visibility list that only toggles object traces, keeps everything else visible."""
        visible = []
        for idx, trace in enumerate(fig.data):
            trace_name = trace.name if hasattr(trace, 'name') else ''
            # Explicitly keep occupancy map and robot traces always visible
            if trace_name in non_object_trace_names or 'Robot' in trace_name:
                visible.append(True)
            elif idx in object_trace_indices:
                # Only toggle object label traces
                visible.append(show_objects)
            else:
                # Default: keep visible (safety fallback)
                visible.append(True)
        return visible
    
    select_all_visible = create_visibility_for_object_traces(True)
    deselect_all_visible = create_visibility_for_object_traces(False)
    
    # Update layout with buttons
    fig.update_layout(
        title='Interactive Top-Down Object Map<br><sub>Click legend items to show/hide labels</sub>',
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        width=1200,
        height=1200,
        hovermode='closest',
        margin=dict(l=100, r=200, t=100, b=50),  # Add left margin for colorbar, right margin for legend
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            itemclick="toggleothers",  # Allow clicking legend items to toggle
            itemdoubleclick="toggle"   # Double-click to isolate
        ),
        plot_bgcolor='white',
        yaxis=dict(scaleanchor="x", scaleratio=1),  # Equal aspect ratio
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=None,  # No button is active by default
                x=0.01,
                xanchor="left",
                y=1.02,
                yanchor="top",
                buttons=list([
                    dict(
                        label="Select All",
                        method="restyle",
                        args=[{"visible": select_all_visible}]
                    ),
                    dict(
                        label="Deselect All",
                        method="restyle",
                        args=[{"visible": deselect_all_visible}]
                    )
                ]),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
        ]
    )
    
    if show:
        fig.show()
    
    return fig


def viz_object_records_with_worldmap_routine():
    """Load and visualize object records with world map from current run outputs."""
    world_map = load_saved_world_map_from_current_run(sample=False)
    object_records = load_saved_object_records_from_current_run(sample=False)
    
    # Use interactive plotly visualization
    plot_object_records_top_down_interactive(
        object_records, 
        world_map, 
        overlay_map=True, 
        plot_robot=True,
        blacklist_labels=["floor", "wall", "ceiling", "cabinet", "couch", "carpet"],
        show=True
    )
    
    # Uncomment below to use matplotlib version instead
    # fig, ax = plt.subplots(figsize=(12, 12))
    # plot_object_records_top_down(ax, object_records, world_map, overlay_map=True, plot_robot=True, show=False, 
    # blacklist_labels=["floor", "wall", "ceiling", "cabinet", "couch", "carpet"])
    # plt.show()

if __name__ == "__main__":
    # viz_worldmap_with_frontier_routine()
    # viz_object_records_with_worldmap_routine()


    # Check collision at specific pose
    world_map = load_saved_world_map_from_current_run(sample=False)
    # robot_pos = (-9.113582611083984, 11.851309776306152)
    # robot_yaw = -1.19185860076
    # 
    # 'x': -9.145586013793945, 'y': 12.12925910949707, 'yaw': -0.3686411658716803,23444

    robot_pos = (-9.145586013793945, 12.12925910949707)
    robot_yaw = -0.3686411658716803
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax = plot_state(ax, world_map, robot_world_coords=robot_pos, robot_yaw=robot_yaw)
    plt.tight_layout()
    plt.show()
    
    # # viz_worldmap_with_frontier_routine()
    # viz_object_records_with_worldmap_routine()



    # # load object records from /home/tiamat_eval/tiamatl_eval_mvp/logs/current_run_outputs nov 25 backup folder and plot it
    # object_records = load_object_records()
    # plot_object_records_top_down(object_records, world_map)
    # plt.show()

    # # load robot position from /home/tiamat_eval/tiamatl_eval_mvp/logs/current_run_outputs nov 25 backup folder and plot it
    # robot_position = load_robot_position()
    # plot_robot_position(robot_position)
    # plt.show()