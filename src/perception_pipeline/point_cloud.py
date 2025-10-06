import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def get_o3d_cam_intrinsic(height, width):
    """Returns the Open3D camera intrinsic object based on image size."""
    fx = width / 2.0
    fy = width / 2.0
    cx = width / 2.0
    cy = height / 2.0
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

def get_organized_point_cloud(depth_np, camera_pose_7d, camera_intrinsics):
    """Projects a depth image into a 3D point cloud in world coordinates."""
    H, W = depth_np.shape
    fx, fy, cx, cy = camera_intrinsics.intrinsic_matrix[0, 0], camera_intrinsics.intrinsic_matrix[1, 1], camera_intrinsics.intrinsic_matrix[0, 2], camera_intrinsics.intrinsic_matrix[1, 2]
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    Z_cam = depth_np
    X_cam = (u - cx) * Z_cam / fx
    Y_cam = (v - cy) * Z_cam / fy
    points_camera_frame = np.stack([X_cam, Y_cam, Z_cam], axis=-1) # (-1) for creae the new last dimention
    
    points_flat = points_camera_frame.reshape(-1, 3) #grid to simple list of points for matrix multiplication \ (H, W, 3) to (H*W, 3)
    quat_wxyz, position_xyz = camera_pose_7d[:4], camera_pose_7d[4:]
    rotation_matrix = R.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix() # w,x,y,z -> x,y,z,w
    
    c2w_transform = np.eye(4) #identity matrix (4x4) to place info for both rotate and move points from camera fram to world frame
    c2w_transform[:3, :3] = rotation_matrix #placed in the top-left corner
    c2w_transform[:3, 3] = position_xyz #placed in the last column

    points_homogeneous = np.hstack((points_flat, np.ones((points_flat.shape[0], 1)))) #add 4th component to every point in the flattened list #(N, 4)
    #c2w_transform
    # [R, R, R, | T]        R rotaion matrix
    # [R, R, R, | T]        T translation vector X Y Z
    # [R, R, R, | T]
    # [0, 0, 0, | 1]     - points_homogeneous (trick)
    points_world_homogeneous = (c2w_transform @ points_homogeneous.T).T #transformation:matrix multipication. result (N,4) 
    
    return points_world_homogeneous[:, :3].reshape(H, W, 3) #select only x,y,z values (w discarding) #reshape to match input image

def calculate_point_cloud_coverage(pcd1, pcd2, voxel_size=0.05):
    """Calculates the coverage of pcd1 by pcd2 using voxel hashing."""
    if len(pcd1) == 0 or len(pcd2) == 0: return 0.0
    
    pcd1_hashed = np.floor(pcd1 / voxel_size).astype(int) #drop the decimal part and cov to int
    pcd2_hashed = np.floor(pcd2 / voxel_size).astype(int)
    
    pcd1_voxels = set(map(tuple, pcd1_hashed))
    pcd2_voxels = set(map(tuple, pcd2_hashed))
    
    if len(pcd1_voxels) == 0: return 0.0
    
    intersection_size = len(pcd1_voxels.intersection(pcd2_voxels))
    return intersection_size / len(pcd1_voxels)


def calculate_point_cloud_coverage_kdtree(pcd1, pcd2, search_radius=0.4):
    """
    Calculates the coverage of pcd1 by pcd2 using a k-d tree nearest neighbor search.

    This function checks what percentage of points in pcd1 have a neighbor
    in pcd2 within the specified search_radius.

    Args:
        pcd1 (np.array): The source point cloud (N, 3).
        pcd2 (np.array): The target point cloud to check against (M, 3).
        search_radius (float): The maximum distance to consider a point covered.

    Returns:
        float: The coverage score (0.0 to 1.0).
    """
    if len(pcd1) == 0 or len(pcd2) == 0:
        return 0.0

    # 1. convert np arrays to o3d point cloud objs to use Open3d lib tool
    pcd1_o3d = o3d.geometry.PointCloud()
    pcd1_o3d.points = o3d.utility.Vector3dVector(pcd1)

    pcd2_o3d = o3d.geometry.PointCloud()
    pcd2_o3d.points = o3d.utility.Vector3dVector(pcd2)

    # 2. build the k-d tree from the second (target) point cloud
    pcd2_tree = o3d.geometry.KDTreeFlann(pcd2_o3d)

    # 3. for each point in pcd1, find its nearest neighbor in pcd2
    inlier_count = 0
    # compare squared distance to avoid using sqrt in a loop
    search_radius_sq = search_radius**2

    for point in pcd1_o3d.points:
        # search for the 1 nearest neighbor (k=1)
        [k, idx, dist2] = pcd2_tree.search_knn_vector_3d(point, 1) #dis2 is squared distance
        
        # check if the squared distance w search radius: calculation square root is faster operation
        if k > 0 and dist2[0] < search_radius_sq:
            inlier_count += 1
            
    # 4. final coverage
    return inlier_count / len(pcd1_o3d.points)