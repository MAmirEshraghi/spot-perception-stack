import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import torch
import time
from typing import List


def calculate_pd_coverage_hash_voxel(pcd1, pcd2, voxel_size=0.05):
    """Calculates the coverage of pcd1 by pcd2 using voxel hashing."""
    if len(pcd1) == 0 or len(pcd2) == 0: return 0.0
    
    pcd1_hashed = np.floor(pcd1 / voxel_size).astype(int) #drop the decimal part and cov to int
    pcd2_hashed = np.floor(pcd2 / voxel_size).astype(int)
    
    pcd1_voxels = set(map(tuple, pcd1_hashed)) #???? #TODO
    pcd2_voxels = set(map(tuple, pcd2_hashed))
    
    if len(pcd1_voxels) == 0: return 0.0
    
    intersection_size = len(pcd1_voxels.intersection(pcd2_voxels))
    return intersection_size / len(pcd1_voxels)

def calculate_pd_coverage_kdtree(pcd1, pcd2, search_radius=0.4):
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

# Aayam:
def baseline_batch_coverage(query_pcd, pcd_list, voxel_size=0.05):
    """Baseline: Loop through each point cloud individually."""
    device = query_pcd.device
    coverages = []
    for pcd in pcd_list:
        coverage = calculate_point_cloud_coverage_torch(query_pcd, pcd, voxel_size)
        coverages.append(coverage)
    return torch.tensor(coverages, device=device)
def calculate_point_cloud_coverage_torch(pcd1, pcd2, voxel_size=0.05):
    """Original function adapted for torch tensors."""
    if len(pcd1) == 0 or len(pcd2) == 0:
        return 0.0
    
    pcd1_hashed = torch.floor(pcd1 / voxel_size).int()
    pcd2_hashed = torch.floor(pcd2 / voxel_size).int()
    
    pcd1_voxels = set(map(tuple, pcd1_hashed.cpu().numpy()))
    pcd2_voxels = set(map(tuple, pcd2_hashed.cpu().numpy()))
    
    if len(pcd1_voxels) == 0:
        return 0.0
    
    intersection_size = len(pcd1_voxels.intersection(pcd2_voxels))
    return intersection_size / len(pcd1_voxels)
def mega_optimized_batch_coverage(query_pcd, pcd_list, voxel_size=0.05, return_counts = False):
    """
    Fully batched with sparse operations - no loops at all.
    Uses scatter operations for aggregation.
    """
    device = query_pcd.device
    
    if len(query_pcd) == 0:
        return torch.zeros(len(pcd_list), device=device)

    def _compute_point_cloud_hashes(points, voxel_size, P = 73856093, unique = False):
        # Hash query
        voxel_points = torch.floor(points / voxel_size).long()
        unique_voxel_points = torch.unique(voxel_points, dim=0) if unique else voxel_points

        if len(unique_voxel_points) == 0:
            return None, 0
        else:
            point_cloud_hash = (unique_voxel_points[:, 0] + 
                        unique_voxel_points[:, 1] * P + 
                        unique_voxel_points[:, 2] * (P * P))
            return point_cloud_hash, len(unique_voxel_points)

    
    # Build concatenated point cloud with cloud_id for each point
    def _build_concatenated_point_cloud(pcd_list):
        all_pcds, cloud_ids = [], []
        
        for i, pcd in enumerate(pcd_list):
            if len(pcd) > 0:
                all_pcds.append(pcd)
                cloud_ids.append(torch.full((len(pcd),), i, device=device, dtype=torch.long))
                
        if len(all_pcds) == 0:
            return torch.zeros(len(pcd_list), device=device)
        
        all_points = torch.cat(all_pcds, dim=0)
        all_cloud_ids = torch.cat(cloud_ids, dim=0)
        return all_points, all_cloud_ids

    query_hash, query_size = _compute_point_cloud_hashes(query_pcd, voxel_size, unique = True)
    all_points, all_cloud_ids = _build_concatenated_point_cloud(pcd_list)
    all_points_hashed, all_points_size = _compute_point_cloud_hashes(all_points, voxel_size, unique = False)

    if all_points_hashed is None or all_points_size == 0:
        return torch.zeros(len(pcd_list), device=device)
    if query_hash is None or query_size == 0:
        return torch.zeros(len(pcd_list), device=device)  # query point cloud is empty, so return all zeros, no coverage. 
    
    # Find matches with query
    matches = torch.isin(all_points_hashed, query_hash)
    
    # Only keep matching points
    matched_hashes = all_points_hashed[matches]
    matched_cloud_ids = all_cloud_ids[matches]
    
    # Create unique (cloud_id, voxel_hash) pairs
    # This effectively does "unique per cloud"
    combined = torch.stack([matched_cloud_ids, matched_hashes], dim=1)
    unique_pairs = torch.unique(combined, dim=0)
    
    # Count unique voxels per cloud
    unique_cloud_ids = unique_pairs[:, 0]
    counts = torch.bincount(unique_cloud_ids, minlength=len(pcd_list))
    
    # Coverage = matching_voxels / query_voxels
    coverages = counts.float() / query_size
    
    if return_counts:
        return counts.float(), coverages
    else:
        return coverages


def mega_optimized_query_batch_coverage(query_pcd_list: List[torch.Tensor],
                                        pcd_list: List[torch.Tensor],
                                        voxel_size: float = 0.05) -> torch.Tensor:
    """
    Batched coverage for many queries against many reference point clouds.
    Structure mirrors mega_optimized_batch_coverage with small helpers.

    Returns a (num_queries, num_clouds) tensor.
    Coverage(query_i, cloud_j) = (# unique voxels in query_i present in cloud_j) / (# unique voxels in query_i)
    """
    num_queries = len(query_pcd_list)
    num_clouds = len(pcd_list)

    # Determine device from any non-empty tensor, default to CPU
    device = None
    for q in query_pcd_list:
        if q is not None and q.numel() > 0:
            device = q.device
            break
    if device is None:
        for r in pcd_list:
            if r is not None and r.numel() > 0:
                device = r.device
                break
    if device is None:
        device = torch.device('cpu')

    if num_queries == 0 or num_clouds == 0:
        return torch.zeros((num_queries, num_clouds), device=device, dtype=torch.float32)

    P = 73856093

    def _compute_point_cloud_hashes(points, vs, unique=True):
        if points is None or points.numel() == 0:
            return None, 0
        voxel_points = torch.floor(points / vs).long()
        unique_voxel_points = torch.unique(voxel_points, dim=0) if unique else voxel_points
        if len(unique_voxel_points) == 0:
            return None, 0
        pc_hash = (unique_voxel_points[:, 0] +
                   unique_voxel_points[:, 1] * P +
                   unique_voxel_points[:, 2] * (P * P))
        return pc_hash, len(unique_voxel_points)

    def _build_concatenated_point_cloud(pcd_list_inner):
        all_pcds_inner, cloud_ids_inner = [], []
        
        for i, pcd in enumerate(pcd_list_inner):
            if pcd is not None and pcd.numel() > 0:
                all_pcds_inner.append(pcd)
                cloud_ids_inner.append(torch.full((len(pcd),), i, device=device, dtype=torch.long))
        
        if len(all_pcds_inner) == 0:
            return None, None
        
        return torch.cat(all_pcds_inner, dim=0), torch.cat(cloud_ids_inner, dim=0)

    def _compute_query_hashes_and_union(queries, vs):
        per_query_hashes = []
        per_query_sizes = torch.zeros(num_queries, device=device, dtype=torch.long)
        for qi, q in enumerate(queries):
            h, sz = _compute_point_cloud_hashes(q, vs, unique=True)
            per_query_hashes.append(h)
            per_query_sizes[qi] = sz
        if int((per_query_sizes > 0).sum().item()) == 0:
            return per_query_hashes, per_query_sizes, None
        union_list = [h for h in per_query_hashes if h is not None and h.numel() > 0]
        if not union_list:
            return per_query_hashes, per_query_sizes, None
        return per_query_hashes, per_query_sizes, torch.unique(torch.cat(union_list, dim=0))

    def _filter_and_dedup_ref_pairs(all_points_inner, all_cloud_ids_inner, query_union_inner, vs):
        ref_hashes, _ = _compute_point_cloud_hashes(all_points_inner, vs, unique=False)
        if ref_hashes is None:
            return None
        ref_match = torch.isin(ref_hashes, query_union_inner)
        if not ref_match.any():
            return None
        matched_hashes_inner = ref_hashes[ref_match]
        matched_cloud_ids_inner = all_cloud_ids_inner[ref_match]
        combined_inner = torch.stack([matched_cloud_ids_inner, matched_hashes_inner], dim=1)
        return torch.unique(combined_inner, dim=0)

    # 1) Query hashes and union
    query_hashes, query_sizes, query_union = _compute_query_hashes_and_union(query_pcd_list, voxel_size)
    if query_union is None or query_union.numel() == 0:
        return torch.zeros((num_queries, num_clouds), device=device, dtype=torch.float32)

    # 2) Concatenate refs and pre-filter/dedup once
    all_points, all_cloud_ids = _build_concatenated_point_cloud(pcd_list)
    if all_points is None:
        return torch.zeros((num_queries, num_clouds), device=device, dtype=torch.float32)

    unique_pairs = _filter_and_dedup_ref_pairs(all_points, all_cloud_ids, query_union, voxel_size)
    if unique_pairs is None or unique_pairs.numel() == 0:
        return torch.zeros((num_queries, num_clouds), device=device, dtype=torch.float32)

    # 3) Per-query coverage via masking + bincount
    coverages = torch.zeros((num_queries, num_clouds), device=device, dtype=torch.float32)
    unique_pair_clouds = unique_pairs[:, 0]
    unique_pair_hashes = unique_pairs[:, 1]

    for qi, q_hash in enumerate(query_hashes):
        if q_hash is None or query_sizes[qi] == 0:
            continue
        mask = torch.isin(unique_pair_hashes, q_hash)
        if mask.any():
            counts = torch.bincount(unique_pair_clouds[mask], minlength=num_clouds)
            coverages[qi] = counts.to(torch.float32) / query_sizes[qi].to(torch.float32)

    return coverages
