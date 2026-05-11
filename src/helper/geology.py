# geology.py

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def build_well_spatial_index(*, raw_data: dict) -> dict:
    """
    Cleans, normalizes, and builds a spatial index for well data.
    
    Returns:
        dict: A bundle containing the tree, targets, and normalization parameters.
    """
    all_coords = []
    all_targets = []

    for well_id, (hw_df, _) in raw_data.items():
        # 1. Drop rows where TVT_input is NaN
        clean_df = hw_df.dropna(subset=['TVT_input'])
        
        if clean_df.empty:
            continue

        all_coords.append(clean_df[['X', 'Y', 'Z']].to_numpy())
        all_targets.append(clean_df['TVT_input'].to_numpy())

    # Combine all wells
    combined_coords = np.vstack(all_coords)
    combined_targets = np.concatenate(all_targets)

    # 2. Calculate Normalization Parameters (Min-Max Scaling)
    # We store these to scale query points later
    mins = combined_coords.min(axis=0)
    maxs = combined_coords.max(axis=0)
    
    # Avoid division by zero if a dimension is flat
    ranges = np.where((maxs - mins) == 0, 1, maxs - mins)

    normalized_coords = (combined_coords - mins) / ranges

    # 3. Build Tree
    tree = cKDTree(data=normalized_coords)
    
    return {
        "tree": tree,
        "targets": combined_targets,
        "mins": mins,
        "ranges": ranges
    }

def lookup_closest_tvt(*, query_points: np.ndarray, index_bundle: dict) -> np.ndarray:
    """
    Normalizes query points and finds the nearest TVT value.

    Args:
        query_points: [n, 3] numpy array.
        index_bundle: The dictionary returned by build_well_spatial_index.

    Returns:
        [n] numpy array of the closest TVT_input values.
    """
    # Extract bundle components
    tree = index_bundle["tree"]
    targets = index_bundle["targets"]
    mins = index_bundle["mins"]
    ranges = index_bundle["ranges"]

    # 1. Apply the SAME normalization to query points
    # (Point - Reference_Min) / Reference_Range
    norm_queries = (query_points - mins) / ranges

    # 2. Perform batched query
    _, indices = tree.query(x=norm_queries, k=1, workers=-1)
    
    return targets[indices]

def get_n_nearest_neighbors(*, query_points: np.ndarray, index_bundle: dict, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds the N closest points for each query location.

    Args:
        query_points: [n, 3] array of raw coordinates.
        index_bundle: Dictionary containing the tree and normalization params.
        k: Number of nearest neighbors to retrieve.

    Returns:
        neighbor_targets: [n, k] array of TVT values.
        distances: [n, k] array of Euclidean distances (in normalized space).
    """
    tree = index_bundle["tree"]
    targets = index_bundle["targets"]
    
    # Normalize query points using precalculated params
    norm_queries = (query_points - index_bundle["mins"]) / index_bundle["ranges"]

    # query returns: (distances, indices)
    # distances are [n, k], indices are [n, k]
    dists, indices = tree.query(x=norm_queries, k=k, workers=-1)

    # Map indices to TVT values
    # result shape: [n, k]
    neighbor_targets = targets[indices]
    
    return neighbor_targets, dists

def calculate_weighted_tvt(*, neighbor_targets: np.ndarray, distances: np.ndarray, power: float = 2.0) -> np.ndarray:
    """
    Calculates the weighted mean of targets based on inverse distance.

    Args:
        neighbor_targets: [n, k] array of TVT values.
        distances: [n, k] array of distances.
        power: The power of the inverse distance (usually 2 for IDW).

    Returns:
        [n] array of weighted mean TVT values.
    """
    # 1. Calculate weights: 1 / (distance^power)
    # Add a tiny epsilon to avoid division by zero
    eps = 1e-10
    weights = 1.0 / (np.power(distances, power) + eps)

    # 2. Weighted sum of targets
    weighted_sum = np.sum(neighbor_targets * weights, axis=1)

    # 3. Sum of weights for normalization
    sum_of_weights = np.sum(weights, axis=1)

    # Return the weighted average
    return weighted_sum / sum_of_weights