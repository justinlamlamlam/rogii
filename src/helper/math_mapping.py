# math_mapping.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

# Global cache to store preprocessed index data structures for O(1) lookups.
# Key: tuple (id(dataframe), len(dataframe)) -> Value: Preprocessed lookups
_TYPEWELL_CACHE: Dict[Tuple[int, int], Dict[str, Any]] = {}

def _preprocess_typewell(typewell_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Preprocesses the typewell DataFrame to construct lookup tables and spatial indices
    allowing O(1) interpolation and inversion.

    Args:
        typewell_df (pd.DataFrame): DataFrame containing 'TVT' and 'GR' columns.

    Returns:
        dict: Precomputed structures for O(1) execution.
    """
    # Sort by TVT to ensure monotonic sequence for depth logic
    df_sorted = typewell_df.sort_values(by="TVT").copy()
    
    tvt_arr = df_sorted["TVT"].to_numpy(dtype=np.float64)
    gr_arr = df_sorted["GR"].to_numpy(dtype=np.float64)
    
    # 1. Structures for TVT -> GR O(1) Lookup
    # Compute step size for constant spacing check (very common in well logs)
    tvt_diffs = np.diff(tvt_arr)
    is_constant_spacing = False
    step_size = 0.0
    
    if len(tvt_diffs) > 0 and np.allclose(tvt_diffs, tvt_diffs[0], rtol=1e-5):
        is_constant_spacing = True
        step_size = float(tvt_diffs[0])
        
    # Fallback to an interval bucket hash map if spacing is non-constant
    bucket_size = 1.0  # Adjustable resolution for pseudo-O(1) bucket index lookup
    tvt_buckets = {}
    if not is_constant_spacing and len(tvt_arr) > 1:
        min_tvt, max_tvt = tvt_arr[0], tvt_arr[-1]
        for i in range(len(tvt_arr) - 1):
            start_bucket = int(np.floor(tvt_arr[i] / bucket_size))
            end_bucket = int(np.floor(tvt_arr[i + 1] / bucket_size))
            for b in range(start_bucket, end_bucket + 1):
                if b not in tvt_buckets:
                    tvt_buckets[b] = []
                tvt_buckets[b].append(i)

    # 2. Structures for GR -> TVT O(1) Lookup (Inversion)
    # Since GR values are highly oscillatory, we index segment ranges by GR intervals
    gr_min_val, gr_max_val = gr_arr.min(), gr_arr.max()
    gr_interval_span = gr_max_val - gr_min_val
    
    # Use 100 buckets spanning the GR range for pseudo-O(1) retrieval of candidate segments
    num_gr_buckets = 100
    gr_bucket_width = (gr_interval_span / num_gr_buckets) if gr_interval_span > 0 else 1.0
    gr_buckets = [[] for _ in range(num_gr_buckets + 1)]
    
    for i in range(len(gr_arr) - 1):
        gr1, gr2 = gr_arr[i], gr_arr[i + 1]
        seg_min, seg_max = min(gr1, gr2), max(gr1, gr2)
        
        # Identify buckets intersected by segment [seg_min, seg_max]
        idx_start = int(np.floor((seg_min - gr_min_val) / gr_bucket_width)) if gr_bucket_width > 0 else 0
        idx_end = int(np.floor((seg_max - gr_min_val) / gr_bucket_width)) if gr_bucket_width > 0 else 0
        
        # Bound limits to ensure valid index offsets
        idx_start = max(0, min(idx_start, num_gr_buckets))
        idx_end = max(0, min(idx_end, num_gr_buckets))
        
        for b in range(idx_start, idx_end + 1):
            gr_buckets[b].append(i)

    return {
        "tvt_arr": tvt_arr,
        "gr_arr": gr_arr,
        "is_constant_spacing": is_constant_spacing,
        "step_size": step_size,
        "bucket_size": bucket_size,
        "tvt_buckets": tvt_buckets,
        "gr_min_val": gr_min_val,
        "gr_bucket_width": gr_bucket_width,
        "gr_buckets": gr_buckets,
        "num_gr_buckets": num_gr_buckets
    }

def get_precomputed_data(typewell_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns precomputed structures from cache, or populates the cache if not present.

    Args:
        typewell_df (pd.DataFrame): DataFrame containing 'TVT' and 'GR' columns.

    Returns:
        dict: The preprocessed structures containing lookup tables.
    """
    cache_key = (id(typewell_df), len(typewell_df))
    if cache_key not in _TYPEWELL_CACHE:
        _TYPEWELL_CACHE[cache_key] = _preprocess_typewell(typewell_df=typewell_df)
    return _TYPEWELL_CACHE[cache_key]

def tvt_to_gr(tvt: float, typewell_df: pd.DataFrame) -> float:
    """
    Calculates the GR value for a given TVT using linear interpolation 
    based on the piecewise linear function defined in the typewell table.
    Runs in O(1) time using hashed spatial indices.

    Args:
        tvt (float): The Total Vertical Thickness value to map.
        typewell_df (pd.DataFrame): DataFrame containing 'TVT' and 'GR' columns.

    Returns:
        float: The interpolated Gamma Ray value.
    """
    meta = get_precomputed_data(typewell_df=typewell_df)
    tvt_arr = meta["tvt_arr"]
    gr_arr = meta["gr_arr"]
    
    if len(tvt_arr) == 0:
        return np.nan
    if len(tvt_arr) == 1:
        return float(gr_arr[0])
        
    # Extrapolation handling
    if tvt <= tvt_arr[0]:
        return float(gr_arr[0])
    if tvt >= tvt_arr[-1]:
        return float(gr_arr[-1])
        
    idx = -1
    # O(1) Index Identification
    if meta["is_constant_spacing"]:
        # Direct calculation when index has constant grid spacing
        idx = int((tvt - tvt_arr[0]) // meta["step_size"])
        idx = max(0, min(idx, len(tvt_arr) - 2))
    else:
        # Pseudo-O(1) mapping using localized lookup buckets
        b_idx = int(np.floor(tvt / meta["bucket_size"]))
        candidates = meta["tvt_buckets"].get(b_idx, [])
        for c in candidates:
            if tvt_arr[c] <= tvt <= tvt_arr[c + 1]:
                idx = c
                break
        if idx == -1:
            # Fallback to binary search O(log N) only if bucketing boundary misses
            idx = np.searchsorted(tvt_arr, tvt) - 1
            idx = max(0, min(idx, len(tvt_arr) - 2))

    # Perform linear interpolation
    t0, t1 = tvt_arr[idx], tvt_arr[idx + 1]
    g0, g1 = gr_arr[idx], gr_arr[idx + 1]
    
    weight = (tvt - t0) / (t1 - t0)
    return float(g0 + weight * (g1 - g0))

def gr_to_tvt(gr_target: float, typewell_df: pd.DataFrame) -> list:
    """
    Finds all matching TVT values corresponding to a target GR value.
    Runs in O(1) average time by evaluating only candidate segments matching the target GR range.

    Args:
        gr_target (float): The target Gamma Ray value to invert.
        typewell_df (pd.DataFrame): DataFrame containing 'TVT' and 'GR' columns.

    Returns:
        list[float]: A list of all matching TVT candidates.
    """
    meta = get_precomputed_data(typewell_df=typewell_df)
    tvt_arr = meta["tvt_arr"]
    gr_arr = meta["gr_arr"]
    
    if len(tvt_arr) < 2:
        return []
        
    gr_min_val = meta["gr_min_val"]
    gr_bucket_width = meta["gr_bucket_width"]
    num_gr_buckets = meta["num_gr_buckets"]
    
    # Identify target bucket index
    b_idx = int(np.floor((gr_target - gr_min_val) / gr_bucket_width)) if gr_bucket_width > 0 else 0
    if b_idx < 0 or b_idx > num_gr_buckets:
        return [] # target value is strictly outside typewell GR bounds
        
    candidates = meta["gr_buckets"][b_idx]
    results = []
    
    # Process only local candidates mapped to this bucket range
    for idx in candidates:
        g0, g1 = gr_arr[idx], gr_arr[idx + 1]
        t0, t1 = tvt_arr[idx], tvt_arr[idx + 1]
        
        # Check if target is bounded by current segment
        if (g0 <= gr_target <= g1) or (g1 <= gr_target <= g0):
            if g1 == g0:
                # Flat line scenario: Entire interval matches
                results.append(float(t0))
            else:
                weight = (gr_target - g0) / (g1 - g0)
                interpolated_tvt = t0 + weight * (t1 - t0)
                results.append(float(interpolated_tvt))
                
    # Deduplicate matching nodes if any values land exactly on shared segment boundaries
    return sorted(list(set(results)))