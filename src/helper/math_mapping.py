# math_mapping.py

import numpy as np
import pandas as pd

def forward_map(tvt: float, typewell_df: pd.DataFrame) -> float:
    """
    Calculates the GR value for a given TVT using linear interpolation 
    based on the piecewise linear function defined in the typewell table.

    Args:
        tvt (float): The Total Vertical Thickness value to map.
        typewell_df (pd.DataFrame): DataFrame containing 'TVT' and 'GR' columns.

    Returns:
        float: The interpolated Gamma Ray value.
    """
    # Keep original logic: treat as piecewise linear function via interpolation
    gr_interpolated = float(np.interp(
        x=tvt, 
        xp=typewell_df['TVT'].values, 
        fp=typewell_df['GR'].values
    ))
    return gr_interpolated

def inverse_set(gr_target: float, typewell_df: pd.DataFrame) -> list:
    """
    Scans the typewell table to find all TVT values that yield a specific GR target.
    Accounts for non-monotonicity by checking all linear segments.

    Args:
        gr_target (float): The target Gamma Ray value to invert.
        typewell_df (pd.DataFrame): DataFrame containing 'TVT' and 'GR' columns.

    Returns:
        list[float]: A list of all matching TVT candidates.
    """
    tvt_candidates = []
    
    tvt_nodes = typewell_df['TVT'].values
    gr_nodes = typewell_df['GR'].values
    
    # Iterate through each segment to find intersections
    for i in range(len(tvt_nodes) - 1):
        tvt1, tvt2 = tvt_nodes[i], tvt_nodes[i+1]
        gr1, gr2 = gr_nodes[i], gr_nodes[i+1]
        
        # Check if gr_target lies within the current segment interval
        if min(gr1, gr2) <= gr_target <= max(gr1, gr2):
            # Avoid division by zero for flat segments matching target
            if gr1 == gr2:
                if gr1 == gr_target:
                    tvt_candidates.append(float(tvt1))
                continue
            
            # Linear interpolation formula solved for TVT
            t = (gr_target - gr1) / (gr2 - gr1)
            tvt_val = tvt1 + t * (tvt2 - tvt1)
            tvt_candidates.append(float(tvt_val))
            
    # Remove duplicate boundaries if an intersection falls exactly on a node
    return sorted(list(set(tvt_candidates)))

def get_discretized_tvt_grid(typewell_df: pd.DataFrame, step: float = 0.01) -> np.ndarray:
    """
    Discretizes the TVT axis into a fine grid between the minimum and maximum 
    TVT values present in the typewell data.

    Args:
        typewell_df (pd.DataFrame): DataFrame containing 'TVT' and 'GR' columns.
        step (float): The step size for the grid discretization. Defaults to 0.01.

    Returns:
        np.ndarray: A 1D array of discretized TVT values.
    """
    min_tvt = typewell_df['TVT'].min()
    max_tvt = typewell_df['TVT'].max()
    
    # Generate grid space including the upper bound
    tvt_grid = np.arange(start=min_tvt, stop=max_tvt + step, step=step)
    return tvt_grid