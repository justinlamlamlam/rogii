# data_summary.py

import numpy as np
import pandas as pd
import os
from helper.data_handling import get_raw_data

# data_summary.py
import numpy as np

def calculate_well_metrics(raw_data: dict) -> tuple:
    """
    Calculates the mean sequence length of horizontal wells, the mean of the max GR difference 
    within each horizontal well, the mean number of rows in typewells, and the global maximum 
    TVT difference after the last known TVT_input point.

    Args:
        raw_data (dict): Dictionary mapping well_id (str) to a tuple of (horizontal_well, typewell) DataFrames.

    Returns:
        tuple[float, float, float, float]: A tuple containing:
            - mean_seq_length (float): Average row count of horizontal wells.
            - mean_max_gr_diff (float): Average of (max(GR) - min(GR)) for horizontal wells.
            - mean_typewell_rows (float): Average row count of typewells.
            - max_tvt_difference (float): Maximum absolute difference between the TVT of the last point 
              of TVT_input and any TVT after this point across all wells.
    """
    horizontal_lengths = []
    gr_max_differences = []
    typewell_lengths = []
    tvt_differences = []
    
    for prefix, (horizontal_well, typewell) in raw_data.items():
        # 1. Sequence length of horizontal well
        horizontal_lengths.append(len(horizontal_well))
        
        # 2. Max GR difference in horizontal well (max - min)
        if "GR" in horizontal_well.columns and not horizontal_well["GR"].dropna().empty:
            gr_min = horizontal_well["GR"].min()
            gr_max = horizontal_well["GR"].max()
            gr_max_differences.append(gr_max - gr_min)
        
        # 3. Rows of typewell
        typewell_lengths.append(len(typewell))
        
        # 4. Max TVT difference after the last TVT_input point
        if "TVT_input" in horizontal_well.columns and "TVT" in horizontal_well.columns:
            valid_indices = horizontal_well["TVT_input"].notna()
            if valid_indices.any():
                last_pos = np.max(np.where(valid_indices)[0])
                after_df = horizontal_well.iloc[last_pos + 1:]
                if not after_df.empty and not after_df["TVT"].dropna().empty:
                    last_known_tvt = horizontal_well["TVT"].iloc[last_pos]
                    max_diff = float((after_df["TVT"] - last_known_tvt).abs().max())
                    tvt_differences.append(max_diff)
        
    # Calculate means
    mean_seq_length = float(np.mean(horizontal_lengths)) if horizontal_lengths else 0.0
    mean_max_gr_diff = float(np.mean(gr_max_differences)) if gr_max_differences else 0.0
    mean_typewell_rows = float(np.mean(typewell_lengths)) if typewell_lengths else 0.0
    max_tvt_difference = float(np.max(tvt_differences)) if tvt_differences else 0.0
    
    return mean_seq_length, mean_max_gr_diff, mean_typewell_rows, max_tvt_difference

# Example execution template
if __name__ == "__main__":
    # Define directory location
    data_directory = "/home/justinlamlamlam/python/kaggle/rogii/data/raw/train" 
    
    # Load raw data
    data = get_raw_data(data_loc=data_directory)
    
    # Calculate metrics
    mean_len, mean_gr_diff, mean_tw_rows, max_tvt_difference = calculate_well_metrics(raw_data=data)
    
    # Print results
    print(f"Mean sequence length of horizontal well: {mean_len:.2f}")
    print(f"Mean difference between max GR difference in horizontal well: {mean_gr_diff:.2f}")
    print(f"Mean rows of typewell: {mean_tw_rows:.2f}")
    print(f"Max TVT difference after the last TVT_input point: {max_tvt_difference:.2f}")