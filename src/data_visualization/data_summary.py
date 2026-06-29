# data_summary.py

import os
import sys
import argparse

current_script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_script_dir, "..", "..", "src"))
print(src_path)
if src_path not in sys.path:
    sys.path.append(src_path)

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from helper.data_handling import get_raw_data

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

def calculate_tvt_separation_metrics(raw_data: dict) -> tuple:
    """
    Calculates the minimum, mean, and maximum of the TVT separation (max TVT - min TVT)
    computed after the last known TVT_input point for each well.

    Args:
        raw_data (dict): Dictionary mapping well_id (str) to a tuple of (horizontal_well, typewell) DataFrames.

    Returns:
        tuple[float, float, float]: A tuple containing:
            - min_separation (float): Minimum TVT separation across all valid wells.
            - mean_separation (float): Mean TVT separation across all valid wells.
            - max_separation (float): Maximum TVT separation across all valid wells.
    """
    separations = []

    for prefix, (horizontal_well, typewell) in raw_data.items():
        if "TVT_input" in horizontal_well.columns and "TVT" in horizontal_well.columns:
            valid_indices = horizontal_well["TVT_input"].notna()
            if valid_indices.any():
                last_pos = np.max(np.where(valid_indices)[0])
                after_df = horizontal_well.iloc[last_pos + 1:]
                if not after_df.empty and not after_df["TVT"].dropna().empty:
                    tvt_after = after_df["TVT"].dropna()
                    sep = float(tvt_after.max() - tvt_after.min())
                    separations.append(sep)

    min_separation = float(np.min(separations)) if separations else 0.0
    mean_separation = float(np.mean(separations)) if separations else 0.0
    max_separation = float(np.max(separations)) if separations else 0.0

    return min_separation, mean_separation, max_separation

def plot_tvt_separation_distribution(raw_data: dict, save_dir: str) -> None:
    """
    Plots the distribution of TVT separation values across all valid wells as a scatter plot
    and saves the figure to the specified directory.

    Args:
        raw_data (dict): Dictionary mapping well_id (str) to a tuple of (horizontal_well, typewell) DataFrames.
        save_dir (str): Path to the directory where the plot should be saved.

    Returns:
        None
    """
    separations = []

    for prefix, (horizontal_well, typewell) in raw_data.items():
        if "TVT_input" in horizontal_well.columns and "TVT" in horizontal_well.columns:
            valid_indices = horizontal_well["TVT_input"].notna()
            if valid_indices.any():
                last_pos = np.max(np.where(valid_indices)[0])
                after_df = horizontal_well.iloc[last_pos + 1:]
                if not after_df.empty and not after_df["TVT"].dropna().empty:
                    tvt_after = after_df["TVT"].dropna()
                    sep = float(tvt_after.max() - tvt_after.min())
                    separations.append(sep)

    if not separations:
        print("No valid TVT separation data found to plot.")
        return

    os.makedirs(save_dir, exist_ok=True)
    
    x_indices = list(range(len(separations)))
    
    plt.scatter(x_indices, separations, alpha=0.7, edgecolors="k")
    plt.title("Distribution of TVT Separation After Last TVT_input Point")
    plt.xlabel("Well Index")
    plt.ylabel("TVT Separation (Max - Min)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "tvt_separation_scatter.png")
    plt.savefig(save_path)
    plt.close()

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_tvt_separation_distribution_bar(raw_data: dict, save_dir: str) -> None:
    """
    Plots the distribution of TVT separation values across all valid wells as a bar chart (histogram)
    with a bin size of 2 and saves the figure to the specified directory.

    Args:
        raw_data (dict): Dictionary mapping well_id (str) to a tuple of (horizontal_well, typewell) DataFrames.
        save_dir (str): Path to the directory where the plot should be saved.

    Returns:
        None
    """
    separations = []

    for prefix, (horizontal_well, typewell) in raw_data.items():
        if "TVT_input" in horizontal_well.columns and "TVT" in horizontal_well.columns:
            valid_indices = horizontal_well["TVT_input"].notna()
            if valid_indices.any():
                last_pos = np.max(np.where(valid_indices)[0])
                after_df = horizontal_well.iloc[last_pos + 1:]
                if not after_df.empty and not after_df["TVT"].dropna().empty:
                    tvt_after = after_df["TVT"].dropna()
                    sep = float(tvt_after.max() - tvt_after.min())
                    separations.append(sep)

    if not separations:
        print("No valid TVT separation data found to plot.")
        return

    os.makedirs(save_dir, exist_ok=True)
    
    # Define bins with a step size of 2 based on data range
    min_val = np.floor(np.min(separations))
    max_val = np.ceil(np.max(separations))
    bins = np.arange(start=min_val, stop=max_val + 2, step=2)
    
    # Plot distribution as a bar chart histogram
    plt.hist(separations, bins=bins, alpha=0.7, edgecolor="k")
    plt.title("Distribution of TVT Separation After Last TVT_input Point")
    plt.xlabel("TVT Separation (Bin Size = 2)")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "tvt_separation_scatter_bar.png")
    plt.savefig(save_path)
    plt.close()

# Example execution template
if __name__ == "__main__":
    # Define directory location
    data_directory = "/home/justinlamlamlam/python/kaggle/rogii/data/raw/train" 
    
    # Load raw data
    data = get_raw_data(data_loc=data_directory)
    
    # Calculate metrics
    mean_len, mean_gr_diff, mean_tw_rows, max_tvt_difference = calculate_well_metrics(raw_data=data)
    min_sep, mean_sep, max_sep = calculate_tvt_separation_metrics(raw_data=data)
    
    # Print results
    print(f"Mean sequence length of horizontal well: {mean_len:.2f}")
    print(f"Mean difference between max GR difference in horizontal well: {mean_gr_diff:.2f}")
    print(f"Mean rows of typewell: {mean_tw_rows:.2f}")
    print(f"Max TVT difference after the last TVT_input point: {max_tvt_difference:.2f}")
    print(f"Min TVT separation after the point: {min_sep:.2f}")
    print(f"Mean TVT separation after the point: {mean_sep:.2f}")
    print(f"Max TVT separation after the point: {max_sep:.2f}")

    # Plot and save distribution
    plot_tvt_separation_distribution(raw_data=data, save_dir="data/graphs/data_summary/")
    plot_tvt_separation_distribution_bar(raw_data=data, save_dir="data/graphs/data_summary/")