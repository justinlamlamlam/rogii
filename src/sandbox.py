import helper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_tvt_change_stats(raw_data: dict) -> dict:
    """
    Calculates the max, min, mean, and median of the change in "TVT" 
    across all horizontal wells in the raw_data dictionary.

    Args:
        raw_data (dict): Dictionary where keys are well IDs and values 
                         are tuples of (horizontal_well_df, typewell_df).

    Returns:
        dict: A dictionary containing 'max', 'min', 'mean', and 'median' of the TVT changes.
    """
    all_changes = []

    for prefix, data_tuple in raw_data.items():
        horizontal_well = data_tuple[0]
        
        if "TVT" in horizontal_well.columns:
            # Calculate the difference between consecutive rows
            # diff() calculates: row[n] - row[n-1]
            changes = horizontal_well["TVT"].diff().dropna()
            all_changes.extend(changes.tolist())

    if not all_changes:
        return {"max": None, "min": None, "mean": None, "median": None}

    stats = {
        "max": np.max(all_changes),
        "min": np.min(all_changes),
        "mean": np.mean(all_changes),
        "median": np.median(all_changes)
    }

    return stats

def plot_tvt_changes(raw_data: dict) -> None:
    """
    Plots the change in "TVT" as a line plot for the first 20 rows where "TVT_input" is NaN.

    Args:
        raw_data (dict): Dictionary where keys are well IDs and values 
                         are tuples of (horizontal_well_df, typewell_df).
    Returns:
        None: Displays a matplotlib plot.
    """
    plt.figure(figsize=(12, 6))
    
    for prefix, data_tuple in raw_data.items():
        horizontal_well = data_tuple[0]
        
        if "TVT" in horizontal_well.columns and "TVT_input" in horizontal_well.columns:
            # Calculate consecutive change
            tvt_change = horizontal_well["TVT"].diff()
            
            # Filter delta TVT where TVT_input is NaN and select top 20
            filtered_changes = tvt_change[horizontal_well["TVT_input"].isna()].head(20)
            
            if not filtered_changes.empty:
                # Plotting as a line plot
                plt.plot(filtered_changes.index, filtered_changes.values, label=f"Well {prefix}", alpha=0.7)

    plt.title(label="Top 20 Delta TVT (where TVT_input is NaN)")
    plt.xlabel(xlabel="DataFrame Index")
    plt.ylabel(ylabel="Delta TVT")
    plt.legend()
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    plt.show()

def find_well_and_index_with_large_tvt_drop(raw_data: dict) -> None:
    """
    Identifies the well ID and the specific row indices where the "TVT" change is less than -100, 
    then prints the results.

    Args:
        raw_data (dict): Dictionary where keys are well IDs and values 
                         are tuples of (horizontal_well_df, typewell_df).

    Returns:
        None: Prints the well ID and index if found.
    """
    for prefix, data_tuple in raw_data.items():
        horizontal_well = data_tuple[0]
        
        if "TVT" in horizontal_well.columns:
            # Calculate the difference between consecutive rows
            tvt_change = horizontal_well["TVT"].diff()
            
            # Find indices where the condition is met
            # Using .index[condition] to get the specific row labels
            problem_indices = tvt_change[tvt_change < -100].index.tolist()
            
            if problem_indices:
                print(f"Well ID: {prefix}")
                print(f"Problem Indices: {problem_indices}")

if __name__ == "__main__":
    # Load data
    data_path = "/home/justinlamlamlam/python/kaggle/rogii/data/raw/train"
    raw_data = helper.data_handling.get_raw_data(data_loc=data_path)
    tvt_stats = calculate_tvt_change_stats(raw_data=raw_data)
    print(tvt_stats)
    # find_well_and_index_with_large_tvt_drop(raw_data=raw_data)
    plot_tvt_changes(raw_data=raw_data)

