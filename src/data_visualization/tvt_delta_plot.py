import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("/home/justinlamlamlam/python/kaggle/rogii/src")
import helper

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
    # Save the figure
    filename = "data/graphs/tvt_delta/plot.png"
    plt.savefig(fname=filename)
    plt.close() # Close plot to free memory
    print(f"Plot saved as {filename}")

import matplotlib.pyplot as plt
import numpy as np

def plot_tvt_distribution_filtered(raw_data: dict) -> None:
    """
    Plots the distribution of "TVT" changes as a bar chart with a bin width of 0.01,
    limited to the range between -1 and 1.

    Args:
        raw_data (dict): Dictionary where keys are well IDs and values 
                         are tuples of (horizontal_well_df, typewell_df).
    Returns:
        None: Displays a matplotlib plot.
    """
    all_filtered_changes = []

    for prefix, data_tuple in raw_data.items():
        horizontal_well = data_tuple[0]
        
        if "TVT" in horizontal_well.columns and "TVT_input" in horizontal_well.columns:
            # Calculate consecutive change
            tvt_change = horizontal_well["TVT"].diff()
            
            # Filter delta TVT where TVT_input is NaN
            filtered_changes = tvt_change[horizontal_well["TVT_input"].isna()].dropna()
            
            # Filter for values only within -1 and 1 range
            bounded_changes = filtered_changes[(filtered_changes >= -1) & (filtered_changes <= 1)]
            all_filtered_changes.extend(bounded_changes.tolist())

    if not all_filtered_changes:
        print("No data available within the range [-1, 1] for distribution plotting.")
        return

    # Determine bins with 0.01 separation within the fixed range
    bins = np.arange(start=-1, stop=1 + 0.01, step=0.01)

    plt.figure(figsize=(12, 6))
    plt.hist(x=all_filtered_changes, bins=bins, color='royalblue', edgecolor='black', alpha=0.7)
    
    plt.title(label="Distribution of Delta TVT (Range: -1 to 1, Bin Width: 0.01)")
    plt.xlabel(xlabel="Delta TVT")
    plt.ylabel(ylabel="Frequency")
    plt.xlim(left=-1, right=1)
    plt.grid(visible=True, axis='y', linestyle='--', alpha=0.6)
    # plt.show()

    # Save the figure
    filename = "data/graphs/tvt_delta/distribution.png"
    plt.savefig(fname=filename)
    plt.close() # Close plot to free memory
    print(f"Plot saved as {filename}")

def print_tvt_outliers(raw_data: dict, lower_bound: float = -1.0, upper_bound: float = 1.0) -> None:
    """
    Identifies and prints rows where the change in "TVT" (when "TVT_input" is NaN)
    falls outside the specified range.

    Args:
        raw_data (dict): Dictionary where keys are well IDs and values 
                         are tuples of (horizontal_well_df, typewell_df).
        lower_bound (float): The lower threshold for outliers.
        upper_bound (float): The upper threshold for outliers.

    Returns:
        None: Prints outlier information to the console.
    """
    print(f"Searching for TVT delta outliers outside [{lower_bound}, {upper_bound}]...")
    print("-" * 50)
    
    found_outliers = False

    for prefix, data_tuple in raw_data.items():
        horizontal_well = data_tuple[0]
        
        if "TVT" in horizontal_well.columns and "TVT_input" in horizontal_well.columns:
            # Calculate consecutive change
            # We keep the index to track where the outlier occurs
            tvt_change = horizontal_well["TVT"].diff()
            
            # Create a mask for rows where TVT_input is NaN
            is_na_input = horizontal_well["TVT_input"].isna()
            
            # Identify outliers within the NaN input subset
            outlier_mask = (is_na_input) & ((tvt_change < lower_bound) | (tvt_change > upper_bound))
            
            outliers = tvt_change[outlier_mask]
            
            if not outliers.empty:
                found_outliers = True
                for index, value in outliers.items():
                    print(f"Well ID: {prefix} | Index: {index} | Delta TVT: {value:.4f}")

    if not found_outliers:
        print("No outliers found in the specified range.")

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
    plot_tvt_distribution_filtered(raw_data=raw_data)
    print_tvt_outliers(raw_data=raw_data)

