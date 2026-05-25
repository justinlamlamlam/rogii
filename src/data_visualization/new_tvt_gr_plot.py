import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
from typing import Dict, Tuple

# Assuming paths are already setup in your environment
sys.path.append("/home/justinlamlamlam/python/kaggle/rogii/src")
import helper
from helper.global_optimization import GlobalPathOptimizer

def plot_well_data_with_optimization(*, raw_data: dict) -> None:
    """
    Generates and saves a figure with five subplots per well prefix.
    Calculates the optimized TVT path and plots it against the original TVT.

    Args:
        raw_data (dict): Dictionary where keys are well prefixes (str) and values are 
                         tuples of (horizontal_well_df, typewell_df).
    
    Returns:
        None
    """
    os.makedirs("data/graphs", exist_ok=True)
    
    # Process the first 1 items
    for i, (prefix, (horizontal_well, typewell)) in enumerate(raw_data.items()):
        if i >= 1:
            break
            
        # Filter horizontal well data where 'TVT_input' is NaN
        horizontal_filtered = horizontal_well[horizontal_well['TVT_input'].isna()].copy()
        
        if horizontal_filtered.empty:
            print(f"Skipping {prefix}: No missing TVT data to optimize.")
            continue
            
        # 1. Extract Target Sequence and Run Optimization
        gr_sequence = horizontal_filtered["GR"].values
        
        print(f"Optimizing TVT profile for well: {prefix} ({len(gr_sequence)} points)...")
        optimizer = GlobalPathOptimizer(
            gr_sequence=gr_sequence,
            typewell_df=typewell,
            tvt_discretization_step=0.1,  # Adjusted to 0.05 for a balanced performance/speed trade-off
            max_tvt_change_slight=0.1,
            jump_penalty=4.0
        )
        
        optimized_tvt, min_error = optimizer.run()
        horizontal_filtered["TVT_optimized"] = optimized_tvt
        print(f"Optimization successful for {prefix}. Final Squared Error: {min_error:.4f}")

        # 2. Setup Visualization (Expanded to 5 subplots)
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(30, 6))
        
        # Subplot 1: Horizontal Well Only (Filtered)
        axes[0].scatter(
            x=horizontal_filtered["TVT"], 
            y=horizontal_filtered["GR"], 
            label=f"{prefix} Horizontal (Scatter)", 
            color='blue'
        )
        axes[0].set_title(label=f"Horizontal Well (Filtered): {prefix}")
        axes[0].set_xlabel(xlabel="TVT")
        axes[0].set_ylabel(ylabel="GR")
        axes[0].legend()
        axes[0].grid(visible=True)

        # Subplot 2: Typewell Only
        tvt_min = horizontal_filtered["TVT"].min()
        tvt_max = horizontal_filtered["TVT"].max()
        typewell_clipped = typewell[(typewell["TVT"] >= tvt_min) & (typewell["TVT"] <= tvt_max)]
        if typewell_clipped.empty:
            typewell_clipped = typewell

        axes[1].plot(
            typewell_clipped["TVT"], 
            typewell_clipped["GR"], 
            label=f"{prefix} Typewell (Line)", 
            color='red', 
            linewidth=2
        )
        axes[1].set_title(label=f"Typewell (Clipped): {prefix}")
        axes[1].set_xlabel(xlabel="TVT")
        axes[1].set_ylabel(ylabel="GR")
        axes[1].legend()
        axes[1].grid(visible=True)

        # Subplot 3: Combined (Original Filtered Horizontal + Clipped Typewell)
        axes[2].plot(
            horizontal_filtered["TVT"], 
            horizontal_filtered["GR"], 
            label=f"{prefix} Horizontal", 
            color='blue',
            linewidth=2
        )
        axes[2].plot(
            typewell_clipped["TVT"], 
            typewell_clipped["GR"], 
            label=f"{prefix} Typewell", 
            color='red', 
            linewidth=2
        )
        axes[2].set_title(label=f"Combined (Clipped): {prefix}")
        axes[2].set_xlabel(xlabel="TVT")
        axes[2].set_ylabel(ylabel="GR")
        axes[2].legend()
        axes[2].grid(visible=True)

        # Subplot 4: TVT vs Index (Filtered Horizontal Original)
        axes[3].plot(
            horizontal_filtered.index,
            horizontal_filtered["TVT"],
            label="Original TVT",
            color='green',
            linewidth=2
        )
        axes[3].set_title(label=f"TVT vs Index: {prefix}")
        axes[3].set_xlabel(xlabel="Index (i)")
        axes[3].set_ylabel(ylabel="TVT")
        axes[3].legend()
        axes[3].grid(visible=True)

        # Subplot 5: New Comparison Plot (Original vs Optimized)
        axes[4].plot(
            horizontal_filtered.index,
            horizontal_filtered["TVT"],
            label="Original TVT",
            color='green',
            alpha=0.6,
            linewidth=2
        )
        axes[4].plot(
            horizontal_filtered.index,
            horizontal_filtered["TVT_optimized"],
            label="Optimized TVT",
            color='purple',
            linestyle='--',
            linewidth=2
        )
        
        # Highlight Jump Locations (where step difference is large, indicating an exact jump)
        tvt_diff = np.abs(np.diff(horizontal_filtered["TVT_optimized"].values))
        jump_indices = np.where(tvt_diff >= 0.1)[0] + 1
        if len(jump_indices) > 0:
            axes[4].scatter(
                horizontal_filtered.index[jump_indices],
                horizontal_filtered["TVT_optimized"].iloc[jump_indices],
                color='red',
                marker='X',
                s=100,
                zorder=5,
                label='DP Jump Triggered'
            )

        axes[4].set_title(label=f"TVT Comparison (Error: {min_error:.2f})")
        axes[4].set_xlabel(xlabel="Index (i)")
        axes[4].set_ylabel(ylabel="TVT")
        axes[4].legend()
        axes[4].grid(visible=True)

        plt.tight_layout()
        plt.savefig(fname=f"data/graphs/{prefix}_optimized_tvt_plot.png")
        plt.close(fig=fig)

if __name__ == "__main__":
    data_path = "/home/justinlamlamlam/python/kaggle/rogii/data/raw/train"
    raw_train_data = helper.data_handling.get_raw_data(data_loc=data_path)
    plot_well_data_with_optimization(raw_data=raw_train_data)