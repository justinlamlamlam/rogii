import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/justinlamlamlam/python/kaggle/rogii/src")
import helper

def plot_well_data(raw_data: dict) -> None:
    """
    Generates and saves a figure with four subplots per well prefix for the first 10 wells. 
    Plots are saved as '{prefix}_tvt_gr_plot.png' instead of being displayed.

    Args:
        raw_data (dict): Dictionary where keys are well prefixes (str) and values are 
                         tuples of (horizontal_well_df, typewell_df).

    Returns:
        None: Saves the plots to disk.
    """
    # Only process the first 10 items
    for i, (prefix, (horizontal_well, typewell)) in enumerate(raw_data.items()):
        if i >= 10:
            break
            
        # Filter horizontal well data where 'TVT_input' is NaN
        horizontal_filtered = horizontal_well[horizontal_well['TVT_input'].isna()]
        
        # Calculate TVT range for clipping
        if not horizontal_filtered.empty:
            tvt_min = horizontal_filtered["TVT"].min()
            tvt_max = horizontal_filtered["TVT"].max()
            
            # Filter typewell to match the horizontal TVT range
            typewell_clipped = typewell[
                (typewell["TVT"] >= tvt_min) & (typewell["TVT"] <= tvt_max)
            ]
        else:
            typewell_clipped = typewell

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(24, 6))
        
        # Subplot 1: Horizontal Well Only (Filtered)
        axes[0].scatter(
            x=horizontal_filtered["TVT"], 
            y=horizontal_filtered["GR"], 
            label=f"{prefix} Horizontal (Scatter)", 
            # alpha=0.5, 
            # s=10
            color='blue'
        )
        axes[0].set_title(label=f"Horizontal Well (Filtered): {prefix}")
        axes[0].set_xlabel(xlabel="TVT")
        axes[0].set_ylabel(ylabel="GR")
        axes[0].legend()
        axes[0].grid(visible=True)

        # Subplot 2: Typewell Only (Clipped to horizontal range)
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

        # Subplot 3: Combined (Filtered Horizontal + Clipped Typewell)
        axes[2].plot(
            horizontal_filtered["TVT"], 
            horizontal_filtered["GR"], 
            label=f"{prefix} Horizontal", 
            # alpha=0.5, 
            # s=10
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

        # Subplot 4: TVT vs Index (Filtered Horizontal)
        axes[3].plot(
            horizontal_filtered.index,
            horizontal_filtered["TVT"],
            label=f"{prefix} TVT vs Index",
            color='green',
            linewidth=2
        )
        axes[3].set_title(label=f"TVT vs Index: {prefix}")
        axes[3].set_xlabel(xlabel="Index (i)")
        axes[3].set_ylabel(ylabel="TVT")
        axes[3].legend()
        axes[3].grid(visible=True)

        plt.tight_layout()
        # Save instead of showing
        plt.savefig(fname=f"data/graphs/{prefix}_tvt_gr_plot.png")
        plt.close(fig=fig)

if __name__ == "__main__":
    # Load data
    data_path = "/home/justinlamlamlam/python/kaggle/rogii/data/raw/train"
    raw_train_data = helper.data_handling.get_raw_data(data_loc=data_path)
    plot_well_data(raw_train_data)