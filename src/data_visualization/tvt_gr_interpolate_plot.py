import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/justinlamlamlam/python/kaggle/rogii/src")
import helper

def plot_well_data(raw_data: dict) -> None:
    """
    Generates and saves a figure with six subplots per well prefix for the first 10 wells.
    Separates original and interpolated data, and provides both original and interpolated combined plots.
    Plots are saved as '{prefix}_tvt_gr_interpolate_plot.png' instead of being displayed.

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
            
        # Create copies to preserve original data for plotting alongside interpolated data
        horizontal_well_df = horizontal_well.copy()
        typewell_df = typewell.copy()
        
        # Create interpolated columns
        horizontal_well_df["GR_interpolated"] = horizontal_well_df["GR"].interpolate(method="linear", limit_direction="both")
        typewell_df["GR_interpolated"] = typewell_df["GR"].interpolate(method="linear", limit_direction="both")
            
        # Filter horizontal well data where 'TVT_input' is NaN
        horizontal_filtered = horizontal_well_df[horizontal_well_df['TVT_input'].isna()]
        
        # Calculate TVT range for clipping
        if not horizontal_filtered.empty:
            tvt_min = horizontal_filtered["TVT"].min()
            tvt_max = horizontal_filtered["TVT"].max()
            
            # Filter typewell to match the horizontal TVT range
            typewell_clipped = typewell_df[
                (typewell_df["TVT"] >= tvt_min) & (typewell_df["TVT"] <= tvt_max)
            ]
        else:
            typewell_clipped = typewell_df

        fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(36, 6))
        
        # Subplot 1: Horizontal Well Only (Filtered) - Original
        axes[0].scatter(
            x=horizontal_filtered["TVT"], 
            y=horizontal_filtered["GR"], 
            label=f"{prefix} Horizontal (Original)", 
            # alpha=0.5, 
            # s=10
            color='blue'
        )
        axes[0].set_title(label=f"Horizontal Well (Original): {prefix}")
        axes[0].set_xlabel(xlabel="TVT")
        axes[0].set_ylabel(ylabel="GR")
        axes[0].legend()
        axes[0].grid(visible=True)

        # Subplot 2: Horizontal Well Only (Filtered) - Interpolated
        axes[1].scatter(
            x=horizontal_filtered["TVT"], 
            y=horizontal_filtered["GR_interpolated"], 
            label=f"{prefix} Horizontal (Interpolated)", 
            color='cyan'
        )
        axes[1].set_title(label=f"Horizontal Well (Interpolated): {prefix}")
        axes[1].set_xlabel(xlabel="TVT")
        axes[1].set_ylabel(ylabel="GR")
        axes[1].legend()
        axes[1].grid(visible=True)

        # Subplot 3: Typewell Only (Clipped to horizontal range) - Original vs Interpolated
        axes[2].plot(
            typewell_clipped["TVT"], 
            typewell_clipped["GR_interpolated"], 
            label=f"{prefix} Typewell (Interpolated)", 
            color='orange', 
            linewidth=4,
            linestyle='--'
        )
        axes[2].plot(
            typewell_clipped["TVT"], 
            typewell_clipped["GR"], 
            label=f"{prefix} Typewell (Original)", 
            color='red', 
            linewidth=2
        )
        axes[2].set_title(label=f"Typewell (Clipped): {prefix}")
        axes[2].set_xlabel(xlabel="TVT")
        axes[2].set_ylabel(ylabel="GR")
        axes[2].legend()
        axes[2].grid(visible=True)

        # Subplot 4: Combined (Filtered Horizontal + Clipped Typewell) - Original Only
        axes[3].plot(
            horizontal_filtered["TVT"], 
            horizontal_filtered["GR"], 
            label=f"{prefix} Horizontal (Original)", 
            # alpha=0.5, 
            # s=10
            color='blue',
            linewidth=2
        )
        axes[3].plot(
            typewell_clipped["TVT"], 
            typewell_clipped["GR"], 
            label=f"{prefix} Typewell (Original)", 
            color='red', 
            linewidth=2
        )
        axes[3].set_title(label=f"Combined (Original): {prefix}")
        axes[3].set_xlabel(xlabel="TVT")
        axes[3].set_ylabel(ylabel="GR")
        axes[3].legend()
        axes[3].grid(visible=True)

        # Subplot 5: Combined (Filtered Horizontal + Clipped Typewell) - Interpolated Only
        axes[4].plot(
            horizontal_filtered["TVT"], 
            horizontal_filtered["GR_interpolated"], 
            label=f"{prefix} Horizontal (Interp)", 
            # alpha=0.5, 
            # s=10
            color='blue',
            linewidth=2
        )
        axes[4].plot(
            typewell_clipped["TVT"], 
            typewell_clipped["GR_interpolated"], 
            label=f"{prefix} Typewell (Interp)", 
            color='red', 
            linewidth=2
        )
        axes[4].set_title(label=f"Combined (Interpolated): {prefix}")
        axes[4].set_xlabel(xlabel="TVT")
        axes[4].set_ylabel(ylabel="GR")
        axes[4].legend()
        axes[4].grid(visible=True)

        # Subplot 6: TVT vs Index (Filtered Horizontal)
        axes[5].plot(
            horizontal_filtered.index,
            horizontal_filtered["TVT"],
            label=f"{prefix} TVT vs Index",
            color='green',
            linewidth=2
        )
        axes[5].set_title(label=f"TVT vs Index: {prefix}")
        axes[5].set_xlabel(xlabel="Index (i)")
        axes[5].set_ylabel(ylabel="TVT")
        axes[5].legend()
        axes[5].grid(visible=True)

        plt.tight_layout()
        # Save instead of showing
        plt.savefig(fname=f"data/graphs/{prefix}_tvt_gr_interpolate_plot.png")
        plt.close(fig=fig)

if __name__ == "__main__":
    # Load data
    data_path = "/home/justinlamlamlam/python/kaggle/rogii/data/raw/train"
    raw_train_data = helper.data_handling.get_raw_data(data_loc=data_path)
    plot_well_data(raw_data=raw_train_data)