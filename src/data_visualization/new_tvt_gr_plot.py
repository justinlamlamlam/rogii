import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
sys.path.append("./src")
import helper
from helper.global_optimization import optimize_well_tvt, default_optimize_well_tvt
from helper.data_handling import preprocess_data

def plot_well_data(raw_data: dict) -> None:
    """
    Generates and saves a figure with five subplots per well prefix for the first 10 wells.
    Separates original data, provides original combined plots, predicted TVT combined plots, 
    and includes subplots showing TVT vs Index and the ground truth TVT vs predicted TVT over the evaluation zone index.
    Plots are saved as '{prefix}_tvt_gr_interpolate_plot.png' instead of being displayed.

    Args:
        raw_data (dict): Dictionary where keys are well prefixes (str) and values are 
                         tuples of (horizontal_well_df, typewell_df).

    Returns:
        None: Saves the plots to disk.
    }
    """
    processed_data = preprocess_data(raw_data=raw_data)
    # Only process the first 10 items
    for i, (prefix, (horizontal_well, typewell)) in enumerate(sorted(raw_data.items())):
        if i >= 10:
            break
        print(f"well: {prefix}")
            
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


        predicted_tvts = optimize_well_tvt(
            horizontal_df=processed_data[prefix][0],
            typewell_df=processed_data[prefix][1],
            jump_penalty=50.0,
            bin_size=0.1,
            max_tvt_diff=20,
            beam_width=100,
            allow_jump=False,
        )

        # Update figure layout to accommodate 5 columns
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(30, 6))
        
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

        # Subplot 2: Typewell Only (Clipped to horizontal range) - Original vs Interpolated
        axes[1].plot(
            typewell_clipped["TVT"], 
            typewell_clipped["GR_interpolated"], 
            label=f"{prefix} Typewell (Interpolated)", 
            color='orange', 
            linewidth=4,
            linestyle='--'
        )
        axes[1].plot(
            typewell_clipped["TVT"], 
            typewell_clipped["GR"], 
            label=f"{prefix} Typewell (Original)", 
            color='red', 
            linewidth=2
        )
        axes[1].set_title(label=f"Typewell (Clipped): {prefix}")
        axes[1].set_xlabel(xlabel="TVT")
        axes[1].set_ylabel(ylabel="GR")
        axes[1].legend()
        axes[1].grid(visible=True)

        # Subplot 3: Combined (Filtered Horizontal + Clipped Typewell) - Original Only
        axes[2].plot(
            horizontal_filtered["TVT"], 
            horizontal_filtered["GR"], 
            label=f"{prefix} Horizontal (Original)", 
            # alpha=0.5, 
            # s=10
            color='blue',
            linewidth=2
        )
        axes[2].plot(
            typewell_clipped["TVT"], 
            typewell_clipped["GR"], 
            label=f"{prefix} Typewell (Original)", 
            color='red', 
            linewidth=2
        )
        axes[2].set_title(label=f"Combined (Original): {prefix}")
        axes[2].set_xlabel(xlabel="TVT")
        axes[2].set_ylabel(ylabel="GR")
        axes[2].legend()
        axes[2].grid(visible=True)

        # Subplot 4: Combined using Predicted TVT
        axes[3].plot(
            predicted_tvts, 
            horizontal_filtered["GR"], 
            label=f"{prefix} Horizontal (Predicted TVT)", 
            color='purple',
            linewidth=2
        )
        axes[3].plot(
            typewell_clipped["TVT"], 
            typewell_clipped["GR"], 
            label=f"{prefix} Typewell (Original)", 
            color='red', 
            linewidth=2
        )
        axes[3].set_title(label=f"Combined (Predicted TVT): {prefix}")
        axes[3].set_xlabel(xlabel="TVT")
        axes[3].set_ylabel(ylabel="GR")
        axes[3].legend()
        axes[3].grid(visible=True)

        # Subplot 5: Predicted TVT vs Actual TVT over Index
        if len(predicted_tvts) == len(horizontal_filtered):
            axes[4].plot(
                horizontal_filtered.index,
                horizontal_filtered["TVT"],
                label="Actual TVT",
                color="green",
                linewidth=2
            )
            axes[4].plot(
                horizontal_filtered.index,
                predicted_tvts,
                label="Predicted TVT",
                color="purple",
                linewidth=2,
                linestyle="--"
            )
        else:
            # Fallback plotting if index mismatch occurs
            axes[4].plot(
                predicted_tvts, 
                label="Predicted TVT (Index Only)", 
                color="purple", 
                linewidth=2
            )
        axes[4].set_title(label=f"Predicted vs Actual TVT: {prefix}")
        axes[4].set_xlabel(xlabel="Index (i)")
        axes[4].set_ylabel(ylabel="TVT")
        axes[4].legend()
        axes[4].grid(visible=True)

        plt.tight_layout()
        # Save instead of showing
        plt.savefig(fname=f"data/graphs/new_tvt_gr/{prefix}_tvt_gr_interpolate_plot.png")
        plt.close(fig=fig)

if __name__ == "__main__":
    # Load data
    data_path = "/home/justinlamlamlam/python/kaggle/rogii/data/raw/train"
    raw_train_data = helper.data_handling.get_raw_data(data_loc=data_path)
    plot_well_data(raw_data=raw_train_data)