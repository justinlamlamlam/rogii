import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Append source directory to path for imports
sys.path.append("/home/justinlamlamlam/python/kaggle/rogii/src")

# Import lookup functions from math_mapping
from helper.math_mapping import tvt_to_gr, gr_to_tvt

import helper

def plot_mapping_functions(typewell_df: pd.DataFrame, output_dir: str = "data/graphs/mappings") -> None:
    """
    Generates and saves diagnostic plots showing the results of tvt_to_gr and gr_to_tvt lookups
    against the baseline Typewell reference log to verify interpolation accuracy.

    Args:
        typewell_df (pd.DataFrame): DataFrame containing reference 'TVT' and 'GR' columns.
        output_dir (str): Destination directory path where the generated plot will be saved.

    Returns:
        None: Saves the plot to disk.
    """
    if typewell_df.empty:
        print("Provided typewell DataFrame is empty. Skipping plot generation.")
        return

    # Clean and sort baseline array data to define query ranges
    df_sorted = typewell_df.sort_values(by="TVT").dropna(subset=["TVT", "GR"])
    orig_tvt = df_sorted["TVT"].to_numpy(dtype=np.float64)
    orig_gr = df_sorted["GR"].to_numpy(dtype=np.float64)

    if len(orig_tvt) == 0:
        print("No valid numeric data found in typewell. Skipping plot generation.")
        return

    # Generate dense synthetic evaluation arrays spanning the original bounds
    eval_tvt_line = np.linspace(start=orig_tvt.min(), stop=orig_tvt.max(), num=500)
    eval_gr_line = np.linspace(start=orig_gr.min(), stop=orig_gr.max(), num=500)

    # 1. Forward Mapping: Sample tvt_to_gr across the linear space
    pred_gr = [tvt_to_gr(tvt=t, typewell_df=typewell_df) for t in eval_tvt_line]

    # 2. Inverse Mapping: Sample gr_to_tvt (collecting individual point relationships)
    inverse_gr_coords = []
    inverse_tvt_coords = []
    for g in eval_gr_line:
        matched_tvts = gr_to_tvt(gr_target=g, typewell_df=typewell_df)
        for t in matched_tvts:
            inverse_gr_coords.append(g)
            inverse_tvt_coords.append(t)

    # Instantiate figure layout
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # Left Subplot: Forward Mapping (TVT -> GR)
    axes[0].plot(
        eval_tvt_line,
        pred_gr,
        label="Interpolated Function (tvt_to_gr)",
        color="blue",
        linewidth=2,
        zorder=1
    )
    axes[0].scatter(
        orig_tvt,
        orig_gr,
        label="Original Control Points",
        color="red",
        s=30,
        marker="o",
        zorder=2
    )
    axes[0].set_title(label="Forward Mapping: TVT vs Gamma Ray")
    axes[0].set_xlabel(xlabel="Total Vertical Thickness (TVT)")
    axes[0].set_ylabel(ylabel="Gamma Ray (GR)")
    axes[0].legend()
    axes[0].grid(visible=True)

    # Right Subplot: Inverse Mapping (GR -> TVT)
    axes[1].scatter(
        inverse_gr_coords,
        inverse_tvt_coords,
        label="Inversion Solutions (gr_to_tvt)",
        color="purple",
        s=10,
        alpha=0.6,
        zorder=1
    )
    axes[1].scatter(
        orig_gr,
        orig_tvt,
        label="Original Control Points",
        color="red",
        s=30,
        marker="o",
        zorder=2
    )
    axes[1].set_title(label="Inverse Mapping: GR vs TVT")
    axes[1].set_xlabel(xlabel="Gamma Ray (GR)")
    axes[1].set_ylabel(ylabel="Total Vertical Thickness (TVT)")
    axes[1].legend()
    axes[1].grid(visible=True)

    plt.tight_layout()

    # Ensure output directory structure exists before saving
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "mapping_validation_plot.png")
    plt.savefig(fname=save_path)
    plt.close(fig=fig)
    print(f"Validation plots successfully written to: {save_path}")

if __name__ == "__main__":
    # Load raw well logs via existing pipeline
    data_path = "/home/justinlamlamlam/python/kaggle/rogii/data/raw/train"
    raw_train_data = helper.data_handling.get_raw_data(data_loc=data_path)

    if raw_train_data:
        # Extract the first available typewell dataframe for evaluation
        first_key = sorted(raw_train_data.keys())[0]
        _, sample_typewell_df = raw_train_data[first_key]
        
        print(f"Executing mapping validation using typewell data from well: {first_key}")
        plot_mapping_functions(typewell_df=sample_typewell_df)
    else:
        print("No raw training data discovered at specified path.")