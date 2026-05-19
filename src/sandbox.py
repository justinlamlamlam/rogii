import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import helper

def plot_tvt_mapping_verification(typewell_df: pd.DataFrame, num_points: int = 30, subdivisions: int = 5) -> None:
    """
    Plots the original typewell TVT vs GR nodes against finer subdivided 
    interpolated points to verify that the linear mapping functions correctly.

    Args:
        typewell_df (pd.DataFrame): DataFrame containing 'TVT' and 'GR' columns.
        num_points (int): Number of original nodes to plot from the start. Defaults to 30.
        subdivisions (int): Number of intermediate points to generate between each node. Defaults to 5.

    Returns:
        None: Displays a matplotlib plot.
    """
    # Extract the first N rows
    subset_df = typewell_df.head(n=num_points)
    tvt_nodes = subset_df['TVT'].values
    gr_nodes = subset_df['GR'].values

    # Generate subdivided TVT points
    subdivided_tvt = []
    for i in range(len(tvt_nodes) - 1):
        # Linearly space points between node i and node i+1
        segment_tvt = np.linspace(start=tvt_nodes[i], stop=tvt_nodes[i+1], num=subdivisions + 2)
        # Exclude the last point to prevent duplicates in the loop
        subdivided_tvt.extend(segment_tvt[:-1])
    # Add the final node
    subdivided_tvt.append(tvt_nodes[-1])
    subdivided_tvt = np.array(subdivided_tvt)

    # Calculate GR using linear interpolation (simulating forward_map logic)
    subdivided_gr = np.interp(x=subdivided_tvt, xp=typewell_df['TVT'].values, fp=typewell_df['GR'].values)

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot original nodes
    plt.scatter(tvt_nodes, gr_nodes, color='black', zorder=5, label='Original Nodes (0.5 TVT intervals)')
    plt.plot(tvt_nodes, gr_nodes, color='black', linestyle='--', alpha=0.5, label='Node Segments')
    
    # Plot subdivided points
    plt.scatter(subdivided_tvt, subdivided_gr, color='red', marker='x', s=20, zorder=4, label='Subdivided Interpolated Points')

    plt.title(label=f"TVT vs GR Mapping Verification (First {num_points} Nodes)")
    plt.xlabel(xlabel="Total Vertical Thickness (TVT)")
    plt.ylabel(ylabel="Gamma Ray (GR)")
    plt.legend(loc="best")
    plt.grid(visible=True, linestyle=':', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    data_path = "/home/justinlamlamlam/python/kaggle/rogii/data/raw/train"
    raw_train_data = helper.data_handling.get_raw_data(data_loc=data_path)
    for i, (well_id, (hw_df, tw_df)) in enumerate(raw_train_data.items()):
        plot_tvt_mapping_verification(typewell_df=tw_df)
        break