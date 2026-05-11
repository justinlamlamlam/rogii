import helper
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Load data
    data_path = "/home/justinlamlamlam/python/kaggle/rogii/data/raw/train"
    raw_train_data = helper.data_handling.get_raw_data(data_loc=data_path)
    for i, (well_id, (hw_df, tw_df)) in enumerate(raw_train_data.items()):
        print(f"--- Sample {i+1} | Well ID: {well_id} ---")
        print("\nHorizontal Well DataFrame (Head):")
        print(hw_df.head())
        print("\nTypewell DataFrame (Head):")
        print(tw_df.head())
        print("-" * 40 + "\n")
        break

    # Build the index and capture normalization stats
    spatial_bundle = helper.geology.build_well_spatial_index(raw_data=raw_train_data)

    # Example: A batch of raw [X, Y, Z] points
    raw_query_points = np.array([
        [450000.0, 6200000.0, -1500.0],
        [450100.0, 6200100.0, -1550.0]
    ])

    # The lookup function handles scaling the input internally
    closest_tvts = helper.geology.lookup_closest_tvt(
        query_points=raw_query_points, 
        index_bundle=spatial_bundle
    )

    print(f"Nearest non-NaN TVT values: {closest_tvts}")

    # 2. Get the 5 closest neighbors and their distances
    # Example: query_pts is [n, 3]
    val_matrix, dist_matrix = helper.geology.get_n_nearest_neighbors(
        query_points=raw_query_points, 
        index_bundle=spatial_bundle, 
        k=5
    )

    # 3. Compute the single weighted value for each query point
    weighted_predictions = helper.geology.calculate_weighted_tvt(
        neighbor_targets=val_matrix, 
        distances=dist_matrix, 
        power=2.0
    )

    print(f"weighted_predictions: {weighted_predictions}")