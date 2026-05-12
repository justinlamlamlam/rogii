# inference.py

import numpy as np
import pandas as pd
import argparse
import helper

def parse_user_arguments():
    """
    Initializes an ArgumentParser to handle CLI inputs.

    Args:
        None

    Returns:
        argparse.Namespace: An object containing the parsed argument values as attributes.
    """
    parser = argparse.ArgumentParser()

    # String argument
    parser.add_argument(
        "--data_loc", 
        type=str, 
        required=True, 
        help="raw data location"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parsing arguments
    args = parse_user_arguments()

    # load data
    train_data = helper.data_handling.get_raw_data(f"{args.data_loc}/train")
    test_data = helper.data_handling.get_raw_data(f"{args.data_loc}/test")

    # 1. Combine data and build the index
    all_raw_data = {**train_data, **test_data}
    index_bundle = helper.geology.build_well_spatial_index(raw_data=all_raw_data)

    # 2. Iterate and predict
    results = []

    for well_id, (hw_df, _) in test_data.items():
        # Find indices where TVT_input is missing
        nan_mask = hw_df['TVT_input'].isna()
        
        if nan_mask.any():
            # Extract coordinates for the missing rows
            query_coords = hw_df.loc[nan_mask, ['X', 'Y', 'Z']].to_numpy()
            
            # Option A: Get the single closest point
            # predictions = lookup_closest_tvt(query_points=query_coords, index_bundle=index_bundle)
            
            # Option B: Use Weighted Mean (Smoother)
            val_matrix, dist_matrix = helper.geology.get_n_nearest_neighbors(
                query_points=query_coords, 
                index_bundle=index_bundle, 
                k=1
            )
            predictions = helper.geology.calculate_weighted_tvt(neighbor_targets=val_matrix, distances=dist_matrix)
            
            # 3. Format the IDs as {well_id}_{original_index}
            # Use the original DataFrame index to ensure mapping is correct
            original_indices = hw_df.index[nan_mask]
            for idx, pred in zip(original_indices, predictions):
                results.append({
                    "id": f"{well_id}_{idx}",
                    "tvt": pred
                })

    # 4. Save to DataFrame
    output_df = pd.DataFrame(results)

    print(output_df.head())
    output_df.to_csv("submission.csv", index=False)