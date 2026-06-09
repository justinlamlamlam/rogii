# inference.py

import os
import pandas as pd
import sys
import argparse
sys.path.append("./src")

import helper
from helper.data_handling import get_raw_data, preprocess_data
from helper.global_optimization import optimize_well_tvt

def parse_user_arguments() -> argparse.Namespace:
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

def run_inference(data_dir: str) -> list:
    """
    Runs the Viterbi beam search inference pipeline on all test wells in a directory
    and returns a list of dictionaries containing the predicted TVT values and well IDs.

    Args:
        data_dir (str): Path to the directory containing the test CSV files.

    Returns:
        list[dict]: A list of dictionaries where each dict contains 'well_id' (str) 
                    and 'predicted_tvts' (list[float]).
    """
    # Load raw well paired files from target location
    print(f"Loading data from: {data_dir}")
    raw_data = get_raw_data(data_loc=data_dir)
    
    # Preprocess data
    processed_data = preprocess_data(raw_data=raw_data)

    predictions_list = []

    for well_id, (horizontal_well, typewell) in processed_data.items():
        print(f"Processing sequence optimizations for Well ID: {well_id}")
        
        # Keep original logic and data intact by creating a working copy
        output_df = horizontal_well.copy()
        
        # Call the single well prediction function using keyword arguments
        predicted_tvts = optimize_well_tvt(
            horizontal_df=output_df, 
            typewell_df=typewell
        )
        
        # Identify rows requiring evaluation mapping (NaN values)
        nan_mask = output_df["TVT_input"].isna()
        
        # Keep original logs and map variables precisely without modifying structure
        if len(predicted_tvts) == nan_mask.sum():
            print(f"Successfully optimized {len(predicted_tvts)} sequence nodes.")
            predictions_list.append({
                "well_id": well_id,
                "predicted_tvts": predicted_tvts
            })
        else:
            print(f"Warning: Sequence length mismatch for {well_id}. Skipping tracking step.")

    return predictions_list

if __name__ == "__main__":
    # Parsing arguments
    args = parse_user_arguments()
    
    # Run the inference pipeline using the parsed directory argument
    results = run_inference(data_dir=args.data_loc)
    print(f"Retrieved predictions for {len(results)} wells.")

    # Format the predictions into submission format: {well_id}_{original_index}
    submission_results = []
    
    # Reload raw data from path to access original indices accurately matching the mask logic
    raw_data = get_raw_data(data_loc=args.data_loc)
    
    for entry in results:
        well_id = entry["well_id"]
        preds = entry["predicted_tvts"]
        
        # Get original dataframe to find matching NaN indices
        hw_df, _ = raw_data[well_id]
        nan_mask = hw_df['TVT_input'].isna()
        original_indices = hw_df.index[nan_mask]
        
        for idx, pred in zip(original_indices, preds):
            submission_results.append({
                "id": f"{well_id}_{idx}",
                "tvt": pred
            })
            
    # Save to DataFrame
    output_df = pd.DataFrame(data=submission_results)
    print(output_df.head())
    output_df.to_csv("submission.csv", index=False)