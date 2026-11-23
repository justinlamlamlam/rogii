# optimize_from_ml.py

import os
import pandas as pd
import sys
import argparse
import numpy as np

current_script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_script_dir, "..", "..", "src"))

if src_path not in sys.path:
    sys.path.append(src_path)

import helper
from helper.data_handling import get_raw_data, preprocess_data, from_submission_csv
from helper.global_optimization import optimize_well_tvt_from_base

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

def merge_submission_files(
    submission_path: str = "submission.csv",
    ml_submission_path: str = "ml_submission.csv",
    output_path: str = "submission.csv",
) -> pd.DataFrame:
    """Reads two submission CSV files with columns 'id' and 'tvt' and merges them.

    Prioritizes values from submission.csv for each 'id'. If a value is missing or
    an 'id' is not present in submission.csv, falls back to the value in ml_submission.csv.

    Args:
        submission_path (str): File path to the primary submission CSV. Defaults to "submission.csv".
        ml_submission_path (str): File path to the secondary/fallback submission CSV. Defaults to "ml_submission.csv".
        output_path (str): File path where the merged CSV will be saved. Defaults to "final_submission.csv".

    Returns:
        pd.DataFrame: The merged DataFrame containing 'id' and prioritized 'tvt' values.
    """
    # Read both CSV files into DataFrames
    df_sub = pd.read_csv(filepath_or_buffer=submission_path)
    df_ml = pd.read_csv(filepath_or_buffer=ml_submission_path)

    # Perform an outer merge on 'id' to combine all unique IDs from both sources
    merged_df = pd.merge(
        left=df_sub,
        right=df_ml,
        on="id",
        how="outer",
        suffixes=("_sub", "_ml"),
    )

    # Use combine_first to take tvt_sub if available (and non-NaN), otherwise use tvt_ml
    merged_df["tvt"] = merged_df["tvt_sub"].combine_first(other=merged_df["tvt_ml"])

    # Keep only the essential columns
    final_df = merged_df[["id", "tvt"]]

    # Save the result to a new file
    final_df.to_csv(path_or_buf=output_path, index=False)

    return final_df

if __name__ == "__main__":
    # Parsing arguments
    args = parse_user_arguments()
    raw_data = get_raw_data(data_loc=args.data_loc)

    # Load baseline ML solutions from existing submission tracking file
    ml_solutions = from_submission_csv(raw_data_loc=".")

    # Preprocess logs through standard conditioning pipeline stages
    processed_data = preprocess_data(raw_data=raw_data)

    submission_rows = []

    # Iterate through each active well profile to perform local beam search alignment
    for well_id, (horizontal_df, typewell_df) in processed_data.items():
        # Enforce that every single data well must exist in the baseline submission file
        assert well_id in ml_solutions, f"Critical Error: Well ID {well_id} is missing from submission.csv"
        ml_sol = ml_solutions[well_id]
        
        # Run optimized track generation bounded within 1.0 unit of ML guidance baseline
        optimized_path = optimize_well_tvt_from_base(
            horizontal_df=horizontal_df,
            typewell_df=typewell_df,
            ml_solution=ml_sol
        )

        first_none_idx = horizontal_df[horizontal_df["TVT_input"].isna()].index[0]
        
        # Append elements matching submission.csv's exact layout schema: id (well_id_idx), tvt
        for idx, tvt_val in enumerate(optimized_path):
            submission_rows.append({
                "id": f"{well_id}_{idx+first_none_idx}",
                "tvt": tvt_val
            })

    # Reconstruct the optimized targets into a structural DataFrame container matching original submission layout
    sub_df = pd.DataFrame(data=submission_rows)
    
    # Overwrite old target output file with final constrained optimizations
    submission_path = "submission.csv"
    sub_df.to_csv(submission_path, index=False)
    merge_submission_files()
    print(f"Successfully replaced submission file at: {submission_path}")