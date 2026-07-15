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
    print(f"Successfully replaced submission file at: {submission_path}")