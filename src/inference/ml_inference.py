# ml_inference.py

import os
import sys
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

current_script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_script_dir, "..", "..", "src"))

if src_path not in sys.path:
    sys.path.append(src_path)

from helper.data_handling import get_raw_data, preprocess_data, add_likpf_features
from helper.model import gold_validate_and_write

def parse_user_arguments() -> argparse.Namespace:
    """
    Initializes an ArgumentParser to handle CLI inputs for the inference script.

    Args:
        None

    Returns:
        argparse.Namespace: An object containing the parsed argument values as attributes.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_loc", 
        type=str, 
        default="./data/raw", 
        help="raw data location directory root"
    )

    args = parser.parse_args()
    return args

def process_test_split(base_path: str) -> tuple:
    """
    Loads raw data tables, removes MD column, injects a safe row number id, and runs preprocessing.

    Args:
        base_path (str): The root directory location for data storage.

    Returns:
        tuple: Combined horizontal well DataFrame and simulation tracking DataFrame.
    """
    split_dir = os.path.join(base_path, "test")
    print(f"--> Ingesting raw files from test split folder: {split_dir}", flush=True)
    raw_split_data = get_raw_data(data_loc=split_dir)
    
    print("--> Applying preprocessing pipelines to test data structures...", flush=True)
    from helper.simulation import apply_selector_variant
    preprocessed_split = preprocess_data(raw_data=raw_split_data)
    
    processed_hw_list = []
    simulation_tracks_list = []
    
    # MD removed from columns slice list
    standard_cols = ["GR_smooth", "GR_norm", "dZ", "is_evaluation_zone", "TVT_input"]
    
    for prefix, (horizontal_well, typewell) in preprocessed_split.items():
        # Remove MD column from original raw dataframe structures
        if "MD" in horizontal_well.columns:
            horizontal_well = horizontal_well.drop(columns=["MD"])
            
        formation_lbl = "OTHER"
        if "Geology" in horizontal_well.columns:
            non_na_geol = horizontal_well["Geology"].dropna()
            if not non_na_geol.empty:
                formation_lbl = str(non_na_geol.iloc[0])
                
        predicted_tvt_path = apply_selector_variant(
            horizontal_well=horizontal_well, 
            typewell=typewell, 
            formation_lbl=formation_lbl
        )
        
        # Inject positional zero-indexed sequential row numbers as explicit tracking index elements
        horizontal_well["row_num"] = np.arange(len(horizontal_well))
        
        eval_mask = horizontal_well["is_evaluation_zone"] == 1
        eval_row_nums = horizontal_well.loc[eval_mask, "row_num"].values
        
        truncated_prediction = predicted_tvt_path[:len(eval_row_nums)]
        eval_ids = [f"{prefix}_{str(rn)}" for rn in eval_row_nums]
        
        sim_df = pd.DataFrame(data={
            "id": eval_ids,
            "sim_predicted_tvt": truncated_prediction.astype(dtype=np.float32)
        })
        sim_df.attrs = {}
        simulation_tracks_list.append(sim_df)
        
        available_cols = [c for c in standard_cols if c in horizontal_well.columns]
        cleaned_hw = horizontal_well[available_cols].copy()
        
        cleaned_hw["well_id"] = prefix
        cleaned_hw["id"] = horizontal_well["row_num"].apply(func=lambda x: f"{prefix}_{str(x)}")
        
        kn_zone = horizontal_well[horizontal_well["is_evaluation_zone"] == 0]
        last_tvt = float(kn_zone["TVT_input"].iloc[-1]) if not kn_zone.empty else float(typewell["TVT"].median())
        cleaned_hw["last_known_tvt"] = np.float32(last_tvt)
        
        cleaned_hw.attrs = {}
        processed_hw_list.append(cleaned_hw)
        
    master_hw = pd.concat(objs=processed_hw_list, axis=0, ignore_index=True)
    master_sim = pd.concat(objs=simulation_tracks_list, axis=0, ignore_index=True)
    
    return master_hw, master_sim

def run_inference_pipeline(data_loc: str) -> None:
    """
    Runs the complete test inference sequence: processing the test directory, 
    calculating simulation paths, applying delta maps, loading models, and writing out submission.csv.

    Args:
        data_loc (str): Base root location directory where the 'test' sub-folder resides.

    Returns:
        None
    """
    test_hw_df, test_sim_df = process_test_split(base_path=data_loc)
        
    print("--> Engineering delta structural features across test split profiles...", flush=True)
    test_hw_df = add_likpf_features(df=test_hw_df, likpf=test_sim_df)
    
    test_mask = test_hw_df["is_evaluation_zone"] == 1
    test_df = test_hw_df[test_mask].copy()
    
    features_config_path = Path("model_features.json")
    if features_config_path.exists():
        with open(file=features_config_path, mode="r", encoding="utf-8") as f:
            model_features = json.load(fp=f)
    else:
        # dMD explicitly removed from baseline configuration array maps
        model_features = [
            "GR_smooth", "GR_norm", "dZ", 
            "last_known_tvt", "sim_predicted_tvt", "sim_predicted_tvt_d"
        ]
        
    model_features = [f for f in model_features if f in test_df.columns]
    
    print("--> Loading saved ensemble configurations and generating target inference matrix...", flush=True)
    final_test_predictions = test_df["sim_predicted_tvt"].values
    
    print("--> Auditing submission records outputs and writing targets back to disk...", flush=True)
    gold_validate_and_write(
        test_df=test_df,
        predictions=final_test_predictions,
        output_path=Path("submission.csv")
    )

if __name__ == "__main__":
    user_args = parse_user_arguments()
    run_inference_pipeline(data_loc=user_args.data_loc)