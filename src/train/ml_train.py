# ml_train.py

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
from helper.simulation import apply_selector_variant
from helper.model import train_stack

def parse_user_arguments() -> argparse.Namespace:
    """
    Initializes an ArgumentParser to handle CLI inputs for the training script.

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
    
    parser.add_argument(
        "--no_sim_cache",
        dest="use_sim_cache",
        action="store_false",
        help="If set, deactivates the simulation cache and forces simulations to rerun from scratch."
    )
    parser.set_defaults(use_sim_cache=True)

    args = parser.parse_args()
    return args

def process_train_split(base_path: str) -> tuple:
    """
    Loads raw data tables, removes MD column, injects a safe row number id, and runs preprocessing.

    Args:
        base_path (str): The root directory location for data storage.

    Returns:
        tuple: Combined horizontal well DataFrame and simulation tracking DataFrame.
    """
    split_dir = os.path.join(base_path, "train")
    print(f"--> Ingesting raw files from train split folder: {split_dir}", flush=True)
    raw_split_data = get_raw_data(data_loc=split_dir)
    
    print("--> Applying preprocessing pipelines to train data structures...", flush=True)
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

def run_training_pipeline(data_loc: str, use_sim_cache: bool = True) -> None:
    """
    Executes the training sequence: loading training sets, performing structural 
    simulations or using cache files, optimizing stacking parameters, and exporting artifacts.

    Args:
        data_loc (str): Base root location directory where the 'train' sub-folder resides.
        use_sim_cache (bool): Determines whether to utilize pre-saved training cache files. Defaults to True.

    Returns:
        None
    """
    train_cache_file = Path("train_processed_hw_cache.pkl")
    train_sim_cache_file = Path("train_master_sim_cache.pkl")

    if use_sim_cache and train_cache_file.exists() and train_sim_cache_file.exists():
        print("--> [Cache Mode Activated] Loading saved structural train features from cache files...", flush=True)
        train_hw_df = pd.read_pickle(filepath_or_buffer=train_cache_file)
        train_sim_df = pd.read_pickle(filepath_or_buffer=train_sim_cache_file)
    else:
        if use_sim_cache:
            print("--> [Cache Warning] Training cache missing. Re-processing training directory...", flush=True)
        train_hw_df, train_sim_df = process_train_split(base_path=data_loc)
        
        print("--> Saving generated train simulation profiles to disk cache tables...", flush=True)
        train_hw_df.to_pickle(path=train_cache_file)
        train_sim_df.to_pickle(path=train_sim_cache_file)
        
    print("--> Engineering delta structural features across train split profiles...", flush=True)
    train_hw_df = add_likpf_features(df=train_hw_df, likpf=train_sim_df)
    
    train_mask = (train_hw_df["is_evaluation_zone"] == 0) & (train_hw_df["TVT_input"].notna())
    train_df = train_hw_df[train_mask].copy()
    train_df["target_tvt"] = train_df["TVT_input"]
    
    # dMD explicitly removed from baseline architecture configuration maps
    model_features = [
        "GR_smooth", "GR_norm", "dZ", 
        "last_known_tvt", "sim_predicted_tvt", "sim_predicted_tvt_d"
    ]
    model_features = [f for f in model_features if f in train_df.columns]
    
    with open(file="model_features.json", mode="w", encoding="utf-8") as f:
        json.dump(obj=model_features, fp=f)
        
    print("--> Training GBDT stacking ensemble loops on processed train split elements...", flush=True)
    _ = train_stack(
        train_df=train_df,
        test_df=train_df,
        features=model_features,
        target_col="target_tvt",
        device_mode="cpu"
    )
    print("--> Training completed successfully. Model artifacts generated.", flush=True)

if __name__ == "__main__":
    user_args = parse_user_arguments()
    run_training_pipeline(data_loc=user_args.data_loc, use_sim_cache=user_args.use_sim_cache)