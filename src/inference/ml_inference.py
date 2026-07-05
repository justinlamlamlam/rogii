# ml_inference.py

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

current_script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_script_dir, "..", "..", "src"))

if src_path not in sys.path:
    sys.path.append(src_path)

# Importing configured modules from the helper module ecosystem
from helper.data_handling import get_raw_data, preprocess_data, add_likpf_features
from helper.simulation import apply_selector_variant
from helper.model import train_stack, gold_validate_and_write

def parse_user_arguments() -> argparse.Namespace:
    """
    Initializes an ArgumentParser to handle CLI inputs with updated defaults and cache flags.

    Args:
        None

    Returns:
        argparse.Namespace: An object containing the parsed argument values as attributes.
    """
    parser = argparse.ArgumentParser()

    # Location of raw data folder - defaults to ./data/raw
    parser.add_argument(
        "--data_loc", 
        type=str, 
        default="./data/raw", 
        help="raw data location directory root"
    )
    
    # Cache flag renamed to use_sim_cache and defaults to True
    parser.add_argument(
        "--no_sim_cache",
        dest="use_sim_cache",
        action="store_false",
        help="If set, deactivates the simulation cache and forces simulations to rerun from scratch."
    )
    parser.set_defaults(use_sim_cache=True)

    args = parser.parse_args()
    return args

def process_data_split(base_path: str, split_name: str) -> tuple:
    """
    Loads raw data tables and runs complete preprocessing for a specific dataset split folder.

    Args:
        base_path (str): The root directory location for data storage.
        split_name (str): Sub-folder split name ('train' or 'test').

    Returns:
        tuple: Combined horizontal well DataFrame, simulation list container, and an evaluation zone indicator mapping dictionary.
    """
    split_dir = os.path.join(base_path, split_name)
    print(f"--> Ingesting raw files from {split_name} split folder: {split_dir}", flush=True)
    raw_split_data = get_raw_data(data_loc=split_dir)
    
    print(f"--> Applying preprocessing pipelines to {split_name} data structures...", flush=True)
    preprocessed_split = preprocess_data(raw_data=raw_split_data)
    
    processed_hw_list = []
    simulation_tracks_list = []
    
    for prefix, (horizontal_well, typewell) in preprocessed_split.items():
        # Retrieve geological labels
        formation_lbl = "OTHER"
        if "Geology" in horizontal_well.columns:
            non_na_geol = horizontal_well["Geology"].dropna()
            if not non_na_geol.empty:
                formation_lbl = str(non_na_geol.iloc[0])
                
        # Run physics tracking optimizations across target well geometries
        predicted_tvt_path = apply_selector_variant(
            horizontal_well=horizontal_well, 
            typewell=typewell, 
            formation_lbl=formation_lbl
        )
        
        # Format track profiles to sync matrices with tabular data rows
        eval_mask = horizontal_well["is_evaluation_zone"] == 1
        eval_ids = horizontal_well.loc[eval_mask, "MD"].values
        
        sim_df = pd.DataFrame(data={
            "MD": eval_ids,
            "sim_predicted_tvt": predicted_tvt_path.astype(dtype=np.float32)
        })
        simulation_tracks_list.append(sim_df)
        
        # Attach layout properties tracking identities across logs
        horizontal_well["well_id"] = prefix
        
        kn_zone = horizontal_well[horizontal_well["is_evaluation_zone"] == 0]
        last_tvt = float(kn_zone["TVT_input"].iloc[-1]) if not kn_zone.empty else float(typewell["TVT"].median())
        horizontal_well["last_known_tvt"] = np.float32(last_tvt)
        
        processed_hw_list.append(horizontal_well)
        
    master_hw = pd.concat(objs=processed_hw_list, ignore_index=True)
    master_sim = pd.concat(objs=simulation_tracks_list, ignore_index=True)
    
    return master_hw, master_sim

def run_inference_pipeline(data_loc: str, use_sim_cache: bool = True) -> None:
    """
    Executes the end-to-end inference pipeline: reading distinct train and test 
    directories, loading or generating simulation tracking metrics based on a simulation 
    cache preference, adding delta features, and optimizing via model stack trees.

    Args:
        data_loc (str): Base root location directory where 'train' and 'test' sub-folders reside.
        use_sim_cache (bool): Dictates whether to read pre-saved cache files from disk or rerun simulations. Defaults to True.

    Returns:
        None
    """
    train_cache_file = Path("train_processed_hw_cache.pkl")
    train_sim_cache_file = Path("train_master_sim_cache.pkl")
    test_cache_file = Path("test_processed_hw_cache.pkl")
    test_sim_cache_file = Path("test_master_sim_cache.pkl")

    # 1 & 2 & 3. Load or Run Physics Simulation Workflows separately for train and test regions
    if (use_sim_cache and 
            train_cache_file.exists() and train_sim_cache_file.exists() and 
            test_cache_file.exists() and test_sim_cache_file.exists()):
        print("--> [Cache Mode Activated] Loading saved structural train and test features from cache files...", flush=True)
        train_hw_df = pd.read_pickle(filepath_or_buffer=train_cache_file)
        train_sim_df = pd.read_pickle(filepath_or_buffer=train_sim_cache_file)
        test_hw_df = pd.read_pickle(filepath_or_buffer=test_cache_file)
        test_sim_df = pd.read_pickle(filepath_or_buffer=test_sim_cache_file)
    else:
        if use_sim_cache:
            print("--> [Cache Warning] Cache requested but missing profiles detected. Re-processing splits...", flush=True)
            
        # Process train and test directories completely separate from one another
        train_hw_df, train_sim_df = process_data_split(base_path=data_loc, split_name="train")
        test_hw_df, test_sim_df = process_data_split(base_path=data_loc, split_name="test")
        
        # Save split properties to their respective cache tables on disk
        print("--> Saving newly generated split simulation profiles to disk cache tables...", flush=True)
        train_hw_df.to_pickle(path=train_cache_file)
        train_sim_df.to_pickle(path=train_sim_cache_file)
        test_hw_df.to_pickle(path=test_cache_file)
        test_sim_df.to_pickle(path=test_sim_cache_file)
        
    # 4. Process Tabular Delta Feature Engineering via add_likpf_features
    print("--> Engineering delta structural features across train and test split profiles...", flush=True)
    train_hw_df = add_likpf_features(df=train_hw_df, likpf=train_sim_df)
    test_hw_df = add_likpf_features(df=test_hw_df, likpf=test_sim_df)
    
    # Isolate targets using logical zone configurations
    train_mask = (train_hw_df["is_evaluation_zone"] == 0) & (train_hw_df["TVT_input"].notna())
    test_mask = test_hw_df["is_evaluation_zone"] == 1
    
    train_df = train_hw_df[train_mask].copy()
    test_df = test_hw_df[test_mask].copy()
    
    # Map final optimization target values
    train_df["target_tvt"] = train_df["TVT_input"]
    
    # 5. Define Feature Selection Fields
    model_features = [
        "GR_smooth", "GR_norm", "dZ", "dMD", 
        "last_known_tvt", "sim_predicted_tvt", "sim_predicted_tvt_d"
    ]
    model_features = [f for f in model_features if f in train_df.columns]
    
    # 6. Fit and Predict Gradient Boosted Tree Stacking Structures
    print("--> Training GBDT stacking ensemble loops on processed train split elements...", flush=True)
    _, final_test_predictions, _ = train_stack(
        train_df=train_df,
        test_df=test_df,
        features=model_features,
        target_col="target_tvt",
        device_mode="cpu"
    )
    
    # 7. Confirm Compliance Schema and Export Final Submissions csv
    print("--> Auditing submission records outputs and writing targets back to disk...", flush=True)
    gold_validate_and_write(
        test_df=test_df,
        predictions=final_test_predictions,
        output_path=Path("submission.csv")
    )

if __name__ == "__main__":
    # Gather user terminal flags inputs and forward parameters down to pipeline managers
    user_args = parse_user_arguments()
    run_inference_pipeline(data_loc=user_args.data_loc, use_sim_cache=user_args.use_sim_cache)