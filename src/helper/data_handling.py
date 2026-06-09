# data_handling.py

import numpy as np
import pandas as pd
import os

def get_csv_prefixes(data_loc: str) -> set:
    """
    Reads files from a directory, filters for files ending with .csv, 
    extracts the first 8 characters of those filenames, and returns them as a set.

    Args:
        directory_path (str): The path to the directory to scan.

    Returns:
        set: A set containing the first 8 characters of each CSV filename found.
    """
    prefixes = set()
    
    # Iterate through all files in the specified directory
    for filename in os.listdir(path=data_loc):
        # Filter for files ending with .csv
        if filename.endswith('.csv'):
            # Extract first 8 characters and add to the set
            prefixes.add(filename[:8])
            
    return prefixes

def get_raw_data(data_loc: str) -> dict:
    """
    Gets raw data from csv

    Args:
        data_loc (str): The path to the directory to read data from.
    Returns:
        dict[well_id] -> (horizontal_well, typewell)
    """
    prefixes = get_csv_prefixes(data_loc=data_loc)
    raw_data = {}
    for prefix in prefixes:
        horizontal_well, typewell = pd.read_csv(f"{data_loc}/{prefix}__horizontal_well.csv"), pd.read_csv(f"{data_loc}/{prefix}__typewell.csv")
        assert horizontal_well is not None
        assert typewell is not None
        raw_data[prefix] = (horizontal_well, typewell)
    return raw_data

def interpolate_gr(raw_data: dict) -> dict:
    """
    Interpolates missing values linearly in the 'GR' column for both DataFrames in the dictionary.

    Args:
        raw_data (dict): Dictionary mapping well_id to a tuple of (horizontal_well, typewell) DataFrames.

    Returns:
        dict: The updated dictionary with interpolated 'GR' columns in the DataFrames.
    """
    for prefix, (horizontal_well, typewell) in raw_data.items():
        # Interpolate 'GR' column if it exists in horizontal_well
        if "GR" in horizontal_well.columns:
            horizontal_well["GR"] = horizontal_well["GR"].interpolate(method="linear", limit_direction="both")
            
        raw_data[prefix] = (horizontal_well, typewell)
        
    return raw_data

def preprocess_data(raw_data: dict) -> dict:
    """
    preprocess data for a well
    """
    interpolated_data = interpolate_gr(raw_data=raw_data)
    return interpolated_data
