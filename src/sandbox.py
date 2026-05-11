import os
import pandas as pd

def get_csv_prefixes(directory_path: str) -> set:
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
    for filename in os.listdir(path=directory_path):
        # Filter for files ending with .csv
        if filename.endswith('.csv'):
            # Extract first 8 characters and add to the set
            prefixes.add(filename[:8])
            
    return prefixes

train_loc = "../data/raw/train"
prefixs = get_csv_prefixes(train_loc)
print(len(prefixs))
