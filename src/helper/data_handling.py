# data_handling.py

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

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

# ==============================================================================
# STAGE 1: ABSOLUTE SPATIAL ALIGNMENT (PHYSICS-BASED LAYER)
# ==============================================================================

def align_geological_contacts(raw_data: dict, ref_col: str = "EGFDU") -> dict:
    """
    Calculates elevation offsets based on structural geological markers and aligns 
    the horizontal well trajectory with the typewell baseline framework.

    Args:
        raw_data (dict): Dictionary mapping well_id to (horizontal_well, typewell).
        ref_col (str): The name of the geological reference marker column.

    Returns:
        dict: The updated dataset dictionary containing structural elevation corrections.
    """
    for prefix, (horizontal_well, typewell) in raw_data.items():
        if ref_col in horizontal_well.columns and "Z" in horizontal_well.columns and "TVT" in typewell.columns:
            tw_g = typewell.dropna(subset=["Geology"]) if "Geology" in typewell.columns else typewell
            matching_rows = tw_g[tw_g["Geology"] == ref_col] if "Geology" in tw_g.columns else pd.DataFrame()
            
            if not matching_rows.empty and "TVT_input" in horizontal_well.columns:
                ref_tvt = matching_rows["TVT"].min()
                hw_tr = horizontal_well[horizontal_well["TVT_input"].notna()]
                if not hw_tr.empty:
                    offset = (hw_tr["TVT_input"] - (ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]))).mean()
                    horizontal_well["Z_aligned"] = ref_tvt - (horizontal_well["Z"] - horizontal_well[ref_col]) + offset
        raw_data[prefix] = (horizontal_well, typewell)
    return raw_data

def transform_coordinate_deltas(raw_data: dict) -> dict:
    """
    Converts absolute spatial metrics into sequential physical displacement change step arrays.

    Args:
        raw_data (dict): Dictionary mapping well_id to (horizontal_well, typewell).

    Returns:
        dict: Updated dictionary where horizontal wells have computed delta features.
    """
    for prefix, (horizontal_well, typewell) in raw_data.items():
        if "Z" in horizontal_well.columns:
            dz = np.diff(a=horizontal_well["Z"].values)
            horizontal_well["dZ"] = np.insert(arr=dz, obj=0, values=0.0).astype(dtype=np.float64)
        if "MD" in horizontal_well.columns:
            dm = np.diff(a=horizontal_well["MD"].values)
            horizontal_well["dMD"] = np.insert(arr=dm, obj=0, values=0.0).astype(dtype=np.float64)
        raw_data[prefix] = (horizontal_well, typewell)
    return raw_data

# ==============================================================================
# STAGE 2: SIGNAL CONDITIONING & FORMATTING
# ==============================================================================

def denoise_sensors(raw_data: dict, r: int = 5) -> dict:
    """
    Applies non-linear Savitzky-Golay filters and centered rolling means to eliminate 
    high-frequency borehole sensor measurement fluctuations from the log tracks.

    Args:
        raw_data (dict): Dictionary mapping well_id to (horizontal_well, typewell).
        r (int): Window radius scale to pass down to rolling window operations.

    Returns:
        dict: The updated dictionary housing denoised logging curves.
    """
    for prefix, (horizontal_well, typewell) in raw_data.items():
        if "GR" in horizontal_well.columns:
            hgr = horizontal_well["GR"].values
            n = len(hgr)
            if r > 0 and n > max(3, 2 * r + 1):
                win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
                horizontal_well["GR_smooth"] = savgol_filter(x=hgr, window_length=win, polyorder=min(2, win - 1))
            else:
                horizontal_well["GR_smooth"] = hgr.copy()
                
            s = horizontal_well["GR_smooth"].interpolate(limit_direction="both")
            mean_fb = typewell["GR"].mean() if "GR" in typewell.columns else 0.0
            s = s.fillna(value=mean_fb)
            horizontal_well["GR_smooth"] = s.rolling(window=r*2+1, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)
            
        raw_data[prefix] = (horizontal_well, typewell)
    return raw_data

def standardize_reference_grid(raw_data: dict, step: float = 0.2) -> dict:
    """
    Resamples irregular vertical data lines onto an exactly uniform 0.2m gridded spacing configuration.

    Args:
        raw_data (dict): Dictionary mapping well_id to (horizontal_well, typewell).
        step (float): The standardized structural sampling resolution interval in meters.

    Returns:
        dict: The updated dictionary with metadata variables appended to individual containers.
    """
    for prefix, (horizontal_well, typewell) in raw_data.items():
        if "TVT" in typewell.columns and "GR" in typewell.columns:
            tw_tvt = typewell["TVT"].values
            tw_gr = typewell["GR"].values
            tmin, tmax = float(tw_tvt.min()), float(tw_tvt.max())
            tvt_g = np.arange(start=tmin, stop=tmax + step, step=step)
            gr_g = np.interp(x=tvt_g, xp=tw_tvt, fp=tw_gr).astype(dtype=np.float64)
            
            typewell_metadata = {"gr_grid": gr_g, "tmin": tmin, "tstep": step}
            horizontal_well.attrs["grid_meta"] = typewell_metadata
            
        raw_data[prefix] = (horizontal_well, typewell)
    return raw_data

def apply_scale_invariant_normalization(raw_data: dict, win: int = 5) -> dict:
    """
    Applies localized moving window Z-score transformations to create gain-invariant sequence vectors.

    Args:
        raw_data (dict): Dictionary mapping well_id to (horizontal_well, typewell).
        win (int): Size profile of localized structural windows.

    Returns:
        dict: Updated dataset containing scale-normalized logs.
    """
    for prefix, (horizontal_well, typewell) in raw_data.items():
        if "GR_smooth" in horizontal_well.columns:
            vals = horizontal_well["GR_smooth"].values
            rolling_obj = pd.Series(data=vals).rolling(window=win, center=True, min_periods=1)
            means = rolling_obj.mean().values
            stds = rolling_obj.std(ddof=0).values
            horizontal_well["GR_norm"] = ((vals - means) / (stds + 1e-6)).astype(dtype=np.float32)
        raw_data[prefix] = (horizontal_well, typewell)
    return raw_data

# ==============================================================================
# STAGE 3: OPERATIONAL FEATURE & LAYOUT PREPARATION
# ==============================================================================

def extract_dynamic_variance(raw_data: dict) -> dict:
    """
    Measures the local background standard deviation noise fingerprint over matching known subsets.

    Args:
        raw_data (dict): Dictionary mapping well_id to (horizontal_well, typewell).

    Returns:
        dict: Dictionary where horizontal well attributes record specific variance settings.
    """
    for prefix, (horizontal_well, typewell) in raw_data.items():
        gr_sig = 30.0  # Default parameter fallback baseline
        if "TVT_input" in horizontal_well.columns and "GR" in horizontal_well.columns and "TVT" in typewell.columns:
            kn = horizontal_well[horizontal_well["TVT_input"].notna() & horizontal_well["GR"].notna()]
            if len(kn) >= 20:
                residuals = kn["GR"].values - np.interp(x=kn["TVT_input"].values, xp=typewell["TVT"].values, fp=typewell["GR"].values)
                gr_sig = float(np.clip(a=np.std(a=residuals), a_min=10.0, a_max=60.0))
        horizontal_well.attrs["gr_sig"] = gr_sig
        raw_data[prefix] = (horizontal_well, typewell)
    return raw_data

def initialize_trend_velocity(raw_data: dict) -> dict:
    """
    Extracts geometric trajectory velocity rates from historical tracking endpoints.

    Args:
        raw_data (dict): Dictionary mapping well_id to (horizontal_well, typewell).

    Returns:
        dict: Updated dictionary holding systemic momentum values.
    """
    for prefix, (horizontal_well, typewell) in raw_data.items():
        ir = 0.0
        if "TVT_input" in horizontal_well.columns and "Z" in horizontal_well.columns and "MD" in horizontal_well.columns:
            kn = horizontal_well[horizontal_well["TVT_input"].notna()]
            if len(kn) >= 3:
                tail = kn.tail(n=30)
                dt = np.diff(a=tail["TVT_input"].values)
                dz = np.diff(a=tail["Z"].values)
                dm = np.diff(a=tail["MD"].values)
                m = dm > 0
                if m.sum() >= 3:
                    ir = float(np.median(a=((dt + dz)[m] / dm[m])))
        horizontal_well.attrs["initial_trend"] = ir
        raw_data[prefix] = (horizontal_well, typewell)
    return raw_data

def classify_trajectory_geometry(raw_data: dict) -> dict:
    """
    Evaluates profile sequence gaps to categorize geometric well track complexity codes.

    Args:
        raw_data (dict): Dictionary mapping well_id to (horizontal_well, typewell).

    Returns:
        dict: Processed dictionaries containing coded categorical profiles.
    """
    SELECTOR_N_EVAL_THRESHOLD = 4840.0
    SELECTOR_Z_SPAN_THRESHOLDS = (136.73, 185.51)
    
    for prefix, (horizontal_well, typewell) in raw_data.items():
        if "TVT_input" in horizontal_well.columns and "Z" in horizontal_well.columns:
            eval_mask = horizontal_well["TVT_input"].isna().to_numpy()
            n_eval = float(eval_mask.sum())
            z_eval = horizontal_well.loc[eval_mask, "Z"].values.astype(dtype=float)
            z_span = float(np.nanmax(a=z_eval) - np.nanmin(a=z_eval)) if len(z_eval) else 0.0
            
            n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
            z_bin = int(np.searchsorted(a=SELECTOR_Z_SPAN_THRESHOLDS, v=z_span, side="right"))
            horizontal_well.attrs["selector_code"] = n_bin + 2 * z_bin
        raw_data[prefix] = (horizontal_well, typewell)
    return raw_data

def split_sequence_masks(raw_data: dict) -> dict:
    """
    Partitions data metrics to mask active tracking zones from background validation bounds.

    Args:
        raw_data (dict): Dictionary mapping well_id to (horizontal_well, typewell).

    Returns:
        dict: Transformed records where localized indicators mark calculation constraints.
    """
    for prefix, (horizontal_well, typewell) in raw_data.items():
        if "TVT_input" in horizontal_well.columns:
            horizontal_well["is_evaluation_zone"] = horizontal_well["TVT_input"].isna().astype(dtype=np.int32)
        raw_data[prefix] = (horizontal_well, typewell)
    return raw_data

def enforce_matrix_typecasting(raw_data: dict) -> dict:
    """
    Performs typecasting configuration layout transforms for target data arrays.

    Args:
        raw_data (dict): Dictionary mapping well_id to (horizontal_well, typewell).

    Returns:
        dict: Finalized configurations containing cast elements ready for execution.
    """
    for prefix, (horizontal_well, typewell) in raw_data.items():
        numeric_cols_hw = horizontal_well.select_dtypes(include=[np.number]).columns
        horizontal_well[numeric_cols_hw] = horizontal_well[numeric_cols_hw].astype(dtype=np.float32)
        
        numeric_cols_tw = typewell.select_dtypes(include=[np.number]).columns
        typewell[numeric_cols_tw] = typewell[numeric_cols_tw].astype(dtype=np.float32)
        
        raw_data[prefix] = (horizontal_well, typewell)
    return raw_data

# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

def add_likpf_features(df: pd.DataFrame, likpf: pd.DataFrame) -> pd.DataFrame:
    """
    Appends the particle filter trajectory features and computes relative delta depth profiles.

    Args:
        df (pd.DataFrame): The main processed horizontal well feature dataframe.
        likpf (pd.DataFrame): Matrix container storing raw likelihood simulation trajectory logs.

    Returns:
        pd.DataFrame: Augmented dataframe with raw simulation tracks and delta coordinate maps.
    """
    df = df.merge(right=likpf, on="id", how="left")
    for c in [col for col in likpf.columns if col != "id"]:
        df[c] = df[c].fillna(value=df["last_known_tvt"])
        df[c + "_d"] = (df[c] - df["last_known_tvt"]).astype(np.float32)
    return df

# ==============================================================================
# PIPELINE ENSEMBLE MANAGER
# ==============================================================================

def preprocess_data(raw_data: dict) -> dict:
    """
    preprocess data for a well execution workflow by driving entries through 
    the full chronological geological processing sequence step-by-step.

    Args:
        raw_data (dict): Dictionary mapping well_id to a tuple of DataFrames.

    Returns:
        dict: Cleaned, structured, and feature-engineered target matrix dictionaries.
    """
    # Original Initial Preprocessing Step
    data = interpolate_gr(raw_data=raw_data)
    
    # Stage 1: Spatial Transformations
    data = align_geological_contacts(raw_data=data, ref_col="EGFDU")
    data = transform_coordinate_deltas(raw_data=data)
    
    # Stage 2: Resampling & Wave-filtering Operations
    data = denoise_sensors(raw_data=data, r=5)
    data = standardize_reference_grid(raw_data=data, step=0.2)
    data = apply_scale_invariant_normalization(raw_data=data, win=5)
    
    # Stage 3: Feature Architecture Assembly & Memory Compression
    data = extract_dynamic_variance(raw_data=data)
    data = initialize_trend_velocity(raw_data=data)
    data = classify_trajectory_geometry(raw_data=data)
    data = split_sequence_masks(raw_data=data)
    data = enforce_matrix_typecasting(raw_data=data)
    
    return data