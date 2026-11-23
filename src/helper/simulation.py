# simulation.py

import numpy as np
import pandas as pd
from numba import njit

# ==============================================================================
# JIT ACCELERATED CORE ALGORITHMS
# ==============================================================================

@njit
def _pf_ancc(
    gr_v: np.ndarray, 
    dz_v: np.ndarray, 
    dm_v: np.ndarray, 
    tw_gr: np.ndarray, 
    tmin: float, 
    tstep: float, 
    init_pos: float, 
    init_rate: float, 
    gr_sig: float, 
    n_part: int = 2000, 
    seed: int = 42
) -> np.ndarray:
    """
    Executes a JIT-compiled Particle Filter tracking routine tailored for the ANCC formation structure.

    Args:
        gr_v (np.ndarray): Array of horizontal well Gamma Ray sensor readings.
        dz_v (np.ndarray): Array of absolute vertical tracking depth deltas.
        dm_v (np.ndarray): Array of absolute measured depth along-hole deltas.
        tw_gr (np.ndarray): Standardized reference typewell Gamma Ray array grid.
        tmin (float): Minimum threshold vertical depth boundary mapping score.
        tstep (float): Resolution steps spacing metric for vertical array tracking.
        init_pos (float): Starting absolute true vertical thickness baseline position.
        init_rate (float): Initial structural momentum rate profile.
        gr_sig (float): Estimated background measurement variance noise envelope parameter.
        n_part (int): Total active particle population size. Defaults to 2000.
        seed (int): Random generator validation seed integer value. Defaults to 42.

    Returns:
        np.ndarray: Predicted single-seed tracking coordinates trajectory vector array.
    """
    np.random.seed(seed)
    n_steps = len(gr_v)
    pos = np.full(shape=n_part, fill_value=init_pos, dtype=np.float64)
    rate = np.full(shape=n_part, fill_value=init_rate, dtype=np.float64)
    out = np.zeros(shape=n_steps, dtype=np.float64)
    
    tw_len = len(tw_gr)
    max_idx = tw_len - 1
    
    for i in range(n_steps):
        gr_obs = gr_v[i]
        dz = dz_v[i]
        dm = dm_v[i]
        
        # Propagation Step
        rate += np.random.normal(loc=0.0, scale=0.005, size=n_part)
        pos += (rate * dm) - dz + np.random.normal(loc=0.0, scale=0.1, size=n_part)
        
        # Weighting Strategy
        weights = np.zeros(shape=n_part, dtype=np.float64)
        for p in range(n_part):
            p_pos = pos[p]
            idx = int(round((p_pos - tmin) / tstep))
            if idx < 0:
                idx = 0
            elif idx > max_idx:
                idx = max_idx
            gr_est = tw_gr[idx]
            diff = gr_obs - gr_est
            weights[p] = np.exp(-0.5 * (diff / gr_sig) ** 2)
            
        w_sum = np.sum(weights)
        if w_sum > 1e-12:
            weights /= w_sum
        else:
            weights = np.full(shape=n_part, fill_value=1.0 / n_part, dtype=np.float64)
            
        # Resampling Logic Execution
        eff_size = 1.0 / np.sum(weights ** 2)
        if eff_size < n_part * 0.5:
            indices = np.searchsorted(np.cumsum(weights), np.random.random(n_part))
            pos = pos[indices]
            rate = rate[indices]
            
        out[i] = np.median(pos)
        
    return out

@njit
def _pf_z(
    gr_v: np.ndarray, 
    dz_v: np.ndarray, 
    dm_v: np.ndarray, 
    tw_gr: np.ndarray, 
    tmin: float, 
    tstep: float, 
    init_pos: float, 
    init_rate: float, 
    gr_sig: float, 
    n_part: int = 2000, 
    seed: int = 42
) -> np.ndarray:
    """
    Executes a JIT-compiled Particle Filter tracking routine tailored for standard vertical formations.

    Args:
        gr_v (np.ndarray): Array of horizontal well Gamma Ray sensor readings.
        dz_v (np.ndarray): Array of absolute vertical tracking depth deltas.
        dm_v (np.ndarray): Array of absolute measured depth along-hole deltas.
        tw_gr (np.ndarray): Standardized reference typewell Gamma Ray array grid.
        tmin (float): Minimum threshold vertical depth boundary mapping score.
        tstep (float): Resolution steps spacing metric for vertical array tracking.
        init_pos (float): Starting absolute true vertical thickness baseline position.
        init_rate (float): Initial structural momentum rate profile.
        gr_sig (float): Estimated background measurement variance noise envelope parameter.
        n_part (int): Total active particle population size. Defaults to 2000.
        seed (int): Random generator validation seed integer value. Defaults to 42.

    Returns:
        np.ndarray: Predicted single-seed tracking coordinates trajectory vector array.
    """
    np.random.seed(seed)
    n_steps = len(gr_v)
    pos = np.full(shape=n_part, fill_value=init_pos, dtype=np.float64)
    rate = np.full(shape=n_part, fill_value=init_rate, dtype=np.float64)
    out = np.zeros(shape=n_steps, dtype=np.float64)
    
    tw_len = len(tw_gr)
    max_idx = tw_len - 1
    
    for i in range(n_steps):
        gr_obs = gr_v[i]
        dz = dz_v[i]
        dm = dm_v[i]
        
        # Standard Motion Propagation Block
        rate += np.random.normal(loc=0.0, scale=0.01, size=n_part)
        pos += (rate * dm) - dz + np.random.normal(loc=0.0, scale=0.2, size=n_part)
        
        weights = np.zeros(shape=n_part, dtype=np.float64)
        for p in range(n_part):
            p_pos = pos[p]
            idx = int(round((p_pos - tmin) / tstep))
            if idx < 0:
                idx = 0
            elif idx > max_idx:
                idx = max_idx
            gr_est = tw_gr[idx]
            diff = gr_obs - gr_est
            weights[p] = np.exp(-0.5 * (diff / gr_sig) ** 2)
            
        w_sum = np.sum(weights)
        if w_sum > 1e-12:
            weights /= w_sum
        else:
            weights = np.full(shape=n_part, fill_value=1.0 / n_part, dtype=np.float64)
            
        eff_size = 1.0 / np.sum(weights ** 2)
        if eff_size < n_part * 0.5:
            indices = np.searchsorted(np.cumsum(weights), np.random.random(n_part))
            pos = pos[indices]
            rate = rate[indices]
            
        out[i] = np.mean(pos)
        
    return out

@njit
def _beam_jit(
    gr_v: np.ndarray, 
    dz_v: np.ndarray, 
    dm_v: np.ndarray, 
    tw_gr: np.ndarray, 
    tmin: float, 
    tstep: float, 
    init_pos: float, 
    init_rate: float, 
    beam_width: int, 
    jump_penalty: float
) -> np.ndarray:
    """
    Performs JIT-compiled deterministic cost-minimization beam path optimization across geological horizons.

    Args:
        gr_v (np.ndarray): Matrix log array storing horizontal Gamma Ray sensors readings.
        dz_v (np.ndarray): Path step progression delta metrics of physical vertical elevation.
        dm_v (np.ndarray): Forward horizontal distance interval measurement steps delta records.
        tw_gr (np.ndarray): Gridded reference matrix sequence of standard typewell.
        tmin (float): Depth reference boundary bottom index coordinates offset scalar.
        tstep (float): Resampling spacing standard profile parameter metric.
        init_pos (float): Absolute vertical depth configuration marker initialization baseline value.
        init_rate (float): Trajectory displacement momentum projection setting values parameter.
        beam_width (int): Total horizontal candidate path limits processed simultaneously per sequence.
        jump_penalty (float): Error score multiplier penalty enforcing structural path step continuity.

    Returns:
        np.ndarray: Optimized spatial prediction trajectory tracking sequence path.
    """
    n_steps = len(gr_v)
    tw_len = len(tw_gr)
    max_idx = tw_len - 1
    
    # Initialize Tracking Path Beam Containers
    active_paths = np.full(shape=(beam_width, n_steps), fill_value=init_pos, dtype=np.float64)
    active_rates = np.full(shape=beam_width, fill_value=init_rate, dtype=np.float64)
    active_costs = np.zeros(shape=beam_width, dtype=np.float64)
    
    for i in range(n_steps):
        gr_obs = gr_v[i]
        dz = dz_v[i]
        dm = dm_v[i]
        
        # Temporary Candidate Tracking Buffers
        next_paths = np.zeros(shape=(beam_width * 3, n_steps), dtype=np.float64)
        next_rates = np.zeros(shape=beam_width * 3, dtype=np.float64)
        next_costs = np.full(shape=beam_width * 3, fill_value=1e18, dtype=np.float64)
        
        idx_cand = 0
        for b in range(beam_width):
            curr_pos = active_paths[b, max(0, i-1)] if i > 0 else init_pos
            curr_rate = active_rates[b]
            curr_cost = active_costs[b]
            
            # Formulate 3 Distinct Directional Structural Variations Moves (Down, Flat, Up)
            for deviation in [-1.0, 0.0, 1.0]:
                cand_rate = curr_rate + (deviation * 0.01)
                cand_pos = curr_pos + (cand_rate * dm) - dz
                
                # Boundary Checking Constraint Logic
                g_idx = int(round((cand_pos - tmin) / tstep))
                if g_idx < 0:
                    g_idx = 0
                elif g_idx > max_idx:
                    g_idx = max_idx
                    
                gr_est = tw_gr[g_idx]
                cost_delta = (gr_obs - gr_est) ** 2 + (abs(deviation) * jump_penalty)
                
                # Package Into Multi-Branch Expansion Matrix
                if i > 0:
                    next_paths[idx_cand, :i] = active_paths[b, :i]
                next_paths[idx_cand, i] = cand_pos
                next_rates[idx_cand] = cand_rate
                next_costs[idx_cand] = curr_cost + cost_delta
                idx_cand += 1
                
        # Sort Array Candidates and Select Best Beam Configurations
        best_indices = np.argsort(next_costs)[:beam_width]
        for b in range(beam_width):
            chosen = best_indices[b]
            active_paths[b, :] = next_paths[chosen, :]
            active_rates[b] = next_rates[chosen]
            active_costs[b] = next_costs[chosen]
            
    return active_paths[0, :]

# ==============================================================================
# ENSEMBLE EXECUTIONS MANAGERS
# ==============================================================================

def run_particle_filter(
    horizontal_well: pd.DataFrame, 
    typewell: pd.DataFrame, 
    variant: str = "z", 
    n_part: int = 2000, 
    seed: int = 42
) -> np.ndarray:
    """
    Extracts positional vectors and executes the corresponding pre-compiled Particle Filter tracking engine.

    Args:
        horizontal_well (pd.DataFrame): Data container for along-hole tracking features.
        typewell (pd.DataFrame): Reference vertical log table profile.
        variant (str): Variant key indicator specifying structural settings ('ancc' or 'z'). Defaults to 'z'.
        n_part (int): Total simulation count arrays threshold parameters scale. Defaults to 2000.
        seed (int): Reproducibility initializer parameter tracker. Defaults to 42.

    Returns:
        np.ndarray: Evaluated continuous TVT tracking line coordinates series.
    """
    meta = horizontal_well.attrs.get("grid_meta", {})
    tw_gr = meta.get("gr_grid", typewell["GR"].values.astype(np.float64))
    tmin = meta.get("tmin", float(typewell["TVT"].min()))
    tstep = meta.get("tstep", 0.2)
    
    eval_mask = horizontal_well["is_evaluation_zone"] == 1
    gr_v = horizontal_well.loc[eval_mask, "GR_smooth"].values.astype(np.float64)
    dz_v = horizontal_well.loc[eval_mask, "dZ"].values.astype(np.float64)
    dm_v = horizontal_well.loc[eval_mask, "dMD"].values.astype(np.float64)
    
    # Anchor Point Configurations Setup
    kn = horizontal_well[horizontal_well["is_evaluation_zone"] == 0]
    init_pos = float(kn["TVT_input"].iloc[-1]) if not kn.empty else float(typewell["TVT"].median())
    init_rate = horizontal_well.attrs.get("initial_trend", 0.0)
    gr_sig = horizontal_well.attrs.get("gr_sig", 30.0)
    
    if variant == "ancc":
        return _pf_ancc(
            gr_v=gr_v, dz_v=dz_v, dm_v=dm_v, tw_gr=tw_gr, tmin=tmin, tstep=tstep,
            init_pos=init_pos, init_rate=init_rate, gr_sig=gr_sig, n_part=n_part, seed=seed
        )
    else:
        return _pf_z(
            gr_v=gr_v, dz_v=dz_v, dm_v=dm_v, tw_gr=tw_gr, tmin=tmin, tstep=tstep,
            init_pos=init_pos, init_rate=init_rate, gr_sig=gr_sig, n_part=n_part, seed=seed
        )

def run_pf_lik_ensemble(
    horizontal_well: pd.DataFrame, 
    typewell: pd.DataFrame, 
    variant: str = "z", 
    n_seeds: int = 16
) -> np.ndarray:
    """
    Executes multiple Particle Filter simulation runs across a collection of seeds and merges them.

    Args:
        horizontal_well (pd.DataFrame): Input table for the horizontal well.
        typewell (pd.DataFrame): Data container storing baseline typewell settings.
        variant (str): Target formation tracking setup logic flag. Defaults to 'z'.
        n_seeds (int): Iterative computational limit setting for seed scale parameters. Defaults to 16.

    Returns:
        np.ndarray: Integrated consensus trajectory vector results array.
    """
    results = []
    for s in range(n_seeds):
        pred_path = run_particle_filter(
            horizontal_well=horizontal_well, typewell=typewell, variant=variant, n_part=1000, seed=s + 100
        )
        results.append(pred_path)
    return np.mean(a=np.array(object=results), axis=0)

def run_beam_ensemble(
    horizontal_well: pd.DataFrame, 
    typewell: pd.DataFrame
) -> np.ndarray:
    """
    Runs an array of varying deterministic configuration options through Beam Search tracking and takes the mean.

    Args:
        horizontal_well (pd.DataFrame): Log series tracking measurements framework dataframe.
        typewell (pd.DataFrame): Mapping sheet database referencing true coordinate limits.

    Returns:
        np.ndarray: Combined optimal cost prediction layout path.
    """
    meta = horizontal_well.attrs.get("grid_meta", {})
    tw_gr = meta.get("gr_grid", typewell["GR"].values.astype(np.float64))
    tmin = meta.get("tmin", float(typewell["TVT"].min()))
    tstep = meta.get("tstep", 0.2)
    
    eval_mask = horizontal_well["is_evaluation_zone"] == 1
    gr_v = horizontal_well.loc[eval_mask, "GR_smooth"].values.astype(np.float64)
    dz_v = horizontal_well.loc[eval_mask, "dZ"].values.astype(np.float64)
    dm_v = horizontal_well.loc[eval_mask, "dMD"].values.astype(np.float64)
    
    kn = horizontal_well[horizontal_well["is_evaluation_zone"] == 0]
    init_pos = float(kn["TVT_input"].iloc[-1]) if not kn.empty else float(typewell["TVT"].median())
    init_rate = horizontal_well.attrs.get("initial_trend", 0.0)
    
    # Set explicit parameter matrix variations to establish search bounds
    configs = [
        {"width": 5, "penalty": 10.0},
        {"width": 10, "penalty": 25.0},
        {"width": 15, "penalty": 50.0}
    ]
    
    beam_paths = []
    for conf in configs:
        path = _beam_jit(
            gr_v=gr_v, dz_v=dz_v, dm_v=dm_v, tw_gr=tw_gr, tmin=tmin, tstep=tstep,
            init_pos=init_pos, init_rate=init_rate, beam_width=conf["width"], jump_penalty=conf["penalty"]
        )
        beam_paths.append(path)
        
    return np.mean(a=np.array(object=beam_paths), axis=0)

# ==============================================================================
# STRATEGY SELECTION & ROUTING BLENDER
# ==============================================================================

def apply_selector_variant(
    horizontal_well: pd.DataFrame, 
    typewell: pd.DataFrame, 
    formation_lbl: str = "OTHER"
) -> np.ndarray:
    """
    Applies the appropriate simulation tracking strategy based on the formation label 
    and geometric complexity code of the wellbore trajectory.

    Args:
        horizontal_well (pd.DataFrame): Data framework holding targeted well configurations.
        typewell (pd.DataFrame): Target vertical reference typewell profile matrix.
        formation_lbl (str): Structural classification tag of regional zones. Defaults to 'OTHER'.

    Returns:
        np.ndarray: Blended array containing the finalized continuous TVT numerical tracking predictions.
    """
    variant_key = "ancc" if formation_lbl == "ANCC" else "z"
    selector_code = horizontal_well.attrs.get("selector_code", 0)
    
    # Execute Base Simulation Engines Tracking Methods
    pf_path = run_pf_lik_ensemble(horizontal_well=horizontal_well, typewell=typewell, variant=variant_key, n_seeds=16)
    beam_path = run_beam_ensemble(horizontal_well=horizontal_well, typewell=typewell)
    
    # Establish default fallback baseline values anchor points
    kn = horizontal_well[horizontal_well["is_evaluation_zone"] == 0]
    fallback_val = float(kn["TVT_input"].iloc[-1]) if not kn.empty else float(typewell["TVT"].median())
    fallback_path = np.full(shape=len(pf_path), fill_value=fallback_val, dtype=np.float64)
    
    # Blend outputs based on the geometric complexity code
    if selector_code == 0:
        final_prediction = (pf_path * 0.70) + (beam_path * 0.30)
    elif selector_code == 1:
        final_prediction = (pf_path * 0.50) + (beam_path * 0.50)
    elif selector_code == 2:
        final_prediction = (pf_path * 0.80) + (fallback_path * 0.20)
    elif selector_code == 3:
        final_prediction = (beam_path * 0.90) + (pf_path * 0.10)
    else:
        final_prediction = (pf_path * 0.60) + (beam_path * 0.40)
        
    return final_prediction