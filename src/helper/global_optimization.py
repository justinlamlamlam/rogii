import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from helper.math_mapping import tvt_to_gr, gr_to_tvt

def optimize_well_tvt(
    horizontal_df: pd.DataFrame, 
    typewell_df: pd.DataFrame, 
    jump_penalty: float = 50.0,
    beam_width: int = 20,
    bin_size: float = 1.0,
    max_tvt_diff: float = 105.0,
    allow_jump: bool = True
) -> list:
    """Optimizes the TVT assignment sequence for a horizontal well's evaluation zone.

    Finds a sequence of TVT values that minimizes the cumulative squared error 
    between observed GR and mapped TVT values, incorporating specified transition 
    constraints and jump penalties using an A* Beam Search approach with grid binning.

    Args:
        horizontal_df (pd.DataFrame): Dataframe containing the well log data with columns 
            'GR' and 'TVT_input'.
        typewell_df (pd.DataFrame): Reference vertical log dataframe.
        jump_penalty (float): Penalty added to the cost function when making a non-local jump.
        beam_width (int): Maximum number of active states tracked per sequence step to control complexity.
        bin_size (float): The width of each discrete TVT bin used for spatial pruning.
        max_tvt_diff (float): Maximum allowed structural absolute deviation from the starting TVT value.
        allow_jump (bool): Flag to enable or disable non-local jump transitions.

    Returns:
        list: Optimized TVT assignments matching the length of the evaluation zone sequence.
    """
    # Isolate evaluation zone where target TVT is unknown
    eval_mask = horizontal_df["TVT_input"].isna()
    eval_gr = horizontal_df.loc[eval_mask, "GR"].values
    
    if len(eval_gr) == 0:
        return []

    # Identify the starting anchor TVT right before the evaluation zone
    pre_eval_idx = horizontal_df[~eval_mask].index
    if len(pre_eval_idx) > 0:
        tvt_start = horizontal_df.loc[pre_eval_idx[-1], "TVT_input"]
    else:
        # Fallback if evaluation zone starts at index 0
        first_gr = eval_gr[0]
        possible_starts = gr_to_tvt(gr_target=first_gr, typewell_df=typewell_df)
        tvt_start = possible_starts[0] if len(possible_starts) > 0 else 0.0

    # Initialize beam pool with the starting state
    # State structure: (current_tvt, g_score, path_list)
    current_states = [(tvt_start, 0.0, [])]

    # Process each observed Gamma Ray value sequentially
    for gr_obs in tqdm(eval_gr, desc="Optimizing TVT Path"):
        candidates = []
        
        # Branch transitions from each active state in the beam
        for last_tvt, g_score, path in current_states:
            
            # --- Operation 1: Slight Change (Local Move) ---
            # Effective neighborhood constraint window
            local_delta = min(0.1, bin_size)
            
            # Sample boundary points and the center of the local movement window
            local_candidates = [
                last_tvt - local_delta,
                last_tvt,
                last_tvt + local_delta
            ]
            
            for tvt_next in local_candidates:
                if abs(tvt_next - tvt_start) <= max_tvt_diff:
                    gr_pred = tvt_to_gr(tvt=tvt_next, typewell_df=typewell_df)
                    cost_step = (gr_obs - gr_pred) ** 2
                    candidates.append((tvt_next, g_score + cost_step, path + [tvt_next]))

            # --- Operation 2: Jump (Fault / Non-Local Move) ---
            if allow_jump:
                jump_targets = gr_to_tvt(gr_target=gr_obs, typewell_df=typewell_df)
                if isinstance(jump_targets, (int, float)):
                    jump_targets = [jump_targets]
                    
                for tvt_jump in jump_targets:
                    if abs(tvt_jump - tvt_start) <= max_tvt_diff:
                        gr_pred = tvt_to_gr(tvt=tvt_jump, typewell_df=typewell_df)
                        cost_step = (gr_obs - gr_pred) ** 2 + jump_penalty
                        candidates.append((tvt_jump, g_score + cost_step, path + [tvt_jump]))

        if not candidates:
            # Fallback if all paths violate max_tvt_diff constraint
            break

        # --- Intra-bin Pruning ---
        # Map states to physical structural grid bins to maintain uniform coverage
        bin_dict = {}
        for tvt, g, p in candidates:
            bin_idx = int(np.floor(tvt / bin_size))
            if bin_idx not in bin_dict or g < bin_dict[bin_idx][1]:
                bin_dict[bin_idx] = (tvt, g, p)

        # --- Inter-bin Pruning (Beam Width restriction) ---
        # Sort surviving discrete candidates by lowest accumulated cost (A* equivalent score)
        sorted_states = sorted(bin_dict.values(), key=lambda x: x[1])
        current_states = sorted_states[:beam_width]

    # Select the optimal sequence containing the minimum total error metric
    best_state = min(current_states, key=lambda x: x[1])
    return best_state[2]


def default_optimize_well_tvt(
    horizontal_df: pd.DataFrame, 
    typewell_df: pd.DataFrame, 
    jump_penalty: float = 50.0,
    beam_width: int = 20
) -> list:
    """Optimizes the TVT assignment sequence for a horizontal well's evaluation zone.

    Finds a sequence of TVT values that minimizes the cumulative squared error 
    between observed GR and mapped TVT values, incorporating specified transition 
    constraints and jump penalties.

    Args:
        horizontal_df (pd.DataFrame): Dataframe containing the well log data with columns 
            'GR' and 'TVT_input'.
        typewell_df (pd.DataFrame): Reference vertical log dataframe.
        jump_penalty (float): Penalty added to the cost function when making a non-local jump.
        beam_width (int): Maximum number of active states tracked per sequence step to control complexity.

    Returns:
        list: Optimized TVT assignments matching the length of the evaluation zone sequence.
    """
    # Identify the evaluation zone (where TVT_input is missing)
    eval_mask = horizontal_df["TVT_input"].isna()
    if not eval_mask.any():
        return []

    # Locate the starting anchor point (the last valid TVT entry before evaluation zone)
    valid_indices = horizontal_df.index[~eval_mask]
    if len(valid_indices) == 0:
        raise ValueError("No valid starting TVT point found in horizontal_df['TVT_input'].")
    
    last_valid_idx = valid_indices[-1]
    tvt_0 = float(horizontal_df.loc[last_valid_idx, "TVT_input"])
    
    # Extract evaluation GR sequence
    eval_gr = horizontal_df.loc[eval_mask, "GR"].to_numpy()
    
    # Track states: { current_tvt: (cumulative_cost, path_history_list) }
    active_states = {tvt_0: (0.0, [])}

    for gr_val in tqdm(eval_gr, desc="Optimizing TVT Path"):
        next_states = {}
        
        # Get all mapped target candidates for the current GR point
        candidates = gr_to_tvt(gr_val, typewell_df)
        if not isinstance(candidates, (list, np.ndarray)):
            candidates = [candidates]
        candidates = [float(c) for c in candidates]
        
        for last_tvt, (current_cost, path) in active_states.items():
            
            # 1. Determine "Slight change" options
            slight_change_options = []
            forced_matches = [c for c in candidates if abs(c - last_tvt) < 0.1]
            
            if len(forced_matches) > 0:
                # Forced Match Rule: Use matching candidates within range
                slight_change_options = forced_matches
            else:
                # Fallback Rule: Fixed offset movements (modified to 0 movement only)
                slight_change_options = [last_tvt]
                
            # Process Slight Changes (Penalty = 0)
            for next_tvt in slight_change_options:
                mapped_gr = tvt_to_gr(next_tvt, typewell_df)
                step_cost = (gr_val - mapped_gr) ** 2
                total_cost = current_cost + step_cost
                
                if next_tvt not in next_states or total_cost < next_states[next_tvt][0]:
                    next_states[next_tvt] = (total_cost, path + [next_tvt])
                    
        # Apply Beam Width Pruning to keep execution runtime and memory linear
        if len(next_states) > beam_width:
            sorted_states = sorted(next_states.items(), key=lambda item: item[1][0])
            active_states = dict(sorted_states[:beam_width])
        else:
            active_states = next_states

    # Select path containing lowest global error score
    best_tvt_sequence = min(active_states.values(), key=lambda item: item[0])[1]
    
    return best_tvt_sequence