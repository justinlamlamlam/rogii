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
    allow_jump: bool = True,
    tvt_range_min: float = 0.0,
    tvt_range_max: float = 50.0,
    tvt_range_penalty_rate: float = 1.0
) -> list:
    """Optimizes the TVT assignment sequence for a horizontal well's evaluation zone.

    Finds a sequence of TVT values that minimizes the cumulative squared error 
    between observed GR and mapped TVT values, incorporating specified transition 
    constraints, jump penalties, and dynamic window range constraints using an 
    A* Beam Search approach with grid binning.

    Args:
        horizontal_df (pd.DataFrame): Dataframe containing the well log data with columns 
            'GR' and 'TVT_input'.
        typewell_df (pd.DataFrame): Reference vertical log dataframe.
        jump_penalty (float): Penalty added to the cost function when making a non-local jump.
        beam_width (int): Maximum number of active states tracked per sequence step to control complexity.
        bin_size (float): The width of each discrete TVT bin used for spatial pruning.
        max_tvt_diff (float): Maximum allowed structural absolute deviation from the starting TVT value.
        allow_jump (bool): Flag to enable or disable non-local jump transitions.
        tvt_range_min (float): The minimum expected dynamic range spread (max_tvt - min_tvt).
        tvt_range_max (float): The maximum expected dynamic range spread (max_tvt - min_tvt).
        tvt_range_penalty_rate (float): Multiplier applied to the penalty cost when constraints are violated.

    Returns:
        list: Optimized TVT assignments matching the length of the evaluation zone sequence.
    """
    # Isolate evaluation zone where target TVT is unknown
    eval_mask = horizontal_df["TVT_input"].isna()
    eval_gr = horizontal_df.loc[eval_mask, "GR"].values
    
    if len(eval_gr) == 0:
        print("Optimization Loss: 0.0")
        print("TVT Range Loss: 0.0")
        print("Min TVT Seen: 0.0")
        print("Max TVT Seen: 0.0")
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
    # State structure: (current_tvt, g_score, path_list, min_tvt_seen, max_tvt_seen, tvt_range_loss)
    current_states = [(tvt_start, 0.0, [], tvt_start, tvt_start, 0.0)]

    # Process each observed Gamma Ray value sequentially
    for gr_obs in tqdm(eval_gr, desc="Optimizing TVT Path"):
        candidates = []
        
        # Branch transitions from each active state in the beam
        for last_tvt, g_score, path, min_tvt_seen, max_tvt_seen, tvt_range_loss in current_states:
            
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
                    
                    # Track running bounds and evaluate dynamic range penalty
                    new_min = min(min_tvt_seen, tvt_next)
                    new_max = max(max_tvt_seen, tvt_next)
                    current_range = new_max - new_min
                    
                    if tvt_range_min <= current_range <= tvt_range_max:
                        range_penalty = 0.0
                    elif current_range < tvt_range_min:
                        range_penalty = abs(max_tvt_diff - tvt_range_min) * tvt_range_penalty_rate
                    else:
                        range_penalty = abs(max_tvt_diff - tvt_range_max) * tvt_range_penalty_rate
                    
                    cost_step = (gr_obs - gr_pred) ** 2 + range_penalty
                    candidates.append((
                        tvt_next, 
                        g_score + cost_step, 
                        path + [tvt_next], 
                        new_min, 
                        new_max, 
                        tvt_range_loss + range_penalty
                    ))

            # --- Operation 2: Jump (Fault / Non-Local Move) ---
            if allow_jump:
                jump_targets = gr_to_tvt(gr_target=gr_obs, typewell_df=typewell_df)
                if isinstance(jump_targets, (int, float)):
                    jump_targets = [jump_targets]
                    
                for tvt_jump in jump_targets:
                    if abs(tvt_jump - tvt_start) <= max_tvt_diff:
                        gr_pred = tvt_to_gr(tvt=tvt_jump, typewell_df=typewell_df)
                        
                        # Track running bounds and evaluate dynamic range penalty
                        new_min = min(min_tvt_seen, tvt_jump)
                        new_max = max(max_tvt_seen, tvt_jump)
                        current_range = new_max - new_min
                        
                        if tvt_range_min <= current_range <= tvt_range_max:
                            range_penalty = 0.0
                        elif current_range < tvt_range_min:
                            range_penalty = abs(max_tvt_diff - tvt_range_min) * tvt_range_penalty_rate
                        else:
                            range_penalty = abs(max_tvt_diff - tvt_range_max) * tvt_range_penalty_rate
                        
                        cost_step = (gr_obs - gr_pred) ** 2 + jump_penalty + range_penalty
                        candidates.append((
                            tvt_jump, 
                            g_score + cost_step, 
                            path + [tvt_jump], 
                            new_min, 
                            new_max, 
                            tvt_range_loss + range_penalty
                        ))

        if not candidates:
            # Fallback if all paths violate max_tvt_diff constraint
            break

        # --- Intra-bin Pruning ---
        # Map states to physical structural grid bins to maintain uniform coverage
        bin_dict = {}
        for tvt, g, p, mn, mx, rl in candidates:
            bin_idx = int(np.floor(tvt / bin_size))
            if bin_idx not in bin_dict or g < bin_dict[bin_idx][1]:
                bin_dict[bin_idx] = (tvt, g, p, mn, mx, rl)

        # --- Inter-bin Pruning (Beam Width restriction) ---
        # Sort surviving discrete candidates by lowest accumulated cost (A* equivalent score)
        sorted_states = sorted(bin_dict.values(), key=lambda x: x[1])
        current_states = sorted_states[:beam_width]

    # Select the optimal sequence containing the minimum total error metric
    best_state = min(current_states, key=lambda x: x[1])
    
    # Print total metrics before returning the sequence
    print(f"Optimization Loss: {best_state[1]}")
    print(f"TVT Range Loss: {best_state[5]}")
    print(f"Min TVT Seen: {best_state[3]}")
    print(f"Max TVT Seen: {best_state[4]}")
    
    return best_state[2]


def optimize_well_tvt_from_base(
    horizontal_df: pd.DataFrame, 
    typewell_df: pd.DataFrame, 
    ml_solution: list,
    beam_width: int = 20,
    bin_size: float = 1.0,
    max_tvt_diff: float = 105.0
) -> list:
    """Optimizes the TVT assignment sequence for a horizontal well's evaluation zone without jumps or ranges, bounded by an ML solution.

    Finds a sequence of TVT values that minimizes the cumulative squared error 
    between observed GR and mapped TVT values using an A* Beam Search approach 
    with grid binning, while keeping each assignment within 1.0 unit of the ml_solution.

    Args:
        horizontal_df (pd.DataFrame): Dataframe containing the well log data with columns 
            'GR' and 'TVT_input'.
        typewell_df (pd.DataFrame): Reference vertical log dataframe.
        ml_solution (list): Baseline ML model TVT predictions matching the length of the evaluation zone.
        beam_width (int): Maximum number of active states tracked per sequence step to control complexity.
        bin_size (float): The width of each discrete TVT bin used for spatial pruning.
        max_tvt_diff (float): Maximum allowed structural absolute deviation from the starting TVT value.

    Returns:
        list: Optimized TVT assignments matching the length of the evaluation zone sequence.
    """
    # Isolate evaluation zone where target TVT is unknown
    eval_mask = horizontal_df["TVT_input"].isna()
    eval_gr = horizontal_df.loc[eval_mask, "GR"].values
    
    if len(eval_gr) == 0:
        print("Optimization Loss: 0.0")
        print("Min TVT Seen: 0.0")
        print("Max TVT Seen: 0.0")
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
    # State structure: (current_tvt, g_score, path_list, min_tvt_seen, max_tvt_seen)
    current_states = [(tvt_start, 0.0, [], tvt_start, tvt_start)]

    # Process each observed Gamma Ray value sequentially
    for idx, gr_obs in enumerate(tqdm(eval_gr, desc="Optimizing TVT Path")):
        candidates = []
        ml_val = ml_solution[idx]
        
        # Branch transitions from each active state in the beam
        for last_tvt, g_score, path, min_tvt_seen, max_tvt_seen in current_states:
            
            # --- Operation 1: Slight Change (Local Move) ---
            # Effective neighborhood constraint window
            local_delta = min(0.1, bin_size)
            
            # Sample boundary points, center of movement window, AND the ML guideline itself 
            # to prevent the beam from dying if the ML solution steps out of the local window.
            local_candidates = [
                last_tvt - local_delta,
                last_tvt,
                last_tvt + local_delta,
                ml_val  # Dynamic catch-up point
            ]
            
            for tvt_next in local_candidates:
                # Core check: limit total structural deviation AND stay within 1.0 of the ML baseline
                if abs(tvt_next - tvt_start) <= max_tvt_diff and abs(tvt_next - ml_val) <= 1.0:
                    gr_pred = tvt_to_gr(tvt=tvt_next, typewell_df=typewell_df)
                    
                    new_min = min(min_tvt_seen, tvt_next)
                    new_max = max(max_tvt_seen, tvt_next)
                    
                    cost_step = (gr_obs - gr_pred) ** 2
                    candidates.append((
                        tvt_next, 
                        g_score + cost_step, 
                        path + [tvt_next], 
                        new_min, 
                        new_max
                    ))

        if not candidates:
            # Fallback if all paths violate max_tvt_diff or ml_solution constraints
            break

        # --- Intra-bin Pruning ---
        # Map states to physical structural grid bins to maintain uniform coverage
        bin_dict = {}
        for tvt, g, p, mn, mx in candidates:
            bin_idx = int(np.floor(tvt / bin_size))
            if bin_idx not in bin_dict or g < bin_dict[bin_idx][1]:
                bin_dict[bin_idx] = (tvt, g, p, mn, mx)

        # --- Inter-bin Pruning (Beam Width restriction) ---
        # Sort surviving discrete candidates by lowest accumulated cost (A* equivalent score)
        sorted_states = sorted(bin_dict.values(), key=lambda x: x[1])
        current_states = sorted_states[:beam_width]

    # Select the optimal sequence containing the minimum total error metric
    best_state = min(current_states, key=lambda x: x[1])
    
    # Print total metrics before returning the sequence
    print(f"Optimization Loss: {best_state[1]}")
    print(f"Min TVT Seen: {best_state[3]}")
    print(f"Max TVT Seen: {best_state[4]}")
    
    return best_state[2]