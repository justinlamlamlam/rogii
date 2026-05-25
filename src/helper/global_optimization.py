# global_opetimization.py

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from helper.math_mapping import forward_map, inverse_set, get_discretized_tvt_grid

class DPState:
    def __init__(self, *, i: int, v: int):
        """
        Represents a state in the dynamic programming grid.

        Args:
            i (int): Current index in the GR point list.
            v (int): The current TVT bin index.
        """
        self.i = i
        self.v = v
    
    def __eq__(self, other):
        return self.i == other.i and self.v == other.v
    
    def __hash__(self):
        return hash((self.i, self.v))
    
    def __repr__(self):
        return f"DPState(i={self.i}, v={self.v})"


class GlobalPathOptimizer:
    def __init__(self, 
                 *,
                 gr_sequence: np.ndarray,
                 typewell_df: pd.DataFrame,
                 tvt_discretization_step: float = 0.01,
                 max_tvt_change_slight: float = 0.1,
                 jump_penalty: float = 4.0):
        """
        Optimizes the TVT path globally across a discretized state-space grid.

        Args:
            gr_sequence (np.ndarray): Sequence of measured GR targets.
            typewell_df (pd.DataFrame): Reference DataFrame containing 'TVT' and 'GR' profiles.
            tvt_discretization_step (float): Step resolution for the TVT grid axis.
            max_tvt_change_slight (float): Max allowed TVT distance for neighborhood transitions.
            jump_penalty (float): Penalty added to cumulative error when a jump occurs.
        """
        
        self.gr_sequence = gr_sequence
        self.typewell_df = typewell_df
        self.tvt_step = tvt_discretization_step
        self.max_tvt_change_slight = max_tvt_change_slight
        self.jump_penalty = jump_penalty
        
        self._setup_tvt_grid()
        
        # DP state tracking: maps (i, v) -> metric values
        self.dp_states: Dict[Tuple[int, int], float] = {}
        self.parent_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.tvt_at_state: Dict[Tuple[int, int], float] = {}
    
    def _setup_tvt_grid(self):
        """Setup the discretized TVT grid and precompute the local neighborhood range."""
        # Use get_discretized_tvt_grid from math_mapping to establish the grid
        self.tvt_grid = get_discretized_tvt_grid(self.typewell_df, step=self.tvt_step)
        self.tvt_to_bin = {float(tvt): idx for idx, tvt in enumerate(self.tvt_grid)}
        
        # --- OPTIMIZATION: Calculate the maximum index offset allowed for slight changes ---
        # Using a small epsilon to safeguard against floating-point precision issues
        self.max_bin_offset = int(np.floor((self.max_tvt_change_slight + 1e-9) / self.tvt_step))
    
    def _tvt_to_bin_index(self, *, tvt: float) -> int:
        idx = np.searchsorted(self.tvt_grid, tvt, side='left')
        return min(idx, len(self.tvt_grid) - 1)
    
    def _bin_index_to_tvt(self, *, bin_idx: int) -> float:
        return float(self.tvt_grid[bin_idx])
    
    def _compute_error(self, *, gr_measured: float, gr_predicted: float) -> float:
        return (gr_predicted - gr_measured) ** 2
    
    def _transition_slight_change(self, 
                                  *,
                                  i: int, 
                                  v: int,
                                  gr_target: float) -> List[Tuple[Tuple[int, int], float, float]]:
        """
        O(1) Neighborhood Expansion: Only look at valid neighboring bins instead of scanning V.

        Args:
            i (int): Current point index.
            v (int): Current TVT bin index.
            gr_target (float): Target GR value at step i + 1.

        Returns:
            List[Tuple[Tuple[int, int], float, float]]: Transition choices containing (next_state_key, error, next_tvt).
        """
        transitions = []
        
        # --- OPTIMIZATION: Compute localized boundary indices ---
        min_bin_idx = max(0, v - self.max_bin_offset)
        max_bin_idx = min(len(self.tvt_grid) - 1, v + self.max_bin_offset)
        
        # Iterate only through the valid local window, eliminating the O(V) scan
        for next_bin_idx in range(min_bin_idx, max_bin_idx + 1):
            next_tvt = self._bin_index_to_tvt(bin_idx=next_bin_idx)
            
            # Call forward_map imported from math_mapping
            gr_predicted = forward_map(next_tvt, self.typewell_df)
            error = self._compute_error(gr_measured=gr_target, gr_predicted=gr_predicted)
            
            next_state_key = (i + 1, next_bin_idx)
            transitions.append((next_state_key, error, next_tvt))
        
        return transitions
    
    def _transition_jump(self, 
                         *,
                         i: int, 
                         gr_target: float) -> List[Tuple[Tuple[int, int], float, float]]:
        """
        Multi-Branching Jump Transition: Identifies all inverse nodes matching target GR and applies a penalty cost.

        Args:
            i (int): Current point index.
            gr_target (float): Target GR value at step i + 1.

        Returns:
            List[Tuple[Tuple[int, int], float, float]]: Transition choices containing (next_state_key, total_error, next_tvt).
        """
        transitions = []
        # Call inverse_set imported from math_mapping
        tvt_candidates = inverse_set(gr_target, self.typewell_df)
        
        for next_tvt in tvt_candidates:
            next_bin_idx = self._tvt_to_bin_index(tvt=next_tvt)
            
            # Call forward_map imported from math_mapping
            gr_predicted = forward_map(next_tvt, self.typewell_df)
            
            # Base tracking error + fixed jump penalty
            error = self._compute_error(gr_measured=gr_target, gr_predicted=gr_predicted) + self.jump_penalty
            
            next_state_key = (i + 1, next_bin_idx)
            transitions.append((next_state_key, error, next_tvt))
        
        return transitions
    
    def run(self) -> Tuple[List[float], float]:
        """
        Executes the forward dynamic programming pass and extracts the optimal global path.
        Prints the structural complexity parameters (N, M, T) and their product before running.

        Returns:
            Tuple[List[float], float]: Formed by a list representing the optimized TVT sequence and a float for the absolute minimum total error.
        """
        n = len(self.gr_sequence)
        m = len(self.tvt_grid)
        t = len(self.typewell_df)
        product_nmt = n * m * t
        
        print(f"DP Complexity Parameters Initialization:")
        print(f"  N (Sequence Length)       = {n}")
        print(f"  M (Discretized Grid Bins) = {m}")
        print(f"  T (Typewell Node Rows)    = {t}")
        print(f"  N * M * T Upper Bound     = {product_nmt}")
        
        gr_target_0 = self.gr_sequence[0]
        # Call inverse_set imported from math_mapping
        tvt_candidates_0 = inverse_set(gr_target_0, self.typewell_df)
        
        for tvt_val in tvt_candidates_0:
            bin_idx = self._tvt_to_bin_index(tvt=tvt_val)
            
            # Call forward_map imported from math_mapping
            gr_predicted = forward_map(tvt_val, self.typewell_df)
            error = self._compute_error(gr_measured=gr_target_0, gr_predicted=gr_predicted)
            
            state_key = (0, bin_idx)
            self.dp_states[state_key] = error
            self.tvt_at_state[state_key] = tvt_val
        
        for i in tqdm(range(n - 1)):
            gr_target = self.gr_sequence[i + 1]
            
            active_states_at_i = [
                v for (idx, v) in self.dp_states.keys() if idx == i
            ]
            
            for v in active_states_at_i:
                current_state_key = (i, v)
                current_error = self.dp_states[current_state_key]
                
                # Option 1: Slight Change Neighborhood Exploration
                slight_transitions = self._transition_slight_change(i=i, v=v, gr_target=gr_target)
                for next_state_key, transition_error, next_tvt in slight_transitions:
                    total_error = current_error + transition_error
                    
                    if next_state_key not in self.dp_states or total_error < self.dp_states[next_state_key]:
                        self.dp_states[next_state_key] = total_error
                        self.parent_map[next_state_key] = current_state_key
                        self.tvt_at_state[next_state_key] = next_tvt
                
                # Option 2: Discretized Global Jump Branching (Unconditional execution per state)
                jump_transitions = self._transition_jump(i=i, gr_target=gr_target)
                for next_state_key, transition_error, next_tvt in jump_transitions:
                    total_error = current_error + transition_error
                    
                    if next_state_key not in self.dp_states or total_error < self.dp_states[next_state_key]:
                        self.dp_states[next_state_key] = total_error
                        self.parent_map[next_state_key] = current_state_key
                        self.tvt_at_state[next_state_key] = next_tvt
        
        final_states_at_n_minus_1 = [
            ((i, v), error) for (i, v), error in self.dp_states.items() if i == n - 1
        ]
        
        if not final_states_at_n_minus_1:
            raise ValueError("No valid path found in DP optimization.")
        
        best_final_state, min_error = min(final_states_at_n_minus_1, key=lambda x: x[1])
        tvt_sequence = self._reconstruct_path(final_state=best_final_state)
        
        return tvt_sequence, min_error

    def _reconstruct_path(self, *, final_state: Tuple[int, int]) -> List[float]:
        """
        Traces back from the absolute minimal end state to get the comprehensive optimal TVT path.

        Args:
            final_state (Tuple[int, int]): Optimal final state descriptor key (n-1, v).

        Returns:
            List[float]: Ordered sequence of optimized TVT data points.
        """
        path = []
        current_state = final_state
        
        while current_state in self.tvt_at_state:
            path.append(self.tvt_at_state[current_state])
            if current_state not in self.parent_map:
                break
            current_state = self.parent_map[current_state]
        
        path.reverse()
        return path