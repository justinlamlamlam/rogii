# global_optimization.py

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class DPState:
    """
    Represents a state in the DP optimization.
    
    Attributes:
        i (int): Current index in the GR point list.
        j (int): Total jumps consumed (0 to 5).
        s (int): Points elapsed since last jump (capped at 50).
        v (int): TVT bin index.
    """
    
    def __init__(self, i: int, j: int, s: int, v: int):
        self.i = i
        self.j = j
        self.s = s
        self.v = v
    
    def __eq__(self, other):
        return (self.i == other.i and self.j == other.j and 
                self.s == other.s and self.v == other.v)
    
    def __hash__(self):
        return hash((self.i, self.j, self.s, self.v))
    
    def __repr__(self):
        return f"DPState(i={self.i}, j={self.j}, s={self.s}, v={self.v})"


class GlobalPathOptimizer:
    """
    Implements the global path optimization using dynamic programming
    with discretized TVT state space.
    """
    
    def __init__(self, 
                 gr_sequence: np.ndarray,
                 typewell_df: pd.DataFrame,
                 tvt_discretization_step: float = 0.01,
                 max_tvt_change_slight: float = 0.1,
                 max_points_without_jump: int = 50,
                 max_jumps: int = 5):
        """
        Initialize the optimizer.
        
        Args:
            gr_sequence (np.ndarray): Array of GR values to fit.
            typewell_df (pd.DataFrame): Lookup table with 'TVT' and 'GR' columns.
            tvt_discretization_step (float): Step size for TVT discretization.
            max_tvt_change_slight (float): Max TVT change for "slight change" operation.
            max_points_without_jump (int): Max points before requiring a jump.
            max_jumps (int): Maximum number of jumps allowed.
        """
        self.gr_sequence = gr_sequence
        self.typewell_df = typewell_df
        self.tvt_step = tvt_discretization_step
        self.max_tvt_change_slight = max_tvt_change_slight
        self.max_points_without_jump = max_points_without_jump
        self.max_jumps = max_jumps
        
        # Precompute the TVT grid and related lookups
        self._setup_tvt_grid()
        
        # DP state tracking
        self.dp_states: Dict[Tuple[int, int, int, int], float] = {}  # (i,j,s,v) -> cumulative error
        self.parent_map: Dict[Tuple[int, int, int, int], Tuple[int, int, int, int]] = {}  # (i,j,s,v) -> parent (i,j,s,v)
        self.tvt_at_state: Dict[Tuple[int, int, int, int], float] = {}  # (i,j,s,v) -> TVT value
    
    def _setup_tvt_grid(self):
        """Setup the discretized TVT grid."""
        min_tvt = self.typewell_df['TVT'].min()
        max_tvt = self.typewell_df['TVT'].max()
        self.tvt_grid = np.arange(
            start=min_tvt, 
            stop=max_tvt + self.tvt_step, 
            step=self.tvt_step
        )
        
        # Create lookup: TVT value -> bin index
        self.tvt_to_bin = {float(tvt): idx for idx, tvt in enumerate(self.tvt_grid)}
    
    def _tvt_to_bin_index(self, tvt: float) -> int:
        """Convert TVT value to bin index (nearest bin)."""
        idx = np.searchsorted(self.tvt_grid, tvt, side='left')
        return min(idx, len(self.tvt_grid) - 1)
    
    def _bin_index_to_tvt(self, bin_idx: int) -> float:
        """Convert bin index to TVT value."""
        return float(self.tvt_grid[bin_idx])
    
    def _forward_map(self, tvt: float) -> float:
        """
        Map TVT to GR using linear interpolation on the typewell table.
        
        Args:
            tvt (float): TVT value.
        
        Returns:
            float: Interpolated GR value.
        """
        gr_interpolated = float(np.interp(
            x=tvt, 
            xp=self.typewell_df['TVT'].values, 
            fp=self.typewell_df['GR'].values
        ))
        return gr_interpolated
    
    def _inverse_set(self, gr_target: float) -> List[float]:
        """
        Find all TVT values that map to a given GR target.
        
        Args:
            gr_target (float): Target GR value.
        
        Returns:
            List[float]: List of TVT candidates.
        """
        tvt_candidates = []
        tvt_nodes = self.typewell_df['TVT'].values
        gr_nodes = self.typewell_df['GR'].values
        
        for i in range(len(tvt_nodes) - 1):
            tvt1, tvt2 = tvt_nodes[i], tvt_nodes[i+1]
            gr1, gr2 = gr_nodes[i], gr_nodes[i+1]
            
            if min(gr1, gr2) <= gr_target <= max(gr1, gr2):
                if gr1 == gr2:
                    if gr1 == gr_target:
                        tvt_candidates.append(float(tvt1))
                else:
                    t = (gr_target - gr1) / (gr2 - gr1)
                    tvt_val = tvt1 + t * (tvt2 - tvt1)
                    tvt_candidates.append(float(tvt_val))
        
        return sorted(list(set(tvt_candidates)))
    
    def _is_reachable_slight_change(self, tvt_old: float, tvt_new: float) -> bool:
        """
        Check if we can transition from tvt_old to tvt_new via slight change.
        
        Args:
            tvt_old (float): Old TVT value.
            tvt_new (float): New TVT value.
        
        Returns:
            bool: True if transition is allowed.
        """
        return abs(tvt_new - tvt_old) < self.max_tvt_change_slight
    
    def _compute_error(self, gr_measured: float, gr_predicted: float) -> float:
        """Compute squared error."""
        return (gr_predicted - gr_measured) ** 2
    
    def _transition_slight_change(self, 
                                   i: int, 
                                   j: int, 
                                   s: int, 
                                   v: int,
                                   gr_target: float) -> List[Tuple[Tuple[int, int, int, int], float, float]]:
        """
        Explore transitions via slight change (neighborhood expansion).
        
        Args:
            i, j, s, v: Current state coordinates.
            gr_target: Target GR value for the next point.
        
        Returns:
            List of (next_state_key, error, next_tvt_value).
        """
        transitions = []
        current_tvt = self._bin_index_to_tvt(v)
        
        # Explore bins within the reachable TVT range
        for next_bin_idx in range(len(self.tvt_grid)):
            next_tvt = self._bin_index_to_tvt(next_bin_idx)
            
            if self._is_reachable_slight_change(current_tvt, next_tvt):
                gr_predicted = self._forward_map(next_tvt)
                error = self._compute_error(gr_target, gr_predicted)
                
                # Next state: index advances, jumps stay same, points increment, new bin
                next_state_key = (i + 1, j, min(s + 1, self.max_points_without_jump), next_bin_idx)
                transitions.append((next_state_key, error, next_tvt))
        
        return transitions
    
    def _transition_jump(self, 
                         i: int, 
                         j: int, 
                         s: int,
                         gr_target: float) -> List[Tuple[Tuple[int, int, int, int], float, float]]:
        """
        Explore transitions via jump (multi-branching to all inverse set values).
        
        Args:
            i, j, s: Current state coordinates.
            gr_target: Target GR value for the next point.
        
        Returns:
            List of (next_state_key, error, next_tvt_value).
        """
        transitions = []
        
        # Find all TVT values that map to gr_target
        tvt_candidates = self._inverse_set(gr_target)
        
        for next_tvt in tvt_candidates:
            next_bin_idx = self._tvt_to_bin_index(next_tvt)
            gr_predicted = self._forward_map(next_tvt)
            error = self._compute_error(gr_target, gr_predicted)
            
            # Next state: index advances, jump increments, points reset, new bin
            next_state_key = (i + 1, j + 1, 0, next_bin_idx)
            transitions.append((next_state_key, error, next_tvt))
        
        return transitions
    
    def run(self) -> Tuple[List[float], float]:
        """
        Run the DP optimization algorithm.
        
        Returns:
            Tuple of (tvt_sequence, total_error).
        """
        n = len(self.gr_sequence)
        
        # Initialize: at index 0, find all valid starting TVT values
        gr_target_0 = self.gr_sequence[0]
        tvt_candidates_0 = self._inverse_set(gr_target_0)
        
        for tvt_val in tvt_candidates_0:
            bin_idx = self._tvt_to_bin_index(tvt_val)
            gr_predicted = self._forward_map(tvt_val)
            error = self._compute_error(gr_target_0, gr_predicted)
            
            state_key = (0, 0, 0, bin_idx)
            self.dp_states[state_key] = error
            self.tvt_at_state[state_key] = tvt_val
        
        # Forward pass: process each GR point
        for i in tqdm(range(n - 1)):
            gr_target = self.gr_sequence[i + 1]
            
            # Get all active states at index i
            active_states_at_i = [
                (j, s, v) for (idx, j, s, v) in self.dp_states.keys() if idx == i
            ]
            
            for j, s, v in active_states_at_i:
                current_state_key = (i, j, s, v)
                current_error = self.dp_states[current_state_key]
                current_tvt = self.tvt_at_state[current_state_key]
                
                # Try slight change
                slight_transitions = self._transition_slight_change(i, j, s, v, gr_target)
                for next_state_key, transition_error, next_tvt in slight_transitions:
                    total_error = current_error + transition_error
                    
                    # Update if this is the best path to next_state_key
                    if next_state_key not in self.dp_states or total_error < self.dp_states[next_state_key]:
                        self.dp_states[next_state_key] = total_error
                        self.parent_map[next_state_key] = current_state_key
                        self.tvt_at_state[next_state_key] = next_tvt
                
                # Try jump if we have jumps remaining and enough points elapsed
                if j < self.max_jumps and s >= self.max_points_without_jump:
                    jump_transitions = self._transition_jump(i, j, s, gr_target)
                    for next_state_key, transition_error, next_tvt in jump_transitions:
                        total_error = current_error + transition_error
                        
                        if next_state_key not in self.dp_states or total_error < self.dp_states[next_state_key]:
                            self.dp_states[next_state_key] = total_error
                            self.parent_map[next_state_key] = current_state_key
                            self.tvt_at_state[next_state_key] = next_tvt
        
        # Find the best final state
        final_states_at_n_minus_1 = [
            ((i, j, s, v), error) for (i, j, s, v), error in self.dp_states.items() if i == n - 1
        ]
        
        if not final_states_at_n_minus_1:
            raise ValueError("No valid path found in DP optimization.")
        
        best_final_state, min_error = min(final_states_at_n_minus_1, key=lambda x: x[1])
        
        # Reconstruct path
        tvt_sequence = self._reconstruct_path(best_final_state)
        
        return tvt_sequence, min_error
    
    def _reconstruct_path(self, final_state: Tuple[int, int, int, int]) -> List[float]:
        """
        Reconstruct the optimal path by backtracking from final state to start.
        
        Args:
            final_state: The final state (i, j, s, v).
        
        Returns:
            List of TVT values.
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
