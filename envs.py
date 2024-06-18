from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from typing import Iterable, Optional

# TODO: GridState + refactor
@dataclass(frozen=True)
class _State:
    _idx: int
    _grid: GridWorld
    _reward: float

    def __post_init__(self):
        # Check if state within grid
        self._check_idx(self._idx)
        self.row = self._idx // self._grid.w
        self.col = self._idx % self._grid.w

    def _check_idx(self, x: int | _State) -> bool:
        if isinstance(x, _State):
            index = x.idx
        else:
            index = x
        if index < 0 or index >= self._grid.n_states:
            raise ValueError("State index must be within grid")
        return True

    @property
    def idx(self):
        return self._idx

    @property
    def coords(self):
        return self.row, self.col
    
    @property
    def reward(self):
        return self._reward
    
    @property 
    def adjacent_states(self) -> Iterable[_State]:
        return np.where(self._grid.adjacency_matrix[self.idx])[0]
    
    def is_next_to(self, x: int | _State) -> bool:
        # Check if x is a valid state
        self._check_idx(x)
        if isinstance(x, _State):
            return self._grid.adjacency_matrix[self.idx, x.idx]
        else:
            return self._grid.adjacency_matrix[self.idx, x]


    # # TODO: Implement the action space first!!!!
    # # TODO: Implement simple policies first     
    # def prob_transition_to(self, s: int | _State) -> np.float_:
    #     # Check if x is a valid state
    #     self._check_idx(x)
    #     if isinstance(x, _State):
    #         return self._grid.transition_probs[self.idx, x.idx]
    #     else:
    #         return self._grid.transition_probs[self.idx, x]
        
    # @property
    # def transition_probs(self) -> NDArray[np.float_]:
    #     return self._grid.transition_probs[self.idx]
    
    # # @probability

# [[up], [down], [left], [right]]  
ACTION_SPACE = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]) # (4 x 2)
    
@dataclass
class GridWorld:
    _w: int 
    _h: int
    _obstacles: Optional[NDArray[np.int_]] = field(default=None)
    _actions: Optional[NDArray[np.int_]] = field(default=ACTION_SPACE)
    _transition_matrix: Optional[NDArray[np.float_]] = field(default=None)
    _initialized: bool = False

    def __post_init__(self) -> None:
        # TODO: reward structure (maybe latent within the state structure), state space
        self.h = self._h
        self.w = self._w
        self.actions = self._actions
        if self._obstacles is not None:
            self.obstacles = self._obstacles
        if self._transition_matrix is not None:
            self.transition_matrix = self._transition_matrix
        else:
            self.transition_matrix = self._make_default_trans_matrix()
        self._initialized = True

    @property
    def w(self) -> int:
        return self._w
    
    @w.setter
    def w(self, new_w: int) -> None:
        if new_w <= 0:
            raise ValueError("Width must be greater than 0")
        self._w = new_w
        if self._initialized:
            self._obstacles = None # Reset obstacles if grid size changes
            print("Warning, change of grid size has removed obstacles")
    
    @property
    def h(self) -> int:
        return self._h
    
    @h.setter
    def h(self, new_h: int) -> None:
        if new_h <= 0:
            raise ValueError("Height must be greater than 0")
        self._h = new_h
        if self._initialized:
            self._obstacles = None # Reset obstacles if grid size changes
            print("Warning, change of grid size has removed obstacles")

    @property
    def actions(self) -> NDArray[np.int_]:
        return self._actions
    
    @property 
    def default_actions(self) -> NDArray[np.int_]:
        return ACTION_SPACE
    
    @actions.setter
    def actions(self, new_actions: NDArray[np.int_]) -> None:
        # Check shape 
        if np.logical_or(new_actions.shape[1] != 2, new_actions.ndim != 2):
            raise ValueError("Action space must be shape |A| x 2")
        # Check if actions fit within grid
        if np.any(np.logical_or(np.abs(new_actions)[:, 0] > self.w, np.abs(new_actions)[:, 1] > self.h)):
            raise ValueError("Actions are too large for grid size")
        # Check if actions are unique
        if np.unique(new_actions, axis=0).shape[0] != new_actions.shape[0]:
            raise ValueError("Actions must be unique")
        self._actions = new_actions

    @property
    def obstacles(self) -> NDArray[np.int_]:
        return self._obstacles
    
    @obstacles.setter
    def obstacles(self, new_obstacles: NDArray[np.int_]) -> None:
        # Check size
        if new_obstacles.ndim != 2:
            raise ValueError("Obstacles must be a 2D array")
        # Check shape
        if new_obstacles.shape[0] != 2:
            raise ValueError("Obstacles must have 2 rows")
        # Check if within grid
        if not np.all(np.logical_and(new_obstacles >= 0, new_obstacles < self.n_states)):
            raise ValueError("Obstacles must be within the grid")
        # Check if obstacles unique
        if np.unique(new_obstacles, axis=1).shape[1] != new_obstacles.shape[1]:
            raise ValueError("Obstacles must be unique")
        # Check for self obstacles
        if np.any(np.diff(new_obstacles, axis=0) == 0):
            raise ValueError("Obstacles must be between different states")
        # Check for non-neighbor obstacles
        if not np.all(self.base_adjacency_matrix[new_obstacles[0], new_obstacles[1]]):
            raise ValueError("Obstacles must be between adjacent states")

        self._obstacles = new_obstacles
    
    @property
    def grid_indices(self) -> NDArray[np.int_]:
        return np.arange(self.n_states).reshape((self.h, self.w))
    
    @property
    def n_states(self) -> int:
        return self.w * self.h
    
    def coords_to_idx(self, row: NDArray[np.int_] | np.int_, col: NDArray[np.int_] | np.int_) -> NDArray[np.int_] | np.int_:
        return self.w * row + col
    
    def idx_to_coords(self, s: NDArray[np.int_] | np.int_) -> Iterable[NDArray[np.int_] | np.int_]:
        return s // self.w, s % self.w
    
    @property 
    def row_idxs(self) -> NDArray[np.int_]:
        return self.idx_to_coords(np.arange(self.n_states))[0]
    
    @property 
    def col_idxs(self) -> NDArray[np.int_]:
        return self.idx_to_coords(np.arange(self.n_states))[1]
    
    @property
    def row_idx_mesh(self) -> NDArray[np.int_]:
        return self.row_idxs.reshape((self.h, self.w))
    
    @property
    def col_idx_mesh(self) -> NDArray[np.int_]:
        return self.col_idxs.reshape((self.h, self.w))
    
    @property
    def meshgrid(self) -> Iterable[NDArray[np.int_]]:
        return self.row_idx_mesh, self.col_idx_mesh
    
    def _make_base_adjacency_matrix(self) -> NDArray[np.bool_]:
        # Matrix construction assuming non-toroidal grid
        vert_dists = np.abs(self.row_idxs[:, None] - self.row_idxs[None, :])
        hor_dists = np.abs(self.col_idxs[:, None] - self.col_idxs[None, :]) 
        adj_mat = hor_dists + vert_dists == 1 

        return adj_mat
    
    @property
    def base_adjacency_matrix(self) -> NDArray[np.bool_]:
        return self._make_base_adjacency_matrix()
    
    @property
    def adjacency_matrix(self) -> NDArray[np.bool_]:
        adj_mat = self.base_adjacency_matrix

        # Inserting obstacles
        if self._obstacles is not None:
            adj_mat[self._obstacles[0], self._obstacles[1]] = False
            adj_mat[self._obstacles[1], self._obstacles[0]] = False # Enforce symmetry
    
        return adj_mat

    def _make_default_trans_matrix(self) -> NDArray[np.float_]:
        current_states = np.broadcast_to(np.stack(self.meshgrid, axis=0), (*self.actions.shape, self.h, self.w))
        successors = current_states + self.actions[..., None, None] # (|A|, 2, h, w)
        # Check if new states are within bounds 
        bounds_mask = (successors < 0) + (successors[:, [0]] >= self.h) + (successors[:, [1]] >= self.w) # (|A|, 2, h, w)
        # Shift over incorrect states
        successors[bounds_mask] = current_states[bounds_mask]
        # Convert from grid representation to state representation
        current_states = np.reshape(current_states[:, 0] * self.w  + current_states[:, 1], (len(self.actions), -1))
        successors = np.reshape(successors[:, 0] * self.w + successors[:, 1], (len(self.actions), -1))
        # Check for obstacles
        obstacles_mask = ~self.adjacency_matrix[current_states, successors]
        # Shift over incorrect states
        successors[obstacles_mask] = current_states[obstacles_mask]
        # Convert into 1-hot encoding
        p = np.zeros((self.n_states, len(self.actions), self.n_states))
        a, s = np.meshgrid(np.arange(p.shape[1]), np.arange(p.shape[2]), indexing='ij')
        p[successors, a, s] = 1
        return p

    @property
    def default_trans_matrix(self) -> NDArray[np.float_]:
        return self._make_default_trans_matrix
    
    @property 
    def transition_matrix(self) -> NDArray[np.float_]:
        return self._transition_matrix
    
    @transition_matrix.setter
    def transition_matrix(self, p: NDArray[np.float_]) -> None:        
        # Check for shape 
        if p.shape != (self.n_states, len(self.actions), self.n_states):
            raise ValueError("Matrix shape does not match grid shape or action space")
        # Check if valid proabability distribution 
        if np.any(np.logical_or(p < 0, p > 1)):
            raise ValueError("Probabilities must be between 0 and 1")
        if not np.all(np.isclose(p.sum(axis=0), 1.0)):
            raise ValueError("P(s'|a, s) is not normalized for a given action and prior state")
        self._transition_matrix = p