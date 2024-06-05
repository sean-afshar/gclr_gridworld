from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

# TODO: GridState + refactor
@dataclass(frozen=True)
class _State:
    _idx: int
    _grid: GridWorld

    def __post_init__(self):
        self.row = self._idx // self.grid.w
        self.col = self._idx % self.grid.w

    @property
    def idx(self):
        return self._idx

    @property
    def idx_to_coords(self):
        return self.row, self.col


@dataclass
class GridWorld:
    _w: int 
    _h: int
    _obstacles: NDArray[np.int_] = field(default=None)

    def __post_init__(self) -> None:
        self.h = self._h
        self.w = self._w
        if self._obstacles is not None:
            self.obstacles = self._obstacles

    @property
    def w(self) -> int:
        return self._w
    
    @w.setter
    def w(self, new_w: int) -> None:
        if new_w <= 0:
            raise ValueError("Width must be greater than 0")
        self._w = new_w
    
    @property
    def h(self) -> int:
        return self._h
    
    @h.setter
    def h(self, new_h: int) -> None:
        if new_h <= 0:
            raise ValueError("Height must be greater than 0")
        self._h = new_h

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
        if not np.all(self._base_adjacency_matrix[new_obstacles[0], new_obstacles[1]]):
            raise ValueError("Obstacles must be between adjacent states")

        self._obstacles = new_obstacles
    
    @property
    def grid_indices(self) -> NDArray[np.int_]:
        return np.arange(self.n_states).reshape((self._h, self._w))
    
    @property
    def n_states(self) -> int:
        return self._w * self._h
    
    @property
    def _base_adjacency_matrix(self) -> NDArray[np.bool_]:
        # Matrix construction assuming non-toroidal grid
        row_idxs = np.arange(self.n_states) // self._w
        col_idxs = np.arange(self.n_states) % self._w
        vert_dists = np.abs(row_idxs[:, None] - row_idxs[None, :])
        hor_dists = np.abs(col_idxs[:, None] - col_idxs[None, :]) 
        adj_mat = hor_dists + vert_dists == 1 

        return adj_mat
    
    @property
    def adjacency_matrix(self) -> NDArray[np.bool_]:
        adj_mat = self._base_adjacency_matrix

        # Inserting obstacles
        if self._obstacles is not None:
            adj_mat[self._obstacles[0], self._obstacles[1]] = False
            adj_mat[self._obstacles[1], self._obstacles[0]] = False # Enforce symmetry
    
        return adj_mat