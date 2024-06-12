from __future__ import annotations
from dataclasses import dataclass
from envs import GridWorld, _State
import numpy as np
from numpy.typing import NDArray
from typing import List

@dataclass
class Agent:
    grid: GridWorld
    state: int | _State
    policy: dict[_State | int, NDArray[np.float_]] # TODO: Implement default policy
    _initialized: bool = False

    def _check_state(self, state: _State | int) -> bool:
        if isinstance(state, _State):
            state = state.idx
        if state < 0 or state >= self.grid.n_states:
            raise ValueError("State index must be within grid")
        return True
    
    def _check_policy(self, policy: dict[_State | int, NDArray[np.float_]]) -> bool:
        # Check if policy is defined for all states 
        if len(policy) != self.grid.n_states:
            raise ValueError("Policy must have a probability for each state")
        # Check if action probabilities are between 0 and 1
        if np.any(np.logical_or(self.policy_matrix < 0, self.policy_matrix > 1)):
            raise ValueError("Probabilities must be between 0 and 1")
        # Check if action probabilities sum to 1 per existing state
        if not np.all(np.isclose(self.policy_matrix.sum(axis=1), 1)):
            raise ValueError("Policy probabilities must sum to 1 for each state")
        return True

    #TODO: Implement post init checks
    def __post_init__(self) -> None:
        self._check_state(self.state)
        self._check_policy(self.policy)
        self._initialized = True

    @property
    def state(self) -> int | _State:
        return self._state
    
    @state.setter
    def state(self, new_state: int | _State) -> None:
        if self._check_state(new_state):
            self._state = new_state

    @property
    def policy(self) -> dict[_State | int, NDArray[np.float_]]:
        return self._policy
    
    @policy.setter 
    def policy(self, new_policy: dict[_State | int, NDArray[np.float_]]) -> None:
        if self._check_policy(new_policy):
            self._policy = new_policy

    def change_state_policy(self, new_state_policy: NDArray[np.float_]) -> None:
        # Check if probabilities sum to 1
        if not np.isclose(new_state_policy.sum(), 1):
            raise ValueError("Policy probabilities must sum to 1")
        self.policy[self.state] = new_state_policy
        
    def change_state_action_prob(self, state: _State | int, action: int, p: np.float_) -> None:
        new_policy = self.policy.copy()
        new_policy[state][action] = p
        if self._check_policy(new_policy):
            self.policy = new_policy

    # TODO: Implement this once we implement transition matrix
    def n_step_transition_prob(self, target: int | _State, n: int) -> np.float_:
        if self._check_state(target):
            pass

    def prob_transition_to(self, next_state: int | _State) -> np.float_:
        return self.n_step_transition_prob(next_state, n = 1)
    
    @property
    def policy_matrix(self) -> NDArray[np.float_]:
        return np.array([x for x in self.policy.values()])
    
    @property
    def next_action_probs(self) -> NDArray[np.float_]:
        return self.policy[self.state]
    
    # TODO: update this function once state space is implemented in GridWorld
    @property
    def next_states(self) -> NDArray[np.int_]:
        if isinstance(self.state, _State):
            return np.where(self.grid.adjacency_matrix[self.state.idx])[0]
        else:
            return np.where(self.grid.adjacency_matrix[self.state])[0]