from typing import Any, Dict, Optional

import numpy as np

from envs.wrapped_environments.base import BaseEnvironment, ResetReturn, StepReturn


class PureCoordinationGame(BaseEnvironment):
    """Single-step cooperative matrix game with a constant observation."""

    def __init__(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)
        self.agents = ["agent_0", "agent_1"]
        self.num_agents = len(self.agents)

        scenario = (scenario or "").lower()
        if scenario in ("coord", "coordination", "matrix2x2"):
            matrix = np.array([[10.0, 9.0], [8.0, 0.0]], dtype=np.float32)
        else:
            matrix = np.array(
                [
                    [5.0, 0.0, -50.0],
                    [0.0, 0.0, -50.0],
                    [-50.0, -50.0, 100.0],
                ],
                dtype=np.float32,
            )

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("reward matrix must be square")
        self.reward_matrix = matrix
        self.num_actions = int(self.reward_matrix.shape[0])
        self._obs = np.zeros((1,), dtype=np.float32)
        self._state = np.zeros((1,), dtype=np.float32)
        self._done = True

    def reset(self) -> ResetReturn:
        self._done = False
        observations = {agent: self._obs.copy() for agent in self.agents}
        legals = {agent: np.ones((self.num_actions,), dtype=np.float32) for agent in self.agents}
        info = {"state": self._state.copy(), "legals": legals}
        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        if self._done:
            observations, info = self.reset()
            terminals = {agent: np.array(True) for agent in self.agents}
            truncations = {agent: np.array(False) for agent in self.agents}
            rewards = {agent: np.array(0.0, dtype=np.float32) for agent in self.agents}
            return observations, rewards, terminals, truncations, info  # type: ignore

        a0 = int(np.asarray(actions[self.agents[0]]).item())
        a1 = int(np.asarray(actions[self.agents[1]]).item())

        reward = np.array(self.reward_matrix[a0, a1], dtype=np.float32)

        observations = {agent: self._obs.copy() for agent in self.agents}
        rewards = {agent: reward for agent in self.agents}
        terminals = {agent: np.array(True) for agent in self.agents}
        truncations = {agent: np.array(False) for agent in self.agents}
        legals = {agent: np.ones((self.num_actions,), dtype=np.float32) for agent in self.agents}
        info = {"state": self._state.copy(), "legals": legals}

        self._done = True
        return observations, rewards, terminals, truncations, info  # type: ignore

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        raise AttributeError(name)
