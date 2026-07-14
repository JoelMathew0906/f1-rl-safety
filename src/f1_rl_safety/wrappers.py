import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple

from .f1_env import F1RaceEnv, RaceRegime


class DiscreteF1ActionWrapper(gym.ActionWrapper):
    """Wrap F1RaceEnv with a discrete action space.

    Each discrete action index maps to a tuple (pit_decision, tyre_choice, risk_level)
    in the underlying continuous action space. This is shared by value-based agents
    such as DQN, SARSA and REINFORCE.
    """

    def __init__(
        self,
        regime: RaceRegime = RaceRegime.UNCONSTRAINED,
        n_laps: int = 52,
        seed: int | None = None,
    ):
        base_env = F1RaceEnv(regime=regime, n_laps=n_laps, seed=seed)
        super().__init__(base_env)

        # Define a compact discrete action set.
        # Pit: 0 = no pit, 1 = pit
        # Tyre: 0 = SOFT, 1 = MEDIUM, 2 = HARD (dry-focused baseline)
        # Risk bins: [-1.0, -0.5, 0.0, 0.5, 1.0]
        self._risk_bins: List[float] = [-1.0, -0.5, 0.0, 0.5, 1.0]
        self._tyre_indices: List[int] = [0, 1, 2]
        self._pit_options: List[int] = [0, 1]

        self._action_map: List[Tuple[int, int, float]] = []
        for pit in self._pit_options:
            for tyre in self._tyre_indices:
                for risk in self._risk_bins:
                    self._action_map.append((pit, tyre, risk))

        # Expose discrete action space while keeping the original observation space.
        self.action_space = spaces.Discrete(len(self._action_map))

    def action(self, act: int):
        """Transform a discrete action index into the underlying Box action.

        Gymnasium's ActionWrapper calls this method automatically before
        passing the action to the wrapped environment.
        """
        pit_decision, tyre_choice, risk_level = self._action_map[int(act)]

        box_action = np.array(
            [float(pit_decision), float(tyre_choice), float(risk_level)],
            dtype=np.float32,
        )
        return box_action
