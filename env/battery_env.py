import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BatteryChargingEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # Observation:
        # [SOC, Temperature, Voltage]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 20.0, 2.5]),
            high=np.array([1.0, 60.0, 4.2]),
            dtype=np.float32
        )

        # Action:
        # Charging current (0.5C to 3C)
        self.action_space = spaces.Box(
            low=np.array([0.5]),
            high=np.array([3.0]),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.soc = 0.10
        self.temperature = 25.0
        self.voltage = 3.2

        state = np.array(
            [self.soc, self.temperature, self.voltage],
            dtype=np.float32
        )

        return state, {}

    def step(self, action):

        current = action[0]

        # Simple battery dynamics
        self.soc += current * 0.005
        self.temperature += current * 0.1
        self.voltage += current * 0.01

        # Reward
        reward = self.soc * 10

        # Penalties
        if self.temperature > 45:
            reward -= 50

        if self.voltage > 4.2:
            reward -= 50

        terminated = self.soc >= 0.80

        state = np.array(
            [self.soc, self.temperature, self.voltage],
            dtype=np.float32
        )

        return state, reward, terminated, False, {}

    def render(self):
        print(
            f"SOC={self.soc:.2f}, "
            f"Temp={self.temperature:.2f}, "
            f"Voltage={self.voltage:.2f}"
        )