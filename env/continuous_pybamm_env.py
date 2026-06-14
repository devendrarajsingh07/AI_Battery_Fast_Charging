"""
Continuous battery charging environment backed by the Single Particle Model.

Observation space (4D):
  [SOC, Voltage (V), Temperature (°C), SOH]

Action space (1D):
  Charging current in C-rate units  → [0.5, 3.0]

Episode ends when:
  - SOC ≥ 0.80  (target reached)
  - Temperature ≥ 45 °C  (thermal runaway risk)
  - Voltage ≥ 4.20 V  (hardware cutoff)
  - SOH drops below 0.80  (cell too degraded)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from battery_model.thermal_model import ThermalModel
from battery_model.spm_model import SingleParticleModel


class ContinuousBatteryEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(
            low=np.array([0.5], dtype=np.float32),
            high=np.array([3.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array([0.0, 2.5, 20.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 4.2, 60.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.thermal = ThermalModel()
        self.spm = SingleParticleModel()

        # SOH degradation accumulators
        self._total_ah = 0.0          # lifetime Ah throughput
        self._heat_integral = 0.0     # Σ max(T-35, 0) dt

        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.soc = 0.10
        self.soh = 1.0
        self.time_step = 0

        self.thermal.temperature = 25.0
        self.thermal.ambient_temp = 25.0
        self.temperature = 25.0

        self.spm.reset(soc_init=0.10)

        # voltage from SPM at rest (no current)
        self.voltage = self.spm.ocv()

        self._total_ah = 0.0
        self._heat_integral = 0.0

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _get_obs(self):
        return np.array(
            [self.soc, self.voltage, self.temperature, self.soh],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def _update_soh(self, current: float):
        """
        Semi-empirical capacity fade model.

        Two contributions:
          1. Cycle-stress: proportional to i²  (high-current side reactions)
          2. Calendar / thermal stress: Arrhenius-like excess-heat term

        Calibrated so a 20 000-step run at 2C loses roughly 1-2 % SOH.
        """
        cycle_loss = (current ** 2) * 8e-6
        thermal_excess = max(0.0, self.temperature - 35.0)
        thermal_loss = thermal_excess * 1.5e-5

        self.soh -= cycle_loss + thermal_loss
        self.soh = max(self.soh, 0.0)

    # ------------------------------------------------------------------
    def _calculate_reward(self, current: float) -> float:
        """
        Reward shaping:
          +  fast SOC progress
          -  voltage overshoot (heavy penalty near 4.2 V)
          -  temperature above safe zone (35 °C)
          -  SOH degradation
          -  high current when already warm (discourages reckless charging)
        """
        r = self.soc * 20.0

        # voltage guard
        if self.voltage > 4.10:
            r -= (self.voltage - 4.10) * 600.0

        # temperature guard
        if self.temperature > 35.0:
            r -= (self.temperature - 35.0) * 8.0

        # aging
        r -= (1.0 - self.soh) * 120.0

        # discourage high current when hot
        if self.temperature > 32.0 and current > 2.0:
            r -= (current - 2.0) * (self.temperature - 32.0) * 2.0

        return float(r)

    # ------------------------------------------------------------------
    def step(self, action):
        current = float(np.clip(action[0], 0.5, 3.0))
        self.time_step += 1

        # — physics update —
        self.spm.update_soc(current)
        self.soc = self.spm.soc

        self.temperature = self.thermal.update(current)

        # voltage via SPM (OCV + ohmic drop + overpotential)
        self.voltage = self.spm.terminal_voltage(current, self.temperature)

        self._update_soh(current)

        reward = self._calculate_reward(current)

        terminated = (
            self.soc >= 0.80
            or self.temperature >= 45.0
            or self.voltage >= 4.20
            or self.soh <= 0.80
        )

        info = {
            "soc": self.soc,
            "voltage": self.voltage,
            "temperature": self.temperature,
            "soh": self.soh,
            "current": current,
            "step": self.time_step,
        }

        return self._get_obs(), reward, terminated, False, info

    # ------------------------------------------------------------------
    def render(self):
        print(
            f"[t={self.time_step:3d}] "
            f"SOC={self.soc:.3f}  "
            f"V={self.voltage:.3f}V  "
            f"T={self.temperature:.1f}°C  "
            f"SOH={self.soh:.4f}"
        )