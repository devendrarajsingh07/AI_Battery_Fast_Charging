"""
Continuous battery charging environment backed by the Single Particle Model.

Observation space (4D):
  [SOC, Voltage (V), Temperature (°C), SOH]

Action space (1D):
  Requested charging current as a C-rate → [0.5, 3.0]
  (1C = 3 A for the 3 Ah cell, so 3.0 requests 9 A)

Charger behaviour (CC-CV):
  The requested current is applied as long as the terminal voltage stays at
  or below 4.20 V. Once the cell reaches 4.20 V the charger holds the voltage
  and tapers the current (constant-voltage phase), exactly as a real charger
  does. The agent can always ask for less current than the CV limit.

Episode ends when:
  - SOC ≥ 0.80              (target reached — success)
  - Temperature ≥ 45 °C     (thermal safety limit — failure)
  - SOH ≤ 0.80              (cell too degraded — failure)
  - 180 steps (3 h)         (time limit — truncated)
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

V_MAX = 4.20            # charge voltage limit (V)
T_MAX = 45.0            # thermal safety limit (°C)
T_COMFORT = 35.0        # heat penalty starts above this (°C)
SOC_START = 0.10
SOC_TARGET = 0.80
MAX_STEPS = 180         # 3 h of simulated time


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
        self.capacity_ah = SingleParticleModel.CAPACITY_AH

        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.soc = SOC_START
        self.soh = 1.0
        self.time_step = 0

        self.thermal.temperature = 25.0
        self.thermal.ambient_temp = 25.0
        self.temperature = 25.0

        self.spm.reset(soc_init=SOC_START)

        # voltage from SPM at rest (no current)
        self.voltage = self.spm.ocv()

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _get_obs(self):
        return np.array(
            [self.soc, self.voltage, self.temperature, self.soh],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def _soh_loss(self, c_rate: float) -> float:
        """
        Semi-empirical capacity fade for one step.

        Two contributions:
          1. Cycle stress: proportional to (C-rate)²  (high-current side reactions)
          2. Thermal stress: excess temperature above 35 °C
        """
        cycle_loss = (c_rate ** 2) * 8e-6
        thermal_loss = max(0.0, self.temperature - T_COMFORT) * 1.5e-5
        return cycle_loss + thermal_loss

    # ------------------------------------------------------------------
    def _calculate_reward(self, d_soc: float, soh_loss: float) -> float:
        """
        Per-step reward:
          +  charge added this step       (200 per unit SOC → 140 for 10 → 80 %)
          −  1 per minute spent charging  (rewards finishing sooner)
          −  heat above 35 °C
          −  capacity lost this step
          − 100 once if the 45 °C safety limit is reached
        """
        r = 200.0 * d_soc
        r -= 1.0
        r -= 1.0 * max(0.0, self.temperature - T_COMFORT)
        r -= 4000.0 * soh_loss
        if self.temperature >= T_MAX:
            r -= 100.0
        return float(r)

    # ------------------------------------------------------------------
    def step(self, action):
        requested_c = float(np.clip(action[0], 0.5, 3.0))
        self.time_step += 1

        # — charger: CC up to the requested rate, CV once at 4.20 V —
        i_request = requested_c * self.capacity_ah
        i_cv = self.spm.max_current_for_voltage(V_MAX, self.temperature, i_request)
        current = min(i_request, i_cv)                      # amps
        c_rate = current / self.capacity_ah
        mode = "CC" if current >= i_request - 1e-9 else "CV"

        # — electrochemistry (evaluated at the start-of-step temperature) —
        eta = self.spm.overpotential(current, self.temperature)
        heat_w = current ** 2 * self.spm.r_internal(self.temperature) + current * eta

        prev_soc = self.soc
        self.spm.update_soc(current)
        self.soc = self.spm.soc

        self.voltage = min(
            self.spm.terminal_voltage(current, self.temperature), V_MAX
        )

        # — heat and ageing —
        self.temperature = self.thermal.update(current, heat_w=heat_w)

        soh_loss = self._soh_loss(c_rate)
        self.soh = max(self.soh - soh_loss, 0.0)

        reward = self._calculate_reward(self.soc - prev_soc, soh_loss)

        terminated = (
            self.soc >= SOC_TARGET
            or self.temperature >= T_MAX
            or self.soh <= 0.80
        )
        truncated = (not terminated) and self.time_step >= MAX_STEPS

        info = {
            "soc": self.soc,
            "voltage": self.voltage,
            "temperature": self.temperature,
            "soh": self.soh,
            "current": c_rate,              # applied C-rate
            "current_a": current,           # applied current in amps
            "requested_c_rate": requested_c,
            "mode": mode,
            "step": self.time_step,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        print(
            f"[t={self.time_step:3d}] "
            f"SOC={self.soc:.3f}  "
            f"V={self.voltage:.3f}V  "
            f"T={self.temperature:.1f}°C  "
            f"SOH={self.soh:.4f}"
        )
