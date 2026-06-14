import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybamm
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from battery_model.thermal_model import ThermalModel


class PyBaMMBatteryEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(
            low=np.array([0.5], dtype=np.float32),
            high=np.array([3.0], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([0.0, 2.5, 20.0]),
            high=np.array([1.0, 4.2, 60.0]),
            dtype=np.float32
        )

        self.thermal = ThermalModel()

        self.reset()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.soc = 0.10
        self.voltage = 3.20
        self.temperature = 25.0

        state = np.array(
            [
                self.soc,
                self.voltage,
                self.temperature
            ],
            dtype=np.float32
        )

        return state, {}

    def step(self, action):

        current = float(action[0])

        # -----------------------------
        # PyBaMM Simulation
        # -----------------------------

        model = pybamm.lithium_ion.SPM()

        params = pybamm.ParameterValues("Chen2020")

        sim = pybamm.Simulation(
            model,
            parameter_values=params
        )

        sim.solve([0, 60])

        voltage_data = sim.solution[
            "Terminal voltage [V]"
        ].entries

        self.voltage = float(
            voltage_data[-1]
        )

        # -----------------------------
        # Thermal Model
        # -----------------------------

        self.temperature = (
            self.thermal.update(current)
        )

        # -----------------------------
        # SOC Update
        # -----------------------------

        self.soc += current * 0.01

        self.soc = min(
            self.soc,
            1.0
        )

        # -----------------------------
        # Reward Function
        # -----------------------------

        soc_reward = self.soc * 10

        temp_penalty = 0

        if self.temperature > 35:

            temp_penalty = (
                self.temperature - 35
            ) * 10

        voltage_penalty = 0

        if self.voltage > 4.1:

            voltage_penalty = (
                self.voltage - 4.1
            ) * 500

        reward = (
            soc_reward
            - temp_penalty
            - voltage_penalty
        )

        # -----------------------------
        # Safety Termination
        # -----------------------------

        terminated = (
            self.soc >= 0.80
            or self.temperature >= 45
            or self.voltage >= 4.20
        )

        # -----------------------------
        # New State
        # -----------------------------

        state = np.array(
            [
                self.soc,
                self.voltage,
                self.temperature
            ],
            dtype=np.float32
        )

        info = {
            "soc": self.soc,
            "voltage": self.voltage,
            "temperature": self.temperature,
            "current": current
        }

        return (
            state,
            reward,
            terminated,
            False,
            info
        )

    def render(self):

        print(
            f"SOC={self.soc:.2f}, "
            f"Voltage={self.voltage:.2f}, "
            f"Temp={self.temperature:.2f}"
        )