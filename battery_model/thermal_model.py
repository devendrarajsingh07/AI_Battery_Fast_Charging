import numpy as np


class ThermalModel:

    def __init__(self):

        self.ambient_temp = 25.0
        self.temperature = 25.0

    def update(self, current):

        heat_generated = current ** 2 * 0.2

        cooling = 0.05 * (
            self.temperature - self.ambient_temp
        )

        self.temperature += (
            heat_generated - cooling
        )

        return self.temperature