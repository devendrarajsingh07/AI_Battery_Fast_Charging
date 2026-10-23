"""
Lumped (single-node) thermal model of a cylindrical cell.

    m·c_p · dT/dt = Q_gen − h·A · (T − T_amb)

Q_gen is the heat generated inside the cell in watts. The environment
passes the ohmic + overpotential heat from the SPM; if it is omitted the
model falls back to I²R heating with the 25 °C internal resistance.

Parameters are typical for an 18650-format cell (~45 g) under light forced-air
cooling. They are representative values, not fitted to a specific cell.
"""

THERMAL_MASS = 45.0      # J/K   — m·c_p  (≈ 45 g × 1000 J/(kg·K))
H_A = 0.20               # W/K   — convective conductance to ambient
R_DEFAULT = 0.048        # Ω     — used only when heat_w is not supplied
DT_SECONDS = 60.0        # one environment step
SUBSTEPS = 6             # explicit-Euler sub-steps per environment step


class ThermalModel:

    def __init__(self):
        self.ambient_temp = 25.0
        self.temperature = 25.0

    def update(self, current, heat_w=None):
        """
        Advance the cell temperature by one 60 s step.

        current : charging current in A
        heat_w  : heat generated in W (optional; defaults to I²·R_DEFAULT)
        """
        q_gen = current ** 2 * R_DEFAULT if heat_w is None else heat_w
        dt = DT_SECONDS / SUBSTEPS
        for _ in range(SUBSTEPS):
            q_loss = H_A * (self.temperature - self.ambient_temp)
            self.temperature += (q_gen - q_loss) * dt / THERMAL_MASS
        return self.temperature
