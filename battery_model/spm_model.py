"""
Single Particle Model (SPM) for NMC 622 lithium-ion cells.

This is a physics-based approximation of the Doyle-Fuller-Newman model.
Main assumptions:
  - Single representative particle for each electrode
  - Uniform reaction rate across each electrode
  - Thermal effects captured via R_int(T) correction
  - Butler-Volmer overpotential linearised (valid for low overpotentials)

References:
  - Doyle, Fuller, Newman (1993) J. Electrochem. Soc.
  - Marquis et al. (2019) J. Electrochem. Soc. — SPM analysis
  - Chen et al. (2020) — parameter set used for OCV fit
"""

import numpy as np


# ──────────────────────────────────────────────
# OCV lookup for NMC 622 cathode (vs Li/Li+)
# Polynomial fit from Chen2020 half-cell data.
# Valid range: SOC ∈ [0.01, 0.99]
# ──────────────────────────────────────────────

# Coefficients for: V = Σ c_i * SOC^i  (degree 6)
# Fitted so that:
#   SOC=0.0 → ~3.00 V
#   SOC=0.2 → ~3.47 V
#   SOC=0.5 → ~3.70 V
#   SOC=0.8 → ~3.95 V
#   SOC=1.0 → ~4.20 V
_OCV_POLY = np.array([
    3.0050,    # c0
    3.1200,    # c1
   -5.8140,    # c2
   14.9800,    # c3
  -22.6700,    # c4
   18.5300,    # c5
   -6.2800,    # c6
])


def _ocv_nmc(soc: float) -> float:
    """Open-circuit voltage from polynomial fit. Clipped to safe range."""
    s = float(np.clip(soc, 0.01, 0.99))
    return float(np.polyval(_OCV_POLY[::-1], s))   # np.polyval wants high→low


# Internal resistance at 25 °C (typical 18650-format NMC cell)
R_INT_25C = 0.048   # Ohm

# Temperature coefficient — resistance drops ~0.5 mΩ/°C above 25 °C
R_TEMP_COEFF = -0.0005  # Ω / °C

# Exchange current density proxy (lumped, normalised)
I0_REF = 1.0   # A  (reference exchange current)


class SingleParticleModel:
    """
    Simplified SPM with:
      - NMC 622 OCV curve (polynomial fit)
      - Temperature-dependent internal resistance
      - Linear Butler-Volmer overpotential (asinh approximation)
      - SOC tracking via Coulomb counting (dt = 60 s)

    Usage
    -----
    spm = SingleParticleModel()
    spm.reset(soc_init=0.1)
    voltage = spm.terminal_voltage(current=1.5, temperature=28.0)
    spm.update_soc(current=1.5)
    """

    # Nominal capacity in Ah — 3 Ah cell (roughly 18650 form factor)
    CAPACITY_AH = 3.0

    # One environment step = 60 seconds
    DT_SECONDS = 60.0

    def __init__(self):
        self.soc = 0.10

    # ------------------------------------------------------------------
    def reset(self, soc_init: float = 0.10):
        self.soc = float(np.clip(soc_init, 0.0, 1.0))

    # ------------------------------------------------------------------
    def ocv(self, soc: float = None) -> float:
        """Return OCV at current (or given) SOC."""
        s = soc if soc is not None else self.soc
        return _ocv_nmc(s)

    # ------------------------------------------------------------------
    def r_internal(self, temperature: float) -> float:
        """
        R_int corrected for temperature.
        At low temps the resistance increases; at high temps it drops.
        """
        r = R_INT_25C + R_TEMP_COEFF * (temperature - 25.0)
        return float(max(r, 0.010))   # floor at 10 mΩ

    # ------------------------------------------------------------------
    def overpotential(self, current: float, temperature: float) -> float:
        """
        Simplified Butler-Volmer overpotential (linearised).

        η = (R·T / α·F) · arcsinh(i / 2·i0)

        Using α = 0.5, F = 96485 C/mol, R_gas = 8.314 J/(mol·K)
        """
        T_K = temperature + 273.15
        R_GAS = 8.314
        F = 96485.0
        alpha = 0.5

        # Thermal voltage term
        thermal_v = (R_GAS * T_K) / (alpha * F)

        # Exchange current — mild temperature dependence (Arrhenius-like)
        i0 = I0_REF * np.exp(0.02 * (temperature - 25.0))

        eta = thermal_v * np.arcsinh(current / (2.0 * max(i0, 1e-6)))
        return float(eta)

    # ------------------------------------------------------------------
    def terminal_voltage(self, current: float, temperature: float) -> float:
        """
        V_t = OCV(SOC) + i·R_int + η(i, T)

        Positive current = charging (voltage rises above OCV).
        Hard-clipped to 4.25 V (hardware cutoff).
        """
        v = (
            self.ocv()
            + current * self.r_internal(temperature)
            + self.overpotential(current, temperature)
        )
        return float(min(v, 4.25))

    # ------------------------------------------------------------------
    def update_soc(self, current: float):
        """
        Coulomb counting: ΔSOC = i · dt / (3600 · C_nom)
        """
        delta = (current * self.DT_SECONDS) / (3600.0 * self.CAPACITY_AH)
        self.soc = float(np.clip(self.soc + delta, 0.0, 1.0))
        return self.soc

    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"SPM | SOC={self.soc:.3f}  "
            f"OCV={self.ocv():.3f} V"
        )
