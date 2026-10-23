"""
Compare the trained SAC agent against constant-current charging.

Runs one episode per policy in ContinuousBatteryEnv, prints a summary
table and saves results/plots/rl_vs_constant.png.

Usage (from project root):
    python results/compare_policies.py
"""

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from stable_baselines3 import SAC
from env.continuous_pybamm_env import ContinuousBatteryEnv

mpl.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#0d1117",
    "axes.edgecolor":    "#21262d",
    "axes.labelcolor":   "#8b949e",
    "axes.titlecolor":   "#c9d1d9",
    "axes.grid":         True,
    "grid.color":        "#1c2128",
    "grid.linewidth":    0.8,
    "xtick.color":       "#6e7681",
    "ytick.color":       "#6e7681",
    "text.color":        "#c9d1d9",
    "lines.linewidth":   2.0,
    "savefig.facecolor": "#0d1117",
    "savefig.dpi":       150,
    "font.family":       "DejaVu Sans",
})

MODEL_PATH = "results/models/continuous_sac_model"
CONSTANT_RATES = [1.5, 2.0, 2.5]
COLORS = {"SAC agent": "#39d353", 1.5: "#58a6ff", 2.0: "#bc8cff", 2.5: "#ff7b72"}


def run_episode(policy):
    env = ContinuousBatteryEnv()
    state, _ = env.reset()
    hist = {"soc": [], "temperature": [], "voltage": [], "current": [], "soh": []}
    done = False
    while not done:
        state, _, done, _, info = env.step(policy(state))
        for k in hist:
            hist[k].append(info[k])
    if info["soc"] >= 0.80:
        hist["end"] = "SOC target"
    elif info["temperature"] >= 45.0:
        hist["end"] = "45 °C limit"
    elif info["voltage"] >= 4.20:
        hist["end"] = "4.20 V cutoff"
    else:
        hist["end"] = "SOH limit"
    return hist


model = SAC.load(MODEL_PATH, device="cpu")
runs = {"SAC agent": run_episode(lambda s: model.predict(s, deterministic=True)[0])}
for rate in CONSTANT_RATES:
    runs[rate] = run_episode(lambda s, r=rate: np.array([r], dtype=np.float32))

# ── summary table ───────────────────────────────────────────────────
print(f"\n{'Policy':<16}{'Minutes':>8}{'Final SOC':>11}{'Peak T':>9}{'Final SOH':>11}  Ended by")
for name, h in runs.items():
    label = name if isinstance(name, str) else f"Constant {name}"
    print(f"{label:<16}{len(h['soc']):>8}{h['soc'][-1]:>10.1%}"
          f"{max(h['temperature']):>8.1f}°{h['soh'][-1]:>11.4f}  {h['end']}")

# ── chart ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for name, h in runs.items():
    label = name if isinstance(name, str) else f"Constant {name}"
    width = 2.6 if isinstance(name, str) else 1.6
    t = range(1, len(h["soc"]) + 1)
    ax1.plot(t, np.array(h["soc"]) * 100, color=COLORS[name], lw=width, label=label)
    ax2.plot(t, h["temperature"], color=COLORS[name], lw=width, label=label)

ax2.axhline(45, color="#f85149", linestyle="--", linewidth=1.2, label="Safety limit 45 °C")
ax1.set_title("State of charge")
ax1.set_ylabel("SOC (%)")
ax2.set_title("Cell temperature")
ax2.set_ylabel("Temperature (°C)")
for ax in (ax1, ax2):
    ax.set_xlabel("Time (min)")
    ax.legend(facecolor="#0d1117", edgecolor="#21262d", labelcolor="#c9d1d9", fontsize=9)
fig.tight_layout()

out = os.path.join(os.path.dirname(__file__), "plots", "rl_vs_constant.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out)
print(f"\nSaved → {os.path.relpath(out)}")
