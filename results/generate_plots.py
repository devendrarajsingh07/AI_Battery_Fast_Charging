"""
Generate static result plots from one charging episode.

Uses a fixed 1.7 C-rate action (no RL agent required) so anyone
can regenerate the plots without needing the trained weights.

Usage (from project root):
    python results/generate_plots.py
"""

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from env.continuous_pybamm_env import ContinuousBatteryEnv

# ── dark style to match dashboard feel ──────────────────────────────
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

COLOR = {
    "soc":   "#39d353",
    "volt":  "#58a6ff",
    "temp":  "#ffa657",
    "soh":   "#bc8cff",
    "curr":  "#ff7b72",
}


# ── run episode ──────────────────────────────────────────────────────
env = ContinuousBatteryEnv()
state, _ = env.reset()

soc_history   = []
voltage_history = []
temp_history  = []
soh_history   = []
current_history = []

done = False
action = np.array([1.7], dtype=np.float32)   # 1.7 C — decent fast-charge rate

while not done:
    state, reward, done, _, info = env.step(action)
    soc_history.append(info["soc"])
    voltage_history.append(info["voltage"])
    temp_history.append(info["temperature"])
    soh_history.append(info["soh"])
    current_history.append(info["current"])

steps = list(range(len(soc_history)))

# ── helpers ─────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(out_dir, exist_ok=True)


def savefig(name, y_data, ylabel, color, hline=None, hline_label=""):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, y_data, color=color)
    ax.fill_between(steps, y_data, alpha=0.12, color=color)
    if hline is not None:
        ax.axhline(hline, color="#f85149", linestyle="--", linewidth=1.2,
                   label=hline_label)
        ax.legend(facecolor="#0d1117", edgecolor="#21262d",
                  labelcolor="#c9d1d9", fontsize=9)
    ax.set_xlabel("Time step (× 60 s)")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " vs Time")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name))
    plt.close(fig)
    print(f"  saved → plots/{name}")


# ── plot each metric ─────────────────────────────────────────────────
print(f"\nEpisode length: {len(steps)} steps ({len(steps) * 60 // 60} min simulated)\n")

savefig("soc_vs_time.png",         soc_history,     "State of Charge",     COLOR["soc"],
        hline=0.80, hline_label="Target 80 %")
savefig("voltage_vs_time.png",     voltage_history, "Terminal Voltage (V)", COLOR["volt"],
        hline=4.20, hline_label="CV cutoff 4.20 V")
savefig("temperature_vs_time.png", temp_history,    "Temperature (°C)",    COLOR["temp"],
        hline=45, hline_label="Safety limit 45 °C")
savefig("soh_vs_time.png",         soh_history,     "State of Health",     COLOR["soh"])
savefig("current_vs_time.png",     current_history, "Charging Current (C)", COLOR["curr"])

print("\nAll plots saved to results/plots/")