# AI Battery Fast Charging Optimizer

Trains a **Soft Actor-Critic (SAC)** reinforcement learning agent to adaptively
control lithium-ion battery charging — faster charging, less heat, better longevity.

---
 live app at   devrajsingh07.streamlit.app 

## What's inside

```
AI_Battery_Fast_Charging/
├── agent/
│   ├── train_sac.py             # train the SAC agent from scratch
│   ├── test_sac.py              # run the basic env with a loaded model
│   └── test_continuous_sac.py  # run the continuous (SPM) env
│
├── battery_model/
│   ├── thermal_model.py         # I²R heating + convective cooling
│   └── spm_model.py             # Single Particle Model — OCV, R_int, η
│
├── env/
│   ├── battery_env.py           # simple 3-state env (SOC/Temp/Voltage)
│   ├── continuous_pybamm_env.py # 4-state env backed by SPM (used for training)
│   └── pybamm_env.py            # PyBaMM direct-call env (slow, for reference)
│
├── dashboard/
│   └── app.py                   # Streamlit dashboard
│
└── results/
    ├── models/
    │   └── continuous_sac_model.zip  # pre-trained weights
    └── plots/                        # static PNGs (run generate_plots.py)
```

---

## Training your own agent

```bash
python agent/train_sac.py
```

Default: 20 000 timesteps (~5 min on CPU). Model saved to `results/models/`.

To retrain from scratch just delete the existing `.zip` and run again.

---

## How the physics works

The charging environment uses a simplified **Single Particle Model**:

```
V_terminal = OCV(SOC) + i · R_int(T) + η(i, T)
```

| Term | Description |
|------|-------------|
| `OCV(SOC)` | Open-circuit voltage — degree-6 polynomial fit for NMC 622 |
| `R_int(T)` | Internal resistance, temperature-corrected (48 mΩ at 25 °C) |
| `η(i, T)` | Overpotential — linearised Butler-Volmer |

SOC is updated via Coulomb counting (dt = 60 s). SOH degrades from both
high-current stress and sustained elevated temperature.

---

## Reward design

The agent maximises:

```
R = 20·SOC
  − 600·max(V − 4.10, 0)    # heavy penalty near voltage cutoff
  − 8·max(T − 35, 0)         # temperature penalty above comfort zone
  − 120·(1 − SOH)            # aging penalty
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `stable-baselines3` | SAC implementation |
| `gymnasium` | RL environment API |
| `torch` | neural network backend |
| `pybamm` | electrochemical reference (pybamm_env) |
| `streamlit` | dashboard |
| `plotly` | interactive charts |
| `numpy / pandas / matplotlib` | data handling and static plots |

---

## Roadmap

- [ ] Full PyBaMM state coupling (solid diffusion, electrolyte)
- [ ] Aging dataset integration (NASA, Oxford battery datasets)
- [ ] Multi-cell pack simulation
- [ ] Digital Twin / hardware-in-the-loop export
