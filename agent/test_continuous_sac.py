import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from stable_baselines3 import SAC
from env.continuous_pybamm_env import ContinuousBatteryEnv

env = ContinuousBatteryEnv()

model = SAC.load(
    "results/models/continuous_sac_model"
)

state, _ = env.reset()

done = False

while not done:

    action, _ = model.predict(
        state,
        deterministic=True
    )

    state, reward, terminated, truncated, info = env.step(
        action
    )
    done = terminated or truncated

    print(
        f"SOC={state[0]:.2f}, "
        f"Voltage={state[1]:.2f}, "
        f"Temp={state[2]:.2f}, "
        f"C-rate={info['current']:.2f} ({info['mode']}), "
        f"Reward={reward:.2f}"
    )

print("\nAI Charging Complete")