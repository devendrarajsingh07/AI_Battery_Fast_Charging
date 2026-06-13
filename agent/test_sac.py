import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from stable_baselines3 import SAC
from env.battery_env import BatteryChargingEnv

env = BatteryChargingEnv()

model = SAC.load("agent/sac_battery_model")

state, _ = env.reset()

done = False

while not done:

    action, _ = model.predict(
        state,
        deterministic=True
    )

    state, reward, done, _, _ = env.step(action)

    print(
        f"SOC={state[0]:.2f}, "
        f"Temp={state[1]:.2f}, "
        f"Voltage={state[2]:.2f}, "
        f"Action={action[0]:.2f}"
    )

print("\nAI Charging Complete")