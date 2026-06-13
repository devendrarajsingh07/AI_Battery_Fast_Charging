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

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256
)

model.learn(
    total_timesteps=20000
)

model.save(
    "results/models/continuous_sac_model"
)

print("Training Complete")