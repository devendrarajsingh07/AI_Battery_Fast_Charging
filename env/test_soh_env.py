from continuous_pybamm_env import ContinuousBatteryEnv

env = ContinuousBatteryEnv()

state, _ = env.reset()

done = False

while not done:

    action = env.action_space.sample()

    state, reward, done, _, _ = env.step(action)

    print(
        f"SOC={state[0]:.2f}, "
        f"Voltage={state[1]:.2f}, "
        f"Temp={state[2]:.2f}, "
        f"SOH={state[3]:.5f}, "
        f"Reward={reward:.2f}"
    )

print("Simulation Complete")