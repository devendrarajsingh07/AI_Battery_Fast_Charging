from battery_env import BatteryChargingEnv

env = BatteryChargingEnv()

state, _ = env.reset()

done = False

while not done:

    action = env.action_space.sample()

    state, reward, done, _, _ = env.step(action)

    env.render()

print("Episode Finished")