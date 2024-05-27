from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from agentsim import HideAndSeekEnv


# Check if the custom environment follows the gym API
check_env(HideAndSeekEnv())

# Create and wrap the environment
env = HideAndSeekEnv()

# Create the PPO model
model = PPO("MlpPolicy", env, n_steps=1024, n_epochs=4, batch_size=32, verbose=2)

# Train the agent
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_hide_and_seek_3d")

# Load the trained model
model = PPO.load("ppo_hide_and_seek_3d")

# Test the trained agent
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)  # Extract the observation from the reset tuple
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()

env.close()