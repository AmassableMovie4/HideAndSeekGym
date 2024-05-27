import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class HideAndSeekEnv(gym.Env):
    def __init__(self):
        super(HideAndSeekEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(6)  # up, down, left, right, forward, backward
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(6,), dtype=np.float32)  # Positions of hider and seeker

        self.stage_size = 10  # Define the stage size
        self.increment = 0.1  # Define the fixed increment value

        # Initialize PyBullet
        if p.isConnected():
            p.disconnect()
        self.client = p.connect(p.DIRECT)  # Use DIRECT mode for testing without visualization
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.plane = p.loadURDF("plane.urdf")
        print("Plane loaded:", self.plane)

    def reset(self, seed=None, options=None):
        # Handle the seed parameter
        if seed is not None:
            np.random.seed(seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.plane = p.loadURDF("plane.urdf")
        print("Plane reset and loaded:", self.plane)

        # Randomize positions ensuring they are at least 1 meter apart
        self.hider_pos = self._random_position()
        self.seeker_pos = self._random_position()
        while np.linalg.norm(self.hider_pos - self.seeker_pos) < 1.0:
            self.seeker_pos = self._random_position()

        self.hider = p.loadURDF("r2d2.urdf", self.hider_pos)
        self.seeker = p.loadURDF("r2d2.urdf", self.seeker_pos)
        print("Hider loaded at:", self.hider_pos)
        print("Seeker loaded at:", self.seeker_pos)

        return self._get_observation(), {}

    def step(self, action):
        self._apply_action(action)

        p.stepSimulation()
        time.sleep(1./240.)  # Slow down the simulation for better visualization

        observation = self._get_observation()
        reward = self._compute_reward()
        terminated = bool(self._check_done())
        truncated = False  # This example does not use truncation, but you can add this logic if needed
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        if p.isConnected():
            p.disconnect()

    def _random_position(self):
        return np.random.uniform(low=0, high=self.stage_size, size=3)

    def _apply_action(self, action):
        if action == 0:  # Forward
            self.hider_pos[1] += self.increment
        elif action == 1:  # Backward
            self.hider_pos[1] -= self.increment
        elif action == 2:  # Left
            self.hider_pos[0] -= self.increment
        elif action == 3:  # Right
            self.hider_pos[0] += self.increment
        elif action == 4:  # Up
            self.hider_pos[2] += self.increment
        elif action == 5:  # Down
            self.hider_pos[2] -= self.increment

        # Ensure the agent stays within the stage boundaries
        self.hider_pos = np.clip(self.hider_pos, 0, self.stage_size)

        # Update the hider's position in the simulation
        p.resetBasePositionAndOrientation(self.hider, self.hider_pos.tolist(), [0, 0, 0, 1])

    def _get_observation(self):
        hider_pos, _ = p.getBasePositionAndOrientation(self.hider)
        seeker_pos, _ = p.getBasePositionAndOrientation(self.seeker)
        return np.concatenate([hider_pos, seeker_pos]).astype(np.float32)

    def _compute_reward(self):
        hider_pos, _ = p.getBasePositionAndOrientation(self.hider)
        seeker_pos, _ = p.getBasePositionAndOrientation(self.seeker)
        distance = np.linalg.norm(np.array(hider_pos) - np.array(seeker_pos))
        return -distance  # Negative reward for being close

    def _check_done(self):
        hider_pos, _ = p.getBasePositionAndOrientation(self.hider)
        seeker_pos, _ = p.getBasePositionAndOrientation(self.seeker)
        distance = np.linalg.norm(np.array(hider_pos) - np.array(seeker_pos))
        return distance < 1.0  # Done if distance is less than 1 meter
