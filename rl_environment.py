import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SeismicAlertEnv(gym.Env):
    def __init__(self, model, X, y, threshold=5.5):
        super().__init__()
        self.model = model
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.threshold = threshold
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(X.shape[1],), dtype=np.float32
        )
        self.current = 0

    def reset(self, **kwargs):
        self.current = 0
        obs = self.X.iloc[0].values.astype(np.float32)
        return obs, {}

    def step(self, action):
        obs = self.X.iloc[self.current].values.astype(np.float32)
        pred = self.model.predict(obs.reshape(1,-1))[0]
        # Define reward logic here
        done = self.current >= len(self.X)-1
        self.current += 1
        next_obs = self.X.iloc[self.current].values.astype(np.float32) if not done else np.zeros_like(obs)
        return next_obs, 0, done, False, {}
