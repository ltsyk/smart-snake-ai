import gym
import numpy as np
from gym import spaces
from .game import SnakeGame, GRID_SIZE, DIR_VECTORS

class SnakeEnv(gym.Env):
    """Gym wrapper for the SnakeGame."""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        # Observation: flattened grid (0 empty, 1 snake, 2 food)
        self.observation_space = spaces.Box(low=0, high=2,
                                            shape=(GRID_SIZE * GRID_SIZE,),
                                            dtype=np.int8)
        # Actions: 0=UP,1=DOWN,2=LEFT,3=RIGHT
        self.action_space = spaces.Discrete(4)
        self.game = None

    def reset(self):
        self.game = SnakeGame()
        self.game.reset()
        return self.game.get_state()

    def step(self, action):
        # Apply action
        self.game.step(action)
        state = self.game.get_state()
        reward = 1 if self.game.score > 0 else 0
        done = self.game.done
        info = {}
        return state, reward, done, info

    def render(self, mode='human'):
        if self.game:
            self.game.render()

    def close(self):
        pass