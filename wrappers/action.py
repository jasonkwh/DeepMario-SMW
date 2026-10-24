import gymnasium as gym
import numpy as np


class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.combos = [
            [],  # do nothing
            ["LEFT"],  # move left
            ["RIGHT"],
            ["B"],  # jump
            ["A"],  # spin jump
            ["DOWN"],  # crouching
            ["RIGHT", "A"],  # right spin jump
            ["LEFT", "A"],  # left spin jump
            ["RIGHT", "B"],  # right jump
            ["LEFT", "B"],  # left jump
            ["Y"],  # pick up item
            ["Y", "DOWN"],  # pick up item + crouch
            ["Y", "B"],  # jump + pick up item
            ["RIGHT", "Y"],  # run right, fire right
            ["LEFT", "Y"],  # run left, fire left
            ["RIGHT", "Y", "B"],  # run right + jump + fire
            ["LEFT", "Y", "B"],  # run left + jump + fire
            ["RIGHT", "Y", "A"],  # run right + spin jump + fire
            ["LEFT", "Y", "A"],  # run left + spin jump + fire
            # ["UP"], # up
            # ["UP", "B"], # jumping out of water
            # ["UP", "A"], # spin jump out of water / Yoshi dismount
            # ["UP", "Y"],
        ]
        buttons = env.unwrapped.buttons
        self.actions = []

        for combo in self.combos:
            action = np.zeros(env.action_space.n, dtype=np.uint8)
            for button in combo:
                action[buttons.index(button)] = 1
            self.actions.append(action)
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action].copy()
