import gym
import cv2
import torch
import numpy as np
from collections import deque

W, H = (84, 84)  # Crop two rows, then downsample by 2x (fast, clean image).


class ProcgenEnv:
    def __init__(self, args):
        """
        Procgen environment. Currently only supports playing an infinite
        number of games.
        """
        self.device = args.device
        self.env = gym.make(args.game, start_level=args.seed)
        self.window = args.history_length  # Number of frames to concatenate
        self.training = True  # Consistent with model training mode
        self.obs_shape = (self.window, H, W)
        self._obs = np.zeros(shape=self.obs_shape, dtype=np.uint8)
        self._step_counter = 0

    def get_obs(self):
        return self._obs.copy()

    def _get_state(self):
        state = cv2.resize(self.get_obs(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    @property
    def action_space(self):
        return self.env.action_space.n

    def _reset_buffer(self):
        self._obs = np.zeros(shape=self.obs_shape, dtype=np.uint8)

    def reset(self):
        self.env.reset()
        self._reset_buffer()
        self._update_obs()
        self._step_counter = 0

        observation = self._get_state()
        return observation

    def _update_obs(self):
        _, img, _ = self.env.env.observe()
        gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
        resized_gray = cv2.resize(gray, (W, H), cv2.INTER_NEAREST)
        self._obs = np.concatenate([self._obs[1:], resized_gray[np.newaxis]], axis=0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            _, r, done, _ = self.env.step(action)
            reward += r
            self._update_obs()
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        # rew = np.array(0., dtype=np.float32)
        # for _ in range(self._frame_skip - 1):
        #     _, r, done, info = self.env.step(action)
        #     rew += r
        # game_obs, r, done, info = self.env.step(action)
        # rew += r
        # self._update_obs()
        # # TODO: maybe do something about EnvInfo?
        # info = EnvInfo(**info)
        # self._step_counter += 1
        # return EnvStep(self.get_obs(), reward, done, info)


    def render(self, wait=10, show_full_obs=False):
        """Shows game screen via cv2, with option to show all frames in observation."""
        img = self.get_obs()
        if show_full_obs:
            shape = img.shape
            img = img.reshape(shape[0] * shape[1], shape[2])
        else:
            img = img[-1]
        cv2.imshow(self._id, img)
        cv2.waitKey(wait)
