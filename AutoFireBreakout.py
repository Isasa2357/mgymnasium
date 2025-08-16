
from numpy import ndarray

import os
import numpy as np
import cv2

import gymnasium as gym
import ale_py
from gymnasium import Env

class AutoFireBreakout:
    def __init__(self, frame_skip: int=4, fires_count: int=1):
        self._base_env = gym.make("ALE/Breakout-v5", render_mode="human")
        self._frame_skip = frame_skip
        self._need_fire = True
        self._fires_count = fires_count
        self._fire_action = 1
        self._lifeCrop1 = cv2.imread(os.path.join('mgymnasium', 'breakout_lifeCrip', 'lifeCrop1.png'))
        self._lifeCrop2 = cv2.imread(os.path.join('mgymnasium', 'breakout_lifeCrip', 'lifeCrop2.png'))
        self._lifeCrop3 = cv2.imread(os.path.join('mgymnasium', 'breakout_lifeCrip', 'lifeCrop3.png'))
        self._lifeCrop4 = cv2.imread(os.path.join('mgymnasium', 'breakout_lifeCrip', 'lifeCrop4.png'))
        self._lifeCrop5 = cv2.imread(os.path.join('mgymnasium', 'breakout_lifeCrip', 'lifeCrop5.png'))
        self._cnt_liefCrop = self._lifeCrop5
    
    def reset(self):
        self._need_fire = True
        self._cnt_liefCrop = self._lifeCrop5
        return self._base_env.reset()
    
    def step(self, action: int):
        if self._need_fire:
            self._fire()

        status_lst = list()
        total_reward = 0.0
        done = False
        for _ in range(self._frame_skip):
            next_state, reward, terminated, truncated, _ = self._base_env.step(action)
            status_lst.append(next_state)
            total_reward += float(reward)
            done = terminated or truncated

        self.update_need_fire(status_lst[-1])
        
        return np.array(status_lst, dtype=np.int8), total_reward, done
    
    def _fire(self):
        for _ in range(self._fires_count):
            print('fire')
            self._base_env.step(self._fire_action)
        self._need_fire = False
    
    def update_need_fire(self, state: ndarray):
        lifeCrop = self.get_lifeCrop(state)
        if not np.array_equal(lifeCrop, self._cnt_liefCrop):
            self._need_fire = True
        self._cnt_liefCrop = lifeCrop

    def get_lifeCrop(self, state: ndarray):
        '''
        残機部分を繰り抜く
        '''
        chnl1_begin = 0
        chnl1_end = 17
        chnl2_begin = 100
        chnl2_end = 112
        return state[chnl1_begin:chnl1_end, chnl2_begin:chnl2_end, :]

env = AutoFireBreakout()

done = False
state, _ = env.reset()
while not done:
    print('hello')
    action = np.random.choice([2, 3])
    next_state, reward, done = env.step(action)
    state = next_state