from code import interact
import os
import time
import logging
import gym
import minerl
from minerl.human_play_interface.human_play_interface import HumanPlayInterface

from drlhp import HumanPreferencesEnvWrapper

if __name__ == '__main__':
    # setup environment
    env = gym.make("MineRLBasaltBuildVillageHouse-v0")

    # add learning from human preferences wrapper
    os.makedirs('preferences', exist_ok=True)
    env = HumanPreferencesEnvWrapper(
        env,
        segment_length=100,
        max_prefs_in_db=10000,
        max_pref_interface_segs=10000,
        collect_prefs=True,
        train_reward=False,
        synthetic_prefs=False,
        prefs_dir='preferences', # path of existing preferences
        log_dir='preferences',
        zoom_ratio=1,
        obs_transform_func=lambda x:x['pov']) # returns only pixels as obs


    # # add human play interface wrapper
    # env = HumanPlayInterface(env)

    print('Collecting data...')
    obs = env.reset()
    done = False
    step = 0

    while not done:
        # if step % 20 == 0:
        #     print(f"ENV STEP {step}")
        ac = env.action_space.sample()
        ac['ESC'] = 0 # prevents premature end of episode
        ac["camera"] = [0, 0] # no camera movements
        ac["inventory"] = 0
        ac["use"] = 0
        obs, reward, done, info = env.step(ac)#, override_if_human_input = True)
        step += 1
        #env.render()
        
        # save preferences and make sure we are not using trained rewards
        # print(f"Using trained rewards? {env.using_reward_from_predictor}")
        env.save_prefs()

    env.close()