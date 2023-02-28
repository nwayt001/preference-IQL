#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:05:42 2022

@author: nicholas
"""
from code import interact
import gym
import minerl
from minerl.human_play_interface.human_play_interface import HumanPlayInterface


# Uncomment to see more logs of the MineRL launch
# import coloredlogs
# coloredlogs.install(logging.DEBUG)

env = gym.make("MineRLBasaltBuildVillageHouse-v0")

# add human play interface wrapper
env = HumanPlayInterface(env)

obs = env.reset()

done = False
while not done:
    ac = env.action_space.noop()
    # Spin around to see what is around us
    ac["camera"] = [0, 0]
    ac["forward"] = 0
    ac["inventory"] = 0
    ac["use"] = 0
    ac["ESC"] = 0
    obs, reward, done, info = env.step(ac, override_if_human_input = True)
    #env.render()
env.close()

