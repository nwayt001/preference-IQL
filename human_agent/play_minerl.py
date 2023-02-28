#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:05:42 2022

Script to play MineRL using the human play interface

input arguments:
-env: environment name (e.g. "MineRLBasaltBuildVillageHouse-v0")

BASALT environments:
MineRLBasaltFindCave-v0
MineRLBasaltMakeWaterfall-v0
MineRLBasaltCreateVillageAnimalPen-v0
MineRLBasaltBuildVillageHouse-v0

@author: nicholas
"""
import gym
from argparse import ArgumentParser
import minerl
from human_play_interface import HumanPlayInterface


def main():
    parser = ArgumentParser("Play MineRL")
    parser.add_argument("--env", "-e", type=str, default="MineRLBasaltBuildVillageHouse-v0")
    args = parser.parse_args()

    # create environment
    env = gym.make(args.env)

    # add human play interface wrapper
    env = HumanPlayInterface(env)
    obs = env.reset()

    done = False
    while not done:
        ac = env.action_space.noop()
        obs, reward, done, info = env.step(ac, override_if_human_input = True)
    env.close()


# main entry point
if __name__ == '__main__':
    main()
