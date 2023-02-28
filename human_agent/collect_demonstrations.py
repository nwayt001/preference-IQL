#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:05:42 2022

Script to collect demonstrations in MineRL using the human play interface

input arguments:
-env: environment name (e.g. "MineRLBasaltBuildVillageHouse-v0")
-num_demos: number of demonstrations to collect
-save_dir: directory to save the demonstrations
-dowsample: whether to downsample the demonstration image sizes

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
import numpy as np
import time
from datetime import datetime
import cv2


def collect_demonstration(args):
    # create environment
    env = gym.make(args.env)

    # add human play interface wrapper
    env = HumanPlayInterface(env)

    for demo in range (args.num_demos):
        print("Collecting demonstration {}".format(demo))
        # get timestamp
        current_time = datetime.now()
        time_stamp = current_time.timestamp()
        date_time = datetime.fromtimestamp(time_stamp)

        observations = []
        actions = []

        # reset environment
        obs = env.reset()

        done = False
        while not done:
            ac = env.action_space.noop()
            obs, reward, done, info = env.step(ac, override_if_human_input = True)
            if args.downsample:
                obs["pov"] = cv2.resize(obs["pov"], (128, 128))
            observations.append(obs["pov"])
            actions.append(info['taken_action'])

        # save the observations and actions
        data = dict()
        data['observations'] = observations
        data['actions'] = actions
        np.save(args.save_dir + "/" + args.env + '_' + str(demo) + '_' + str(date_time) + '.npy', data)    

    # quit the environment
    env.close()


def main():
    parser = ArgumentParser("Play MineRL")
    # parser.add_argument("--env", "-e", type=str, default="MineRLBasaltBuildVillageHouse-v0")
    parser.add_argument("--env", "-e", type=str, default="MineRLBasaltFindCave-v0")
    parser.add_argument("--save-dir", "-s", type=str, default="minerl_demonstrations")
    parser.add_argument("--num-demos", "-n", type=int, default=1)
    parser.add_argument("--downsample", "-d", type=int, default=1)
    args = parser.parse_args()

    collect_demonstration(args)


# main entry point
if __name__ == '__main__':
    main()
