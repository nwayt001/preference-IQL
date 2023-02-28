#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:05:42 2022

# Test different VPT models on the BASALT environment tasks

@author: nicholas

@TODO: add training code to VPT
"""
import sys, os
from argparse import ArgumentParser
from email.policy import default
import pickle
import numpy as np
import gym
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from vpt_agent.openai_vpt.agent import MineRLAgent, ENV_KWARGS
#sys.path.insert(0,'../')
from human_agent.human_play_interface import HumanPlayInterface
import tkinter as tk
from tkinter import filedialog


# rollout VPT agent on the 4 different BASALT environment
def main(args):
    # load environment
    #env = HumanSurvival(**ENV_KWARGS).make()


#{'attack': array([1]), 'back': array([0]), 'forward': array([0]), 'jump': array([0]), 'left': array([0]), 'right': array([0]), 'sneak': array([0]), 'sprint': array([0]), 'use': array([0]), 'drop': array([0]), 'inventory': array([0]), 'hotbar.1': array([0]), 'hotbar.2': array([0]), 'hotbar.3': array([0]), 'hotbar.4': array([0]), 'hotbar.5': array([0]), 'hotbar.6': array([0]), 'hotbar.7': array([0]), 'hotbar.8': array([0]), 'hotbar.9': array([0]), 'camera': array([[0., 0.]]), 'ESC': 0}


    # decode Environment
    if args.env == 'cave':
        env = gym.make('MineRLBasaltFindCave-v0')
    elif args.env == 'waterfall':
        env = gym.make('MineRLBasaltMakeWaterfall-v0')
    elif args.env == 'animal':  
        env = gym.make('MineRLBasaltCreateVillageAnimalPen-v0')
    elif args.env == 'house':
        env = gym.make('MineRLBasaltBuildVillageHouse-v0')
    else:
        env = gym.make(args.env)

    # add human play interface wrapper
    if args.use_human_interface or args.use_big_screen:
        print('Using human interface')
        env = HumanPlayInterface(env)

    # get agent weights/path
    root = tk.Tk()
    root.withdraw()
    weights_path = filedialog.askopenfilename()

    # load agent
    agent_parameters = pickle.load(open(args.model_directory + args.model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(device='cuda', policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, Q_head=False)
    agent.load_weights(weights_path)
    
    # set env seed
    #env.seed(args.seed)
    
    # reset environment
    obs = env.reset()
    camera_cntr = 0
    camera_state_x = 0
    camera_state_y = 0
    # rollout the agent
    #episode_end = False
    #end_cntr = 0
    while True:
        minerl_action = agent.get_action(obs)
        minerl_action["ESC"] = 0
        # prevent agent from dropping equipment/items
        minerl_action["drop"] = 0
        minerl_action["inventory"] = 0
        minerl_action["sneak"] = 0

        '''
        if minerl_action["use"] == 1:
            print('use pressed')
            episode_end = True

        if episode_end == True:
                end_cntr += 1

        if episode_end == True and end_cntr > 100:
                minerl_action["ESC"] = 1
                obs, reward, done, info = env.step(minerl_action)
                done = True
        '''

        camera_state_x += minerl_action["camera"][0][0]
        camera_state_x = np.min([camera_state_x, 180])
        camera_state_x = np.max([camera_state_x, -180])
        camera_state_y += minerl_action["camera"][0][1]
        camera_state_y = np.min([camera_state_y, 180])
        camera_state_y = np.max([camera_state_y, -180])
        if camera_cntr == 400:
            print("camera x:", camera_state_x)
            print("camera y:", camera_state_y)
            print("adjusting camera")
            #minerl_action["camera"]=[0, -100]
            rand_num = np.random.randint(-180, 180)
            #minerl_action["camera"]=[-camera_state_x, -camera_state_y + rand_num]
            camera_state_x=0
            camera_state_y=rand_num
            camera_cntr = 0
        camera_cntr += 1

        if args.use_human_interface:
            obs, reward, done, info = env.step(minerl_action, True)
        else:
            obs, reward, done, info = env.step(minerl_action)
            

# main entry point
if __name__ == '__main__':
    parser = ArgumentParser("Run pretrained models on MineRL environment")
    parser.add_argument("--weights-directory", type=str, default="/home/nicholas/mineRL/kairos_minerl_22/train/")
    parser.add_argument("--model-directory", type=str, default="/home/nicholas/mineRL/kairos_minerl_22/train/")
    parser.add_argument("--weights", "-w", type=str, default="foundation-model-2x.weights")
    parser.add_argument("--model", "-m", type=str, default="foundation-model-2x.model")
    parser.add_argument("--env", "-e", type=str, default="MineRLBasaltFindCave-v0")
    parser.add_argument("--seed", "-s", type=int, default=1337)
    parser.add_argument("--use-human-interface", action="store_true")
    parser.add_argument("--use-big-screen", type=bool, default=True)
    args = parser.parse_args()

    main(args)
