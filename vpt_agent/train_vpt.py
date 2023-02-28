#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:05:42 2022

# Test different VPT models on the BASALT environment tasks

@author: nicholas

@TODO: add training code to VPT
"""
import sys
from argparse import ArgumentParser
from email.policy import default
import pickle
from agent import MineRLAgent
import gym
sys.path.insert(0,'../')
from human_agent.human_play_interface import HumanPlayInterface
import torch as th
from os import listdir
from os.path import isfile, join
import numpy as np
from lib.torch_util import default_device_type

# train VPT
def main(args):
    env = gym.make(args.env)
    # get files in directory
    files = [f for f in listdir(args.data_directory) if isfile(join(args.data_directory, f))]

    # load data
    data = dict()
    data['observations'] = []
    data['actions'] = []
    
    for i in range(len(files)):
        tmp = np.load(args.data_directory + files[0], allow_pickle=True)
        tmp = tmp.tolist()
        if i == 0:
            data['observations'] = np.array(tmp['observations'])
        else:
            data['observations'] = np.concatenate([data['observations'], np.array(tmp['observations'])])
        data['actions'].extend(tmp['actions'])
    

    #TODO process the actions to get them into the format that the agent/network expects
    
    # load vpt agent
    agent_parameters = pickle.load(open(args.model_directory + args.model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(args.weights_directory + args.weights)

    # setup loss function and optimizer
    loss_fn = th.nn.NLLLoss()
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=0.001)
    
    # get device
    device = default_device_type()
    
    # put model into training mode
    agent.policy.train()
    optimizer.zero_grad()
    
    # split data into train/test/validation
    
    # loop over each observation in the training set
    
    
    # scratch code
    minerl_obs = dict()
    minerl_obs["pov"] = np.squeeze(data['observations'][1,:,:,:])
    outputs = agent.get_action(minerl_obs)
    agent_action = agent.get_action_raw(minerl_obs)
    
    #agent_input = np.squeeze(data[1,:,:,:])
    #agent_input = {"img": th.from_numpy(agent_input).to(device)}
    
    action = {
            "buttons": agent_action["buttons"].cpu().numpy(),
            "camera": agent_action["camera"].cpu().numpy()
        }
    
    
    outputs2 = agent._env_action_to_agent(outputs)
    
    action = data['actions'][0]
    action['camera'] = np.array(action['camera'])
    action['camera'] = outputs['camera']
    outputs3=  agent._env_action_to_agent(action)
    
    out1 = agent.action_mapper.from_factored(action)
    
    
    # end scratch code
    
    loss = loss_fn(agent.forward(outputs), th.tensor(actions))
    loss.backward()
    optimizer.step()


# main entry point
if __name__ == '__main__':
    parser = ArgumentParser("Run pretrained models on MineRL environment")
    parser.add_argument("--weights-directory", type=str, default="/home/nicholas/mineRL/Video-Pre-Training/weights/")
    parser.add_argument("--model-directory", type=str, default="/home/nicholas/mineRL/Video-Pre-Training/model/")
    parser.add_argument("--weights", "-w", type=str, default="bc-house-3x.weights")
    parser.add_argument("--model", "-m", type=str, default="foundation-model-3x.model")
    parser.add_argument("--env", "-e", type=str, default="MineRLBasaltBuildVillageHouse-v0")
    parser.add_argument("--use-human-interface", action="store_true")
    parser.add_argument("--data-directory", type=str, default="/home/nicholas/mineRL/kairos_minerl_22/human_agent/minerl_demonstrations/")
    
    args = parser.parse_args()

    main(args)
