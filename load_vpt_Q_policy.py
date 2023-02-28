#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For now, this script just loads the Q-head version of the policy. Will add more to
this while testing it.
"""

import numpy as np
from argparse import ArgumentParser
import pickle
import gym
from vpt_agent.openai_vpt.agent import MineRLAgent, ENV_KWARGS
#sys.path.insert(0,'../')
from human_agent.human_play_interface import HumanPlayInterface



# rollout VPT agent on the 4 different BASALT environment
def main(args):
    # load environment
    #env = HumanSurvival(**ENV_KWARGS).make()
    env = gym.make(args.env)

    # add human play interface wrapper
    if args.use_human_interface:
        print('Using human interface')
        env = HumanPlayInterface(env)

    # load agent
    agent_parameters = pickle.load(open(args.model_directory + args.model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    
    """
    The only change here is setting Q_head = True when instantiating the Mine-RL
    agent. This tells the MineRLAgent to instantiate a different type of policy.
    Q_head is False by default not to mess with existing code.
    """
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs,
                        Q_head = True)
    agent.load_weights(args.weights_directory + args.weights)
    

           

# main entry point
if __name__ == '__main__':
    parser = ArgumentParser("Run pretrained models on MineRL environment")
    parser.add_argument("--weights-directory", type=str, default="../vpt_models/")
    parser.add_argument("--model-directory", type=str, default="../vpt_models/")
    parser.add_argument("--weights", "-w", type=str, default="bc-house-3x.weights")
    parser.add_argument("--model", "-m", type=str, default="foundation-model-3x.model")
    parser.add_argument("--env", "-e", type=str, default="MineRLBasaltBuildVillageHouse-v0")
    parser.add_argument("--use-human-interface", action="store_true")
    args = parser.parse_args()

    main(args)
