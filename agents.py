#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Jul 23 2022

# Main Agent code for the BASALT competition

@author: nicholas

"""

import pickle
from  vpt_agent.openai_vpt.agent import MineRLAgent, ENV_KWARGS

# Basic Random Agent
class Agent():
    def __init__(self, env):
        # handle to the environment
        self.env = env

    # load agent
    def load_agent(self):
        pass

    # get agent action
    def get_action(self, obs):
        # get the action from the environment
        return self.env.action_space.sample()


# VPT Agent
class VPTAgent(Agent):
    def __init__(self, env):
        super().__init__(env)

        self.model_directory = "train/"
        self.model = "foundation-model-2x.model"
        self.weights = "rl-from-early-game-2x.weights"

        self.load_agent()

    # load vpt agent
    def load_agent(self):
        self.agent_parameters = pickle.load(open(self.model_directory + self.model, "rb"))
        policy_kwargs = self.agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = self.agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
        self.agent = MineRLAgent(self.env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
        self.agent.load_weights(self.model_directory + self.weights)

    # get agent action
    def get_action(self, obs):

        # get the action from the agent
        action = self.agent.get_action(obs)
        
        # post-process the action
        action["ESC"] = 0

        return action
        

# IQ-Learn Agent
class IQLearnAgent(Agent):
    def __init__(self, env, model="foundation-model-2x.model", weights="rl-from-early-game-2x.weights"):
        super().__init__(env)

        self.model_directory = "train/"
        self.model = model
        self.weights = weights

        self.load_agent()

    # load iq-learn based vpt agent
    def load_agent(self):
        self.agent_parameters = pickle.load(open(self.model_directory + self.model, "rb"))
        policy_kwargs = self.agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = self.agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
        self.agent = MineRLAgent(device='cuda', policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, Q_head=False)
        self.agent.load_weights(self.model_directory + self.weights)

    # reset the hidden state of the agent
    def reset_agent(self):
        self.agent.reset()

    # get agent action
    def get_action(self, obs):

        # get the action from the agent
        action = self.agent.get_action(obs)
        
        # post-process the action
        action["ESC"] = 0

        return action
        

# Preference-Based IQ-Learn Agent
# TODO Not implemented yet





        