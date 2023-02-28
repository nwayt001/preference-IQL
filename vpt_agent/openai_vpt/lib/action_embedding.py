#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action embedding class. This defines an action embedding, meant to be given to the 
policy's Q-head.
The action embedding is concatenated with the state representation from the VPT
model, and then passed through the Q-head.
"""

import numpy as np
import torch


class ActionEmbedding:
    
    def __init__(self, agent, env_action_dim, type = 'tile', num_tiles = 20):
        """
        env_action_dim is the pre-embedded length of an action vector.
        num_tiles: number of times to tile the action with the action tiling 
                   embedding.
        
        For now, only action tiling is implemented, but I tried to set this
        up in such a way that we could add more.
        """
        
        self.agent = agent
        self.type = type
        self.num_tiles = num_tiles

        if type == 'tile':
            self.embed_function = self.tile_embed
            self.embedding_dim = num_tiles * env_action_dim
        else:
            raise NotImplementedError
    
    def policy_action_to_embedding(self, policy_action):
        """
        Convert action outputted by policy to embedding.
        """
        
        env_action = self.agent._agent_action_to_env(policy_action)
        env_action_emb = self.env_action_to_embedding(env_action)

        return env_action_emb
    
    def env_action_to_embedding(self, env_action, numpy = False):
        """
        Convert action in environment format to embedding.
        """
        
        action_np = self.env_action_to_numpy(env_action)
        
        env_action_emb = self.embed_function(action_np)
        
        # env_action_emb needs to be in torch so it can be exported to device
        if not numpy:
            env_action_emb = torch.from_numpy(env_action_emb)
        
        return env_action_emb
        
    def env_action_to_numpy(self, env_action):
        """
        Convert action in environment format to numpy array.
        """
        
        policy_repr = self.agent.action_transformer.env2policy(env_action)
        action_list = list(policy_repr.values())

        action_np = np.concatenate(action_list, axis=-1)   # Convert action to Numpy array
                
        return action_np
            
    def tile_embed(self, action):
        
        return np.tile(action, self.num_tiles)
    
    