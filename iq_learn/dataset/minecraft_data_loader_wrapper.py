#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper for MineCraft data loader to provide interface expected by IQ-learn.

THIS VERSION: requires the expert batch size and number of data loader workers to be equal.
Episode IDs remain aligned between consecutive batches. For instance, with a batch
size of 5, the episode ids might be [0, 1, 2, 3, 4]. This stays the same every
iteration until one of these episodes finishes. Say episode 2 finishes first and the
next episode is episode 5. Then the episode ids will be [0, 1, 5, 3, 4] (and
not any other order.)
"""

import os
import sys
import numpy as np
import torch
from collections import deque

sys.path.append(os.path.abspath('./'))
from vpt_agent.data_loader import DataLoader


class DataLoaderWrapper():
    
    
    def __init__(self, args, data_dir, agent):
             
        self.batch_size = args.data_loader.batch_size
        
        self.max_episode_len = args.env.max_episode_steps

        # Assert expert/offline batch size = number of offline data loader workers:
        n_workers = args.data_loader.n_workers
        assert n_workers == self.batch_size, \
            'This implementation requires expert batch size = number of data loader workers.'
   
        # Data loader provided by the competition.
        if args.num_demos_skip > 0:
            print('Skipping the first %i demonstrations' % args.num_demos_skip)
        
        self.data_loader = DataLoader(dataset_dir=data_dir, n_workers=n_workers, 
                                 batch_size=self.batch_size, n_epochs=args.data_loader.epochs,
                                 num_demos_skip = args.num_demos_skip, demo_frac = args.data_loader.demo_frac)

        self.agent = agent
        
        # For storing consecutive samples (since we need to return state, next state pairs).
        # We create a deque corresponding to each spot in the batch. Each
        # data loader worker will feed into one of these deques.
        self.data = [deque() for i in range(self.batch_size)]
        
        # Keeps track of the length of each episode.
        self.ep_len = np.zeros(self.batch_size)
        
        self.finished = False    # Keeps track of whether we've finished processing
                                 # all expert data.
    
    def load_data_batch(self):
        """
        Query minecraft data loader for more data.
        """

        # Need at least 2 consecutive samples to get a (state, next_state).
        # So, sample batches from data loader until we have this for all episodes 
        # currently being processed.        
        data_len = np.array([len(d) for d in self.data])
        batch_idxs = np.nonzero(data_len < 2)[0]  # Indices in the batch with insufficient data
    
        # The following counter is just for debugging purposes. I don't think 
        # it's possible for the following while loop to get stuck running forever,
        # but just in case, let's print out warnings/diagnostics if it takes
        # more than a few iterations for it to complete.
        itr_counter = 0
        itrs_warn = 5    # Print warnings starting this number of iterations
    
        while len(batch_idxs) > 0:
            
            # Each data loader worker feeds into one of the batch indices
            new_data = self.data_loader.get_specific_next_outputs(batch_idxs)

            if new_data is None:
                self.finished = True
                return     # Finished all expert data
            else:
                batch_images, batch_actions, batch_episode_id = new_data
        
            for batch_idx, image, action, episode_id in zip(batch_idxs, batch_images, 
                                                            batch_actions, batch_episode_id):
                
                if image is None:     # Don't process episode done flags
                    continue
                
                obs = self.agent._env_obs_to_agent({"pov": image}, numpy = True)
                
                # Convert action to embedded form, since this is what we will use during iq-learn:
                action_emb = self.agent.policy.act_embedding.env_action_to_embedding(action, numpy = True)
                
                # We also need the policy representation of the action to calculate the BC loss:
                action_pol = self.agent._env_action_to_agent(action)
                
                # Add new experience tuple to data:
                self.data[batch_idx].append((obs, (action_emb, action_pol), episode_id))

            # Check if a warning is needed:
            if itr_counter >= itrs_warn:
                print('Main process: Expert data loader is on iteration ',
                      str(itr_counter), '! data len = ', data_len, 'batch_idxs = ', batch_idxs,
                      'ep len = ', self.ep_len)
                
            itr_counter += 1

            data_len = np.array([len(d) for d in self.data])
            batch_idxs = np.nonzero(data_len < 2)[0]   # Indices in the batch with insufficient data
    
        
    def get_samples(self, device):
        """
        Return a batch of samples in the format expected by iq-learn, i.e.:
        batch_state, batch_next_state, batch_action, batch_episode_id, batch_reward, batch_done
        """
        
        batch_size = self.batch_size
        
        # Initialize data to be returned
        batch_state = []
        batch_next_state = []
        batch_action_emb = np.empty((batch_size, 1, self.agent.policy.action_emb_size))
        
        batch_action_pol_buttons = torch.empty((batch_size, 1))
        batch_action_pol_camera = torch.empty((batch_size, 1))
        
        # We'll leave all the rewards as 0 for now, since we don't have a 
        # reward in the Minecraft environment.
        batch_reward = np.zeros((batch_size, 1), dtype = float)
        batch_done = np.zeros(batch_size, dtype = bool)   # All initialized to False

        batch_episode_id = np.zeros(batch_size).astype(int)

        # Need to get  data from data loader before proceeding
        self.load_data_batch()
    
        if self.finished:    # If finished processing all expert data, then we're done!
            return None

        # Get next experience tuple for each episode in the batch:
        for batch_idx in range(batch_size):
            
            state, action, episode_id = self.data[batch_idx].popleft()
            action_emb, action_pol = action
            
            batch_state.append(state['img'])
            batch_action_emb[batch_idx, 0] = action_emb
            
            batch_action_pol_buttons[batch_idx] = action_pol['buttons'][0][0]
            batch_action_pol_camera[batch_idx] = action_pol['camera'][0][0]
            
            batch_episode_id[batch_idx] = episode_id
            
            # Peek at next episode id to check whether the episode has finished:
            next_episode_id = self.data[batch_idx][0][2]
            
            if episode_id != next_episode_id:   # If true, we're on the last experience 
                                                # tuple in the episode
                # Note: we only say done=True if the demonstration does not
                # finish due to hitting the time limit. This has been shown to
                # improve learning.
                if self.ep_len[batch_idx] < self.max_episode_len:
                    batch_done[batch_idx] = True
                    
                next_state = state     # Make next state the same
                
                self.ep_len[batch_idx] = 0    # Keep track of episode length
        
            else:
                    
                next_state = self.data[batch_idx][0][0]   # Peek at next obs in episode
                self.ep_len[batch_idx] += 1    # Keep track of episode length
            
            batch_next_state.append(next_state['img'])            


        # Convert states to tensors:
        batch_state_tensor = np.array(batch_state)
        batch_state_tensor = torch.as_tensor(batch_state_tensor)
        batch_state_tensor = batch_state_tensor.to(device)

        batch_next_state_tensor = np.array(batch_next_state)
        batch_next_state_tensor = torch.as_tensor(batch_next_state_tensor)
        batch_next_state_tensor = batch_next_state_tensor.to(device)

        # Embedded representation of action:
        batch_action_emb_tensor = torch.as_tensor(batch_action_emb)        
        batch_action_emb_tensor = batch_action_emb_tensor.float().to(device)
        
        # Policy representation of action:
        batch_action_pol = {'buttons': batch_action_pol_buttons.to(device), 
                            'camera': batch_action_pol_camera.to(device)}
        
        batch_done = torch.as_tensor(batch_done, dtype=torch.float, device = device).unsqueeze(1)
        
        # Batch episode ID should be a list, NOT pytorch tensor.
        #batch_episode_id = torch.as_tensor(batch_episode_id, dtype=torch.int).unsqueeze(1)
        return batch_state_tensor, batch_next_state_tensor, batch_action_emb_tensor, list(batch_episode_id), \
               batch_reward, batch_done, batch_action_pol
    
    def get_progress_info(self):
        return self.data_loader.get_progress_info()

