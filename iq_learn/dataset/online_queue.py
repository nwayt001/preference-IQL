#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class for launching environment runners and online data queue.
"""

import numpy as np
import torch
from torch.multiprocessing import Process, Queue, Value, Lock
import time
import copy
from omegaconf import open_dict

from make_envs import make_env
from utils.utils import load



def env_runner(runner_idx, online_queue, shared_policy, policy_update_flag, 
               args, device, lock, done_learning):

    # Load policy:
    minecraft_agent, policy = load(args, device, weights_state_dict = shared_policy.state_dict())
    
    # Instantiate the environment:
    # first environment may collect human preferences via GUI
    if runner_idx == 0:
        env = make_env(args, collect_human_prefs=args.human_prefs)
    else:
        env = make_env(args)
    num_runners = args.online_queue.num_env_runners
    
    # Seed env
    # TODO: is this a good way to ensure all environment runners have different seeds?
    env.seed(args.seed + runner_idx)
    
    episode_steps = int(env.task.max_episode_steps)

    # Episode ids are: -1 - runner_idx - iter * num_runners, where iter = 0, 1, 2, ...
    # This ensures that all online episodes get unique and strictly negative episode ids.
    episode_id = -1 - runner_idx
    
    experience_list = []
    
    # Add new experience to the queue every few environment steps:
    steps_add_to_queue = args.online_queue.steps_add
    
    # Do policy rollouts in the environment:
    while True:

        state = env.reset()
        minecraft_agent.reset()    # Reset agent's policy hidden state
            
        #episode_reward = 0
        done = False
        
        for episode_step in range(episode_steps):
    
            if episode_step < args.num_seed_steps:
                # Seed replay buffer with random actions
                action = env.action_space.sample()
            else:
                action = choose_action(minecraft_agent, state)
                
            next_state, reward, done, _ = env.step(action)
            #episode_reward += reward
    
            if episode_step % 10 == 0:
                print(f'Env runner {runner_idx}: Step {episode_step} out of {episode_steps}')
    
            # only store done true when episode finishes without hitting timelimit (allow infinite bootstrap)
            done_no_lim = done
            
            if episode_step + 1 == episode_steps:
                done_no_lim = 0
                
                if not done:
                    print('Env runner %i: done = False at time horizon!' % runner_idx)
                    done = 1    # Make sure done = True if at episode horizon
    
            # apply transformations to actions and states before adding to online queue
            experience = (minecraft_agent._env_obs_to_agent(state, numpy = True),
                minecraft_agent._env_obs_to_agent(next_state, numpy = True),
                minecraft_agent.policy.act_embedding.env_action_to_embedding(action, numpy = True),
                episode_id,
                reward,
                done_no_lim
            )        
        
            experience_list.append(experience)

            # save human preferences to disk
            if runner_idx == 0 and args.human_prefs and episode_step % 10 == 0:
                env.save_prefs()
    
            if len(experience_list) >= steps_add_to_queue:
            
                # Add experience batch to queue:
                add_online_data_to_queues(online_queue, experience_list, lock)
                
                # Reset for next experience batch to add to queue:
                experience_list = []
                

            if policy_update_flag.value:
                
                policy_update_flag.value = False
                
                policy.load_state_dict(shared_policy.state_dict(), strict = False)

            if done:
                break
            state = next_state
            
            printed_sleep_status = False
            
            while True:    # This loop sleeps this process if either 1) the sleep
                           # flag is set to true or 2) there are too many items
                           # in the queue, i.e. they're not being unloaded.

                # checks if user decided to end learning with a keyboard press
                if args.human_prefs and runner_idx == 0:
                    
                    # check if user wants to end learning
                    if env.kill_pref_interface_flag.value:
                        print("PREF_INTERFACE WAS KILLED BY USER. SAVING MODELS...")
                
                        done_learning.value = True

                if done_learning.value:    # Exit when learning finishes
                    finish_env_runner(env, online_queue, runner_idx, args.human_prefs and (runner_idx == 0), lock)
                    return

                with lock:
                    qsize = online_queue.qsize()
                    
                if qsize >= 2 * steps_add_to_queue:
                    if not printed_sleep_status:
                        print('Env runner %i: sleeping, qsize = %i' % (runner_idx, qsize))
                        printed_sleep_status = True
                else:
                    if printed_sleep_status:
                        print('Env runner %i: stopped sleeping, qsize = %i' % (runner_idx, qsize))
                    break

                # still checks for new preferences and saves them to disk
                if runner_idx == 0 and args.human_prefs:
                    env.save_prefs()
                
                time.sleep(2)            

        episode_id -= num_runners

        if done_learning.value:    # Exit if learning has finished
            finish_env_runner(env, online_queue, runner_idx, args.human_prefs and (runner_idx == 0), lock)
            return
        
def finish_env_runner(env, online_queue, runner_idx, human_pref_process, lock):
    """
    Called when we want to exit env runner process.
    """
    
    if not human_pref_process:
        env.close()    # This gets called in the preference GUI process in preference mode
        
        with lock:  # Use lock to ensure process can't terminate while queue is still needed
            online_queue.cancel_join_thread()  # Need this for the program to exit smoothly
    print('Env runner %i: finished!' % runner_idx)
    
        
        
def add_online_data_to_queues(online_queue, experience_list, lock): 
    """
    Add new experience tuple to the queue.
    Format: (state, next_state, action, episode_id, reward, done_no_lim)
    
    done_no_lim only stores done=true when episode finishes without hitting timelimit (allow infinite bootstrap).
    This is used during learning.
    """

    with lock:
        for experience in experience_list:
            
            # Add experience to the queue:
            online_queue.put(experience)


def choose_action(minecraft_agent, state):
    
    """
    Returns a single action in the representation that can be directly 
    passed to the environment.
    
    Don't need to deal with VPT internal state here, since this is handled 
    separately in vpt_agent/openai_vpt/agent.py
    """
    action = minecraft_agent.get_action(state)
    action["ESC"] = 0

    return action
        

class OnlineQueue():
    
    def __init__(self, args, vpt_agent, device = 'cpu'):

        self.args = args
        
        self.num_env_runners = args.online_queue.num_env_runners
               
        self.device = device
        self.batch_size = args.train.batch
        
        assert self.num_env_runners == self.batch_size, 'Online batch size must equal number of env runners!'
        
        # To store experience collected by each process:
        self.online_queues = [Queue() for i in range(self.num_env_runners)]
        
        # Create policy object with shared weights:
        self.shared_policy = copy.deepcopy(vpt_agent.policy).to(device)
        self.shared_policy.share_memory()
        
        # Flags to indicate whether policy weights have been updated (when True,
        # environment runner processes know to copy over the policy weights.)
        self.policy_update_flags = [Value('b', False) for i in range(self.num_env_runners)]
        
        # Multiprocessing locks for reading/writing from shared queues:
        self.locks = [Lock() for i in range(self.num_env_runners)]
        
        self.done_learning = Value('b', False)
        
        # Start processes:
        self.processes = [Process(target = env_runner, args = (i, self.online_queues[i],
                                                          self.shared_policy, self.policy_update_flags[i],
                                                          self.args, device,
                                                          self.locks[i], self.done_learning),
                             daemon = False) \
                      for i in range(self.num_env_runners)]
        
        [process.start() for process in self.processes]


    def check_processes_alive(self):
        """
        Check that each environment runner process is still alive. If an environment 
        runner process has died, then restart it.
        """    
        for i in range(self.num_env_runners):
            
            if not self.processes[i].is_alive():
                
                print('Main process: detected process %i has died, restarting' % i)
                
                # Make sure env.seed will get a unique seed each time we fork a process
                with open_dict(self.args):
                    self.args.seed += self.num_env_runners
                
                process = Process(target = env_runner, args = (i, self.online_queues[i],
                                                                  self.shared_policy, self.policy_update_flags[i],
                                                                  self.args, self.device,
                                                                  self.locks[i], self.done_learning),
                                     daemon = False)
        
                process.start()
                
                self.processes[i] = process
        
        

    def has_sufficient_data(self, begin_learn):
        """
        Return True if every online data queue contains at least one experience tuple.
        """

        # Check size of each online queue:
        sizes = np.empty(self.num_env_runners)

        for i, (queue, lock) in enumerate(zip(self.online_queues, self.locks)):
            
            with lock:
                    
                sizes[i] = queue.qsize()
    
        if begin_learn:
            print('Main process: experience tuples stored in each online queue = ', sizes)
    
        return np.all(sizes >= 1)
            

    def next_batch(self):  

        batch = []        

        for i, (queue, lock) in enumerate(zip(self.online_queues, self.locks)):
            
            with lock:
                    
                experience = queue.get()    # Get next experience tuple

            batch.append(experience)
    
        return batch
    
    
    def get_samples(self, device):
        '''
        Return a batch of samples in the format expected by iq-learn, i.e.:
        batch_state, batch_next_state, batch_action, batch_episode_id, batch_reward, batch_done
        
        Also matches format used by batches taken using the data loader.
        '''
        batch = self.next_batch()

        batch_state, batch_next_state, batch_action, batch_episode_id, batch_reward, batch_done = zip(
            *batch)

        # Convert states to tensors:
        batch_state = [item['img'] for item in batch_state]
        batch_state_tensor = np.array(batch_state)
        batch_state_tensor = torch.as_tensor(batch_state_tensor)
        batch_state_tensor = batch_state_tensor.to(device)

        batch_next_state = [item['img'] for item in batch_next_state]
        batch_next_state_tensor = np.array(batch_next_state)
        batch_next_state_tensor = torch.as_tensor(batch_next_state_tensor)
        batch_next_state_tensor = batch_next_state_tensor.to(device)

        # Stack action as tensor:
        batch_action_tensor = np.array(batch_action)
        batch_action_tensor = torch.as_tensor(batch_action_tensor)     
        batch_action_tensor = batch_action_tensor.float().to(device)

        batch_reward = np.array(batch_reward, dtype = float)[:, np.newaxis]
        batch_done = torch.as_tensor(batch_done, dtype=torch.float, device = device).unsqueeze(1)
        batch_episode_id = list(batch_episode_id)
        
        return batch_state_tensor, batch_next_state_tensor, batch_action_tensor, batch_episode_id, batch_reward, batch_done


    def finish(self):
        
        self.done_learning.value = True
        
    def is_finished(self):
        
        return self.done_learning.value

