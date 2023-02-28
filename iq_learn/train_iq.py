"""
Copyright 2022 Div Garg. All rights reserved.

Example training code for IQ-Learn which minimially modifies `train_rl.py`.
"""

import datetime
import os
import random
import time
import sys
import copy

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from agent.vpt import MinecraftVPT
#from utils.utils import eval_mode
from utils.logger import Logger
from utils.utils import load

from dataset.minecraft_data_loader_wrapper import DataLoaderWrapper
from dataset.online_queue import OnlineQueue

from multiprocessing import active_children

# forces pytorch to use only cpu
FORCE_CPU = False

torch.set_num_threads(2)


def get_args(cfg: DictConfig):
    if FORCE_CPU:
        torch.cuda.is_available = lambda : False
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
        
    # Ensure phase-specific arguments are set correctly: 
    if cfg.phase == 1:    # IQ-learn-only phase
    
        cfg.human_prefs = False
    
        cfg.unfreeze_last_vpt_layer = True
        
        cfg.train.batch = 4
        cfg.online_queue.num_env_runners = 4
        
        cfg.use_reward = True
        #cfg.reward.iq_term = True
        cfg.feed_learned_reward_into_Q = False
    
    elif cfg.phase == 2:  # IQ-learn + reward-learning phase
    
        cfg.human_prefs = True
    
        cfg.unfreeze_last_vpt_layer = False

        # Nick suggested having only 1 environment runner process in Phase 2,
        # so that we can collect segments for pairwise preferences more quickly.
        cfg.train.batch = 1
        cfg.online_queue.num_env_runners = 1
        
        cfg.offline = False

        cfg.use_reward = True
        #cfg.reward.iq_term = False
        cfg.feed_learned_reward_into_Q = True
    
    else:
        raise NotImplementedError('Phase must be equal to 1 or 2.')

    # Make sure correct iq loss terms are being used depending on whether 
    # offline or online learning:
    if cfg.offline:
        
        cfg.method.loss = "value_expert"        
        cfg.method.chi = False  #True
        cfg.method.regularize = False
        
    else:   # online
        
        cfg.method.loss = "value"
        cfg.method.chi = False
        cfg.method.regularize = False  # True

    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    wandb.init(project=args.project_name, entity=args.wandb.username,
               sync_tensorboard=True, reinit=True, config=args)
    
    print('wandb run name:', wandb.run.name)
    
    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    phase1_model_path = 'train/' + args.env.name + '_phase_1'

    # Load policy:
    if args.pretrain or args.phase == 2:   # Load policy checkpoint
        
        if args.phase == 2:
            pretrain_path = phase1_model_path
            pretrain_path = pretrain_path + '.weights'
        else:
            pretrain_path = hydra.utils.to_absolute_path(args.pretrain)
            
        if os.path.isfile(pretrain_path):
            print("=> loading pretrain '{}'".format(pretrain_path))
            minecraft_agent, policy = load(args, device, weights_path = pretrain_path)
        else:
            raise Exception("Did not find checkpoint {}".format(pretrain_path))
    else:   # Load original VPT weights
    
        minecraft_agent, policy = load(args, device)
    
    # Instantiate learning agent:
    vpt_agent = MinecraftVPT(args, policy, minecraft_agent)
    
    if not args.offline:

        online_queue = OnlineQueue(args, vpt_agent, device = 'cpu')

    # Load expert data
    data_dir = hydra.utils.to_absolute_path(f'data/{args.env.name}')
    expert_data_loader = DataLoaderWrapper(args, data_dir, minecraft_agent)

    # Setup log directory and logging
    exp_name_str = '_' + args.env.name
    if args.exp_name != '':
        exp_name_str += '_' + args.exp_name
    log_dir = os.path.join(args.log_dir, 
                           datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d/%H-%M-%S" + exp_name_str))
    writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')
    logger = Logger(args.log_dir,
                    log_frequency=args.log_interval,
                    writer=writer,
                    save_tb=True,
                    agent=args.agent.name)

    learn_steps = 0
    begin_learn = False
    epoch = 0     # Keeps track of how many times we save the model

    start_time = time.time()
    last_save_time = start_time

    while True:
        
        if not args.offline:
            # Make sure env runner processes are alive, and restart if needed:
            online_queue.check_processes_alive()
        
        # Check whether we have enough online data to perform an online gradient
        # update. In offline mode, this always evaluates to True.
        if args.offline or online_queue.has_sufficient_data(begin_learn):
   
            # Start learning
            if begin_learn is False:
                print('Main process: learn begins!')
                begin_learn = True

            # Print progress info:
            print('Main process: Learn steps completed: %i' % learn_steps)

            learn_steps += 1
            
            tasks_proc, total_tasks = expert_data_loader.get_progress_info()
            print('Main process: Expert demos completed or currently being processed: %i of %i (%3.2f percent)' \
                  % (tasks_proc, total_tasks, 100 * (tasks_proc/total_tasks)))

            if not args.offline:
                
                policy_batch = online_queue.get_samples(device)
            else:
                policy_batch = None
                
            # Batch size is specified to data loader when it's initialized, so we don't
            # need to pass it here:
            expert_batch = expert_data_loader.get_samples(device)

            # Check if learning is done! There are 2 ways to finish:
            # 1) We finished going through all the expert data
            # 2) We're using human preferences, and the human pressed the "finish" 
            #    button in the preference GUI.
            if expert_batch is None or (not args.offline and args.human_prefs and online_queue.is_finished()):
            
                if not args.offline:
                    online_queue.finish()   # If we finished via condition 1), we need to tell
                                            # the online queue to finish.
                break

            losses = vpt_agent.iq_update(policy_batch, expert_batch, logger, learn_steps)

            if learn_steps % args.log_interval == 0:
                for key, loss in losses.items():
                    writer.add_scalar(key, loss, global_step=learn_steps)
                    
            # Share updated policy with env runner processes:
            if not args.offline and learn_steps % args.train.policy_update_steps == 0:
                
                state_dict = copy.deepcopy(vpt_agent.policy.state_dict())
                
                # Copy state_dict to correct device for online_queue (likely GPU --> CPU)
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].to(online_queue.device)
                
                online_queue.shared_policy.load_state_dict(state_dict)
                
                # Signal to env runner processes that their local policy weights should be updated:
                for flag in online_queue.policy_update_flags:
                    flag.value = True

            # Log the time after every gradient update:
            logger.log('train/total_time', time.time() - start_time, learn_steps)

            if learn_steps % args.train.save_rate == 0:
                
                logger.log('train/episode', epoch, learn_steps)
                #logger.log('train/episode_reward', episode_reward, learn_steps)
                logger.log('train/duration', time.time() - last_save_time, learn_steps)
                last_save_time = time.time()
                
                logger.dump(learn_steps, save=begin_learn)
                save(vpt_agent, epoch, args, output_dir=log_dir)
                
                epoch += 1
                
        else:   
            
            # OK to sleep between checks to see whether there's enough data
            if begin_learn:
                print('Main process: WARNING: learning process is sleeping. This means we ran out of environment data.')
            
            time.sleep(1)

    # Do final logging, and save final model:
    logger.log('train/episode', epoch, learn_steps)
    #logger.log('train/episode_reward', episode_reward, learn_steps)
    logger.log('train/duration', time.time() - last_save_time, learn_steps)
    
    logger.dump(learn_steps, save=begin_learn)
    
    if args.phase == 1:
        save(vpt_agent, epoch, args, output_dir=log_dir, extra_model_path = phase1_model_path)
    else:
        save(vpt_agent, epoch, args, output_dir=log_dir)
             
    wandb.finish()

    # check for active child processes
    active = active_children()

    # terminate all active children
    for child in active:
        child.terminate()

    # block until all children have closed
    for child in active:
        child.join()

    # report active children
    active = active_children()

    print('Main process: Finished!')
    

def save(agent, epoch, args, output_dir='results', extra_model_path = None):

    if args.method.type == "sqil":
        name = f'sqil_{args.env.name}'
    else:
        name = f'iq_{args.env.name}'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    agent.save(f'{output_dir}/{args.agent.name}_{name}')
    
    # Also save to train directory:
    if extra_model_path is not None:
        agent.save(extra_model_path)



if __name__ == "__main__":
    sys.argv.append('hydra.run.dir=.')
    main()
