# Basic behavioural cloning
# Note: this uses gradient accumulation in batches of ones
#       to perform training.
#       This will fit inside even smaller GPUs (tested on 8GB one),
#       but is slow.

from argparse import ArgumentParser
import pickle
import time

import gym
import minerl
import torch as th
import numpy as np

from vpt_agent.openai_vpt.agent import PI_HEAD_KWARGS, MineRLAgent
from vpt_agent.data_loader import DataLoader
from vpt_agent.openai_vpt.lib.tree_util import tree_map

# Originally this code was designed for a small dataset of ~20 demonstrations per task.
# The settings might not be the best for the full BASALT dataset (thousands of demonstrations).
# Use this flag to switch between the two settings
USING_FULL_DATASET = True

EPOCHS = 1 if USING_FULL_DATASET else 2
# Needs to be <= number of videos
BATCH_SIZE = 64 if USING_FULL_DATASET else 16
# Ideally more than batch size to create
# variation in datasets (otherwise, you will
# get a bunch of consecutive samples)
# Decrease this (and batch_size) if you run out of memory
N_WORKERS = 100 if USING_FULL_DATASET else 20
DEVICE = "cuda"

LOSS_REPORT_RATE = 100
SAVE_WEIGHTS_RATE = 1000
# Tuned with bit of trial and error
LEARNING_RATE = 0.000181
# OpenAI VPT BC weight decay
# WEIGHT_DECAY = 0.039428
WEIGHT_DECAY = 0.0
# KL loss to the original model was not used in OpenAI VPT
KL_LOSS_WEIGHT = 0.1 if USING_FULL_DATASET else 1.0
MAX_GRAD_NORM = 5.0

# Calibrated to roughly 12 hours of training with max_batches = 14000
# NRW: on a 2080TI with 11gb of memory, it takes 20 hours to do 44000 batches (full dataset is closer to 97k batches)
MAX_BATCHES = 100000

# TODO: 
# 1. implement layer wise loss weighting
# 2. more sophisticated null action filtering
# 3. and variable to track hidden state of original model for proper KL loss


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

def behavioural_cloning_train(data_dir, in_model, in_weights, out_weights, KL_LOSS_WEIGHT=KL_LOSS_WEIGHT, freeze_parameters=True):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    # To create model with the right environment.
    # All basalt environments have the same settings, so any of them works here
    env = gym.make("MineRLBasaltFindCave-v0")
    agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)

    # Create a copy which will have the original parameters
    original_agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    original_agent.load_weights(in_weights)
    env.close()

    policy = agent.policy
    original_policy = original_agent.policy

    if freeze_parameters == True:
        # Freeze most params
        for param in policy.parameters():
            param.requires_grad = False
        # Unfreeze final layers
        trainable_parameters = []
        for param in policy.net.lastlayer.parameters():
            param.requires_grad = True
            trainable_parameters.append(param)
        for param in policy.pi_head.parameters():
            param.requires_grad = True
            trainable_parameters.append(param)

    else:
        trainable_parameters = policy.parameters()

    # check the number of parameters
    pytorch_total_params = sum(p.numel() for p in policy.parameters())
    print("Total number of parameters: {}".format(pytorch_total_params))

    # check the number of trainable parameters
    pytorch_total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("Total number of trainable parameters: {}".format(pytorch_total_params))

    # Parameters taken from the OpenAI VPT paper
    # TODO: add layer specific learning rates
    optimizer = th.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    data_loader = DataLoader(
        dataset_dir=data_dir,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS
    )

    start_time = time.time()

    # Keep track of the hidden state per episode/trajectory.
    # DataLoader provides unique id for each episode, which will
    # be different even for the same trajectory when it is loaded
    # up again
    episode_hidden_states = {}
    dummy_first = th.from_numpy(np.array((False,))).to(DEVICE)

    loss_sum = 0
    loss_items = 0
    for batch_i, (batch_images, batch_actions, batch_episode_id) in enumerate(data_loader):
        batch_loss = 0
        for image, action, episode_id in zip(batch_images, batch_actions, batch_episode_id):
            # clean up 
            if image is None and action is None:
                # A work-item was done. Remove hidden state
                if episode_id in episode_hidden_states:
                    removed_hidden_state = episode_hidden_states.pop(episode_id)
                    del removed_hidden_state
                continue

            # get agent action, and skip if action is null (no-op action)
            # Nick W: with this code in place, we are automatically deciding to skip over and ignore all no-op actions
            # in the dataset. mostly this is good since the dataset contains a lot of no-op actions, however,
            # we may want to change this to include at least some no-op actions in the dataset.
            agent_action = agent._env_action_to_agent(action, to_torch=True, check_if_null=True)
            if agent_action is None:
                # Action was null
                continue

            agent_obs = agent._env_obs_to_agent({"pov": image})
            if episode_id not in episode_hidden_states:
                episode_hidden_states[episode_id] = policy.initial_state(1)
            agent_state = episode_hidden_states[episode_id]

            pi_distribution, v_prediction, new_agent_state = policy.get_output_for_observation(
                agent_obs,
                agent_state,
                dummy_first
            )

            with th.no_grad():
                original_pi_distribution, _, _ = original_policy.get_output_for_observation(
                    agent_obs,
                    agent_state,
                    dummy_first
                )

            log_prob  = policy.get_logprob_of_action(pi_distribution, agent_action)
            kl_div = policy.get_kl_of_action_dists(pi_distribution, original_pi_distribution)

            # Make sure we do not try to backprop through sequence
            # (fails with current accumulation)
            new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
            episode_hidden_states[episode_id] = new_agent_state

            # Finally, update the agent to increase the probability of the
            # taken action.
            # Remember to take mean over batch losses
            loss = (-log_prob + KL_LOSS_WEIGHT * kl_div) / BATCH_SIZE
            batch_loss += loss.item()
            loss.backward()

        th.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

        loss_sum += batch_loss
        if batch_i % LOSS_REPORT_RATE == 0:
            time_since_start = time.time() - start_time
            print(f"Time: {time_since_start:.2f}, Batches: {batch_i}, Avrg loss: {loss_sum / LOSS_REPORT_RATE:.4f}")
            loss_sum = 0

        if batch_i > MAX_BATCHES:
            print("Max batches reached, stopping training before full epoch")
            break

        # Save the weights
        if batch_i is not 0 and batch_i % SAVE_WEIGHTS_RATE == 0:
            state_dict = policy.state_dict()
            th.save(state_dict, out_weights)
            print(f"Saved weights to {out_weights}")

    
    # Save the final weights
    state_dict = policy.state_dict()
    th.save(state_dict, out_weights)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the directory containing recordings to be trained on")
    parser.add_argument("--in-model", required=True, type=str, help="Path to the .model file to be finetuned")
    parser.add_argument("--in-weights", required=True, type=str, help="Path to the .weights file to be finetuned")
    parser.add_argument("--out-weights", required=True, type=str, help="Path where finetuned weights will be saved")
    parser.add_argument("--freeze-parameters", type=bool, default=True)

    args = parser.parse_args()
    behavioural_cloning_train(args.data_dir, args.in_model, args.in_weights, args.out_weights, freeze_parameters = args.freeze_parameters)
