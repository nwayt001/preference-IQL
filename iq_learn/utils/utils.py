
import torch
from torch import nn
import numpy as np
import pickle

from vpt_agent.openai_vpt.agent import MineRLAgent


# Load model parameters
def load(args, device, weights_state_dict = None, weights_path = None):

    model_num = args.vpt_model

    model = 'train/foundation-model-' + str(model_num) + 'x.model'
    
    if weights_path is None:
        weights = 'train/foundation-model-' + str(model_num) + 'x.weights'
    else:
        weights = weights_path

    #print("---Loading model---")
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

    agent = MineRLAgent(args, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs,
                         Q_head = True, device=device, reward_head = args.use_reward)
    
    agent.load_weights(weights, weights_state_dict)


    return agent, agent.policy 

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def evaluate(actor, env, episode_id, num_episodes=10, vis=True):
    """Evaluates the policy.
    Args:
      actor: A policy to evaluate.
      env: Environment to evaluate the policy on.
      num_episodes: A number of episodes to average the policy on.
    Returns:
      Averaged reward and a total number of steps.
    """
    total_timesteps = []
    total_returns = []

    while len(total_returns) < num_episodes:
        state = env.reset()
        done = False

        with eval_mode(actor):
            while not done:
                # action = actor.choose_action(state, episode_id, sample=False)
                action = actor.choose_action(state)
                next_state, reward, done, info = env.step(action)
                state = next_state

                if 'episode' in info.keys():
                    total_returns.append(info['episode']['r'])
                    total_timesteps.append(info['episode']['l'])

    return total_returns, total_timesteps


def weighted_softmax(x, weights):
    x = x - torch.max(x, dim=0)[0]
    return weights * torch.exp(x) / torch.sum(
        weights * torch.exp(x), dim=0, keepdim=True)


def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def hard_update(source, target):
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(param.data)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def get_concat_samples(policy_batch, expert_batch, args):
    online_batch_state, online_batch_next_state, online_batch_action, online_episode_id, online_batch_reward, online_batch_done = policy_batch

    expert_batch_state, expert_batch_next_state, expert_batch_action, expert_episode_id, expert_batch_reward, expert_batch_done = expert_batch

    # make sure all tensors are in the same device
    #online_batch_action = online_batch_action.to(args.device)

    if args.method.type == "sqil":
        # convert policy reward to 0
        online_batch_reward = torch.zeros(online_batch_reward.shape)
        # convert expert reward to 1
        expert_batch_reward = torch.ones(expert_batch_reward.shape)

    batch_state = torch.cat([online_batch_state, expert_batch_state], dim=0)
    batch_next_state = torch.cat(
        [online_batch_next_state, expert_batch_next_state], dim=0)
    batch_action = torch.cat([online_batch_action, expert_batch_action], dim=0)
    batch_reward = np.vstack((online_batch_reward, expert_batch_reward))
    batch_done = torch.cat([online_batch_done, expert_batch_done], dim=0)
    is_expert = torch.cat([torch.zeros(online_batch_reward.shape, dtype=torch.bool),
                           torch.ones(expert_batch_reward.shape, dtype=torch.bool)], dim=0)

    batch_episode_id = online_episode_id + expert_episode_id
    return batch_state, batch_next_state, batch_action, batch_episode_id, batch_reward, batch_done, is_expert


def save_state(tensor, path, num_states=5):
    """Show stack framed of images consisting the state"""

    tensor = tensor[:num_states]
    B, C, H, W = tensor.shape
    images = tensor.reshape(-1, 1, H, W).cpu()
    save_image(images, path, nrow=num_states)
    # make_grid(images)


def average_dicts(dict1, dict2):
    return {key: 1/2 * (dict1.get(key, 0) + dict2.get(key, 0))
                     for key in set(dict1) | set(dict2)}