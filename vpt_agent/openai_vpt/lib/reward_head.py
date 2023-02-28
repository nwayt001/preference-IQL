from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.init as init

from vpt_agent.openai_vpt.lib.mlp import MLP
from vpt_agent.openai_vpt.lib.action_head import fan_in_linear
from vpt_agent.openai_vpt.lib.normalize_ewma import NormalizeEwma


class RewardHead(nn.Module):
    """
    MLP reward network head, with the option to normalize reward predictions 
    during execution.
    """

    def __init__(
        self, state_size: int, output_size: int, action_size: int,
        hidden_size: Optional[int] = 64, num_hidden_layers: Optional[int] = 2,
        norm_type: Optional[str] = "ewma", norm_kwargs: Optional[Dict] = None
    ):
        """
        State corresponds to the state feature representation outputted by VPT.
        action_size could correspond to some embedding of the action (rather than 
        the raw action).
        """
        super().__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.action_mean, self.action_max = self.setup_action_normalizer()
        self.device = None
        
        self.input_size = state_size + action_size
        self.output_size = output_size
        self.norm_type = norm_type
        self.num_hidden_layers = num_hidden_layers

        self.mlp = MLP(self.input_size, num_hidden_layers, output_size, hidden_size,
                       nn.ReLU())

        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        self.normalizer = NormalizeEwma(output_size, **norm_kwargs)

    def reset_parameters(self):
        
        for layer_idx in range(self.num_hidden_layers + 1):
            
            layer = self.mlp.layers[layer_idx]
        
            init.orthogonal_(layer.weight)
            fan_in_linear(layer)
            
        self.normalizer.reset_parameters()

    def setup_action_normalizer(self):
        """
        Given size of action space, generates mean and maximum tensors
        to normalize actions during the forward pass
        """
        # max camera val = 10; all the others are binary buttons
        action_max = torch.Tensor([10.,10.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,])
        tile_size = int(self.action_size / len(action_max))
        action_max = action_max.tile((tile_size,))

        action_mean = action_max/2.  # TODO: should this be updated for each batch?

        return action_mean, action_max

    def normalize_actions(self, action):
        """
        Subtract the mean value for each action and divide by its maximum.
        """
        # check if max and mean tensors are already in the correct device
        if self.device is None:
            self.device = action.device
            self.action_max = self.action_max.to(self.device)
            self.action_mean = self.action_mean.to(self.device)
            
        return (action-self.action_mean)/self.action_max

    def forward(self, obs, action, normalize = False):

        # scale actions so they have magnitude similar to the obs ~[-0.5, 0.5]
        action = self.normalize_actions(action)

        obs_action = torch.cat([obs, action], dim=-1)

        Q_out = self.mlp(obs_action)

        if normalize:
            Q_out = self.normalizer(Q_out)

        return Q_out
    

    def denormalize(self, input_data):
        """Convert input value from a normalized space into the original one"""
        return self.normalizer.denormalize(input_data)

    def normalize(self, input_data):
        return self.normalizer(input_data)
