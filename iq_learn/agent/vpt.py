import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW

import copy
from omegaconf import open_dict

sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./learning-from-human-preferences'))
from drlhp.pref_db import PrefDB
from vpt_agent.openai_vpt.lib.tree_util import tree_map
from vpt_agent.openai_vpt.lib.Q_head import QHead

from iq import iq_loss
from utils.utils import soft_update, hard_update, get_concat_samples


# Required methods for IQ-Learn agents
class MinecraftVPT(object):

    def __init__(self, args, policy, minecraft_agent):

        self.gamma = args.gamma
        self.online_batch_size = args.train.batch
        self.device = torch.device(args.device)
        self.args = args
        agent_cfg = args.agent
        reward_cfg = args.reward
        
        self.freeze_vpt_parameters = args.freeze_vpt_parameters
        self.unfreeze_last_vpt_layer = args.unfreeze_last_vpt_layer

        self.critic_tau = agent_cfg.critic_tau

        self.critic_target_update_frequency = agent_cfg.critic_target_update_frequency
        self.learn_temp = agent_cfg.learn_temp
        self.actor_update_frequency = agent_cfg.actor_update_frequency 

        self.log_alpha = torch.tensor(np.log(agent_cfg.init_temp)).to(self.device)
        self.log_alpha.requires_grad = True

        # setup VPT agent
        self.minecraft_agent = minecraft_agent
        self.policy = policy
        self.policy.set_device(self.device)

        self.human_prefs = args.human_prefs

        if self.human_prefs:
            with open_dict(args):
                args.num_action_tiles = policy.act_embedding.num_tiles
                args.action_emb_size = policy.action_emb_size

            self.preferences_manager = PreferencesManager(args, './preferences/train.pkl.gz',
                                                          self.policy, self.device)
            
            # Set preference learning constants:
            self.c = args.preferences.c
            self.delta = args.preferences.delta
        
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        self.target_entropy = -minecraft_agent.action_dim

        # Need to store target policy for computing KL loss term:
        self.kl_loss = args.kl_loss   # Whether we're using the KL divergence loss
        
        if self.kl_loss:
            self.kl_target_update_freq = args.kl_target_update_freq
            
            self.target_policy = copy.deepcopy(self.policy)
            self.target_policy.set_device(self.device)

            for param in self.target_policy.parameters():
                param.requires_grad = False
        
        self.bc_loss = args.bc_loss   # Whether we're using the BC loss
        
        # Bug fix: don't reset parameters after potentially having initialized them to
        # desired quantities.
        #self.policy.Q_head.reset_parameters()
        
        self.use_reward = args.use_reward
        
        #if self.use_reward:
            
        #    self.policy.reward_head.reset_parameters()

        # Determine which policy parameters are supposed to have requires_grad = False,
        # so that we can keep them that way:
        frozen_param_names = []
            
        for name, param in self.policy.named_parameters():
            if not param.requires_grad:
                frozen_param_names.append(name)            

        # To store trainable parameters for the policy and Q-head:
        trainable_parameters_policy = []
        trainable_parameters_Q = []

        self.agent_state_target = False    # Used if calculating KL term

        if self.freeze_vpt_parameters:  # Freeze all params except for the policy head and Q head
                
            for param in self.policy.parameters():
                param.requires_grad = False
                
            if self.unfreeze_last_vpt_layer:   # Unfreeze final VPT layer

                for name, param in policy.net.lastlayer.named_parameters():
                    if 'net.lastlayer.' + name not in frozen_param_names:
                        param.requires_grad = True
                        trainable_parameters_policy.append(param)
                        trainable_parameters_Q.append(param)
                        
                self.agent_state_target = True
                
        else:   # Unfreeze everything
            
            for name, param in self.policy.net.named_parameters():
                if 'net.' + name not in frozen_param_names:
                    param.requires_grad = True
                    trainable_parameters_policy.append(param)  
                    trainable_parameters_Q.append(param)
                    
            self.agent_state_target = True
            
        for name, param in self.policy.pi_head.named_parameters():
            if 'pi_head.' + name not in frozen_param_names:
                param.requires_grad = True
                trainable_parameters_policy.append(param)
        for name, param in self.policy.Q_head.named_parameters():
            if 'Q_head.' + name not in frozen_param_names:
                param.requires_grad = True
                trainable_parameters_Q.append(param)

        if self.use_reward:
            
            trainable_parameters_reward = []
            
            for name, param in self.policy.reward_head.named_parameters():
                if 'reward_head.' + name not in frozen_param_names:
                    param.requires_grad = True
                    trainable_parameters_reward.append(param)
             
        if not args.offline:      # Online + expert batch sizes
            total_batch_size = self.online_batch_size + args.data_loader.batch_size
        else:     # Just expert batch size
            total_batch_size = args.data_loader.batch_size
        
        # Define actor and critic based on VPT agent policy heads:
        if not self.kl_loss:
            self.actor_critic = VPT_actor_critic_wrapper(self.policy, self.device, batch_size = total_batch_size)
        else:
            self.actor_critic = VPT_actor_critic_wrapper(self.policy, self.device, self.target_policy,
                                                         self.agent_state_target, batch_size = total_batch_size)
        
        # Define critic target: this is another Q head initialized to the
        # same weights as the one in the policy.
        self.critic_target = VPT_critic_target(self.policy, self.device)
        
        # Initialize critic target Q to critic Q:
        hard_update(self.critic_net, self.critic_target_net)
        
        # optimizers      
        self.actor_optimizer = Adam(trainable_parameters_policy,
                                    lr=agent_cfg.actor_lr,
                                    betas=agent_cfg.actor_betas)
        self.critic_optimizer = Adam(trainable_parameters_Q,
                                      lr=agent_cfg.critic_lr,
                                      betas=agent_cfg.critic_betas)
        self.log_alpha_optimizer = Adam([self.log_alpha],
                                        lr=agent_cfg.alpha_lr,
                                        betas=agent_cfg.alpha_betas)

        if self.use_reward:
            self.reward_optimizer = AdamW(trainable_parameters_reward,
                                          lr = reward_cfg.reward_lr,
                                          betas = reward_cfg.reward_betas,
                                          weight_decay = reward_cfg.weight_decay)

        self.train()
 

    def train(self, training=True):
        # TODO: does the VPT model also have a train/eval setting that we should
        # modify here?
        self.training = training

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def critic_net(self):
        return self.policy.Q_head

    @property
    def critic_target_net(self):
        return self.critic_target.Q_head_target

    def getV(self, obs, episode_id, update_agent_state = False):

        _, log_prob, current_Q, _, _, _ = \
            self.actor_critic.sample(obs, episode_id, update_agent_state = update_agent_state)

        current_V = current_Q - self.alpha.detach() * log_prob
        return current_V


    def get_targetV(self, obs, episode_id, update_agent_state = False):
        
        _, log_prob, _, q_h, ac_emb, _ = \
            self.actor_critic.sample(obs, episode_id, update_agent_state = update_agent_state) 
            
        current_Q = self.critic_target(q_h, ac_emb)
        
        current_V = current_Q - self.alpha.detach() * log_prob
        return current_V
    

    def update_actor_and_alpha(self, batch, action_policy_rep, logger, step, update_agent_state = False):
        
        obs, _, _, episode_id, _, _, is_expert = batch
        
        _, log_prob, actor_Q, _, _, pd_current = self.actor_critic.sample(obs, episode_id, 
                                    update_agent_state = update_agent_state, stochastic = True)
        
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train/actor_Q_loss', actor_loss, step)
        logger.log('train/target_entropy', self.target_entropy, step)
        logger.log('train/actor_entropy', -log_prob.mean(), step)

        # Calculate KL divergence loss:
        if self.kl_loss:
            
            # First, need to get all action distributions corresponding to the 
            # target policy:
            with torch.no_grad():
                pd_target, _, _, _ = self.actor_critic(obs, episode_id, eval_target = True,
                                                update_agent_state = True, eval_policy_head = True)
   
            kl_losses = self.actor_critic.vpt_policy.get_kl_of_action_dists(pd_current, pd_target)
                
            kl_loss = torch.mean(kl_losses)
            actor_loss += self.args.kl_loss_weight * kl_loss
            
            logger.log('train/actor_kl_loss', kl_loss, step)

        # Calculate the BC loss:
        if self.bc_loss:
            
            pd_expert = {'buttons': pd_current['buttons'][is_expert], 
                         'camera': pd_current['camera'][is_expert]}
            
            log_prob = self.policy.get_logprob_of_action(pd_expert, action_policy_rep)
            bc_loss = -torch.mean(log_prob)

            actor_loss += self.args.bc_loss_weight * bc_loss
            
            logger.log('train/actor_bc_loss', bc_loss, step)
            
        # optimize the actor
        
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()

        losses = {
            'loss/actor': actor_loss.item(),
            'actor_loss/target_entropy': self.target_entropy,
            'actor_loss/entropy': -log_prob.mean().item()}

        if self.learn_temp:
            
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train/alpha_loss', alpha_loss, step)
            logger.log('train/alpha_value', self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            self.log_alpha_optimizer.zero_grad()
            
            losses.update({
                'alpha_loss/loss': alpha_loss.item(),
                'alpha_loss/value': self.alpha.item(),
            })
            
        return losses
        

    def update_critic(self, batch, logger, step):
        args = self.args
    
        obs, next_obs, action, ep_id, _, _, _ = batch
    
        agent = self
    
        current_V = self.getV(obs, ep_id, update_agent_state = False)
    
        if "DoubleQ" in args.q_net._target_:
            
            raise NotImplementedError('Option for 2 Q heads has not been added to VPT model yet.')
            
            # Code that goes here in original iq-learn implementation:
            """current_Q1, current_Q2 = self.critic(obs, action, both=True)
            """
        else:
            
            current_Q, obs_emb = self.actor_critic.critic(obs, action, ep_id, update_agent_state = True)
    
        if args.train.use_target:
            with torch.no_grad():
                next_V = self.get_targetV(next_obs, ep_id, update_agent_state = False)
        else:
            next_V = self.getV(next_obs, ep_id, update_agent_state = False)
    
        done = batch[5]
        next_V = (1 - done) * next_V
    
        if "DoubleQ" in args.q_net._target_:
            
            raise NotImplementedError('Option for 2 Q heads has not been added to VPT model yet.')
            
            # Code that goes here in original iq-learn implementation:
            """q1_loss, loss_dict1 = iq_loss(agent, current_Q1, current_V, next_V, batch)
            q2_loss, loss_dict2 = iq_loss(agent, current_Q2, current_V, next_V, batch)
            critic_loss = 1/2 * (q1_loss + q2_loss)
            # merge loss dicts
            loss_dict = average_dicts(loss_dict1, loss_dict2)"""
        else:
    
            critic_loss, loss_dict, reward_training_targets = \
                iq_loss(agent, current_Q, current_V, next_V, batch)
    
    
        if self.use_reward and args.feed_learned_reward_into_Q:     # Add loss term based on the learned reward
        
            with torch.no_grad():   # Calculate reward loss targets
        
                # Calculate rewards from learned reward model for all items in the batch:
                reward_pred = self.eval_reward(obs_emb, action)
                
                # Calculate targets for reward-based Q loss term:
                reward_loss_targets = reward_pred + self.gamma * next_V
            
            # Supervised learning loss between predictions and targets:
            critic_reward_loss = F.mse_loss(current_Q, reward_loss_targets)
           
            logger.log('train/critic_reward_loss', critic_reward_loss, step)
            logger.log('train/critic_iq_loss', critic_loss, step)
            
            loss_dict['reward_Qloss'] = critic_reward_loss.item()
            
            critic_loss += critic_reward_loss
        
        logger.log('train/critic_loss', critic_loss, step)
        
    
        # Optimize the critic
        critic_loss.backward()
        # step critic
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()
        
        return loss_dict, reward_training_targets.detach(), obs_emb.detach()
    
    
    
    def iq_update(self, policy_batch, expert_batch, logger, step):
    
        args = self.args    
    
        losses = {}
        
        combined_batch = self.get_combined_batch(policy_batch, expert_batch[:-1])
        
        # We only have the policy representation of the action for the expert data,
        # not the online data:
        action_policy_rep = expert_batch[-1]
    
        # Need to update actor before critic due to VPT internal state updating.
        if step % self.actor_update_frequency == 0:

            for i in range(args.num_actor_updates):
                actor_alpha_losses = self.update_actor_and_alpha(combined_batch, action_policy_rep,
                                                                 logger, step, 
                                                                 update_agent_state = False)

            losses.update(actor_alpha_losses)
    
        critic_losses, reward_iq_targets, obs_emb = self.update_critic(combined_batch, logger, step)
        losses.update(critic_losses)
        
        if args.use_reward:
            reward_losses = self.reward_update(reward_iq_targets, obs_emb, combined_batch[2], logger, step)
            losses.update(reward_losses)
    
        # Update critic target network:
        if step % self.critic_target_update_frequency == 0:
            if args.train.soft_update:
                soft_update(self.critic_net, self.critic_target_net,
                            self.critic_tau)
            else:
                hard_update(self.critic_net, self.critic_target_net)
        
        # Update policy target network: this is the target network used for the
        # KL divergence regularizer.
        if self.kl_loss and step % self.kl_target_update_freq == 0:
            
            hard_update(self.policy.pi_head, self.target_policy.pi_head)
            hard_update(self.policy.net, self.target_policy.net)
        
        
        return losses

    
    def reward_update(self, reward_iq_targets, obs_emb, action, logger, step):     
        """
        Reward head gradient update.
        Note that for the IQ-learn term, some of the quantities are already
        calculated in the IQ-learn gradient updates. So we pass those in as 
        input arguments rather than recalculating them here.
        """
        
        losses = {}
        
        if self.args.reward.iq_term:
            
            # Evaluate reward head for the observation embedding and action:
            reward_pred = self.eval_reward(obs_emb, action)
        
            # Get supervised learning loss between reward predictions and targets:
            reward_loss = F.mse_loss(reward_pred, reward_iq_targets)
            
            logger.log('train/reward_iq_loss', reward_loss, step)
            losses['reward_iq_loss'] = reward_loss
            
        else:
            
            reward_loss = torch.tensor(0.0, requires_grad=True)
        
        # check if new preferences are saved in disk
        if self.human_prefs:
            
            if self.preferences_manager.have_new_preferences():
                # load new prefs
                self.preferences_manager.load()

            # only proceed if we have enough preferences saved
            if self.preferences_manager.has_enough_prefs():
                
                # Get predicted rewards for next preference batch from preferences_manager:
                rewards_total, preferences = self.preferences_manager.get_batch_reward_predictions()
                
                # Compute bradley-terry loss
                bt_loss = self.compute_bt_loss(rewards_total, preferences)
                print(f"bt_loss: {bt_loss}")

                # combine reward iq and bradley-terry losses
                reward_loss = reward_loss + bt_loss
                
                logger.log('train/reward_bt_loss', bt_loss, step)
                losses['bt_loss'] = bt_loss
                
        # Optimize the reward
        reward_loss.backward()
        # step critic
        self.reward_optimizer.step()
        self.reward_optimizer.zero_grad()
        
        return losses

    def compute_bt_loss(self, reward_total, preferences):
        '''
        Bradley-Terry Loss with "no preference" option

        Takes as input a (<num_trajs>, 1) tensor with reward values from the trajectories the user
        evaluated. The first half of the tensor are rewards from the left trajectories, the last
        half is from the right trajectories.

        The preferences list should have length <num_trajs>/2 and will have the tuples with the
        human preferences:
            (1.0, 0.0) means the user preferred the left traj
            (0.0, 1.0) means the user preferred the right traj
            (0.5, 0.5) means the user evalated them as equal

        c is a term that quantifies how certain the user was giving the preference. Using 1 for now.

        '''
        print(f"Computing BRADLEY-TERRY loss for {len(preferences)} preferences...")
       
        c = self.c
        delta = self.delta
       
        num_prefs = len(preferences)
        reward_differences = reward_total[:num_prefs] - reward_total[num_prefs:]
       
        p_left_best = 1/(1+torch.exp(delta - c*reward_differences)) + 1e-9
        p_right_best = 1/(1+torch.exp(delta + c*reward_differences)) + 1e-9
       
        # We add 3e-9 because subtracting p_left_best and p_right_best subtracts 2e-9
        p_no_pref = 1 - p_left_best - p_right_best + 3e-9
   
        pref_indicators = torch.tensor(preferences, device=self.device)[:, 0].unsqueeze(1)
        bt_loss = torch.log(p_left_best) * (pref_indicators == 1) + \
                    torch.log(p_right_best) * (pref_indicators == 0) \
                    + torch.log(p_no_pref) * (pref_indicators == 0.5)
       
        return -bt_loss.sum()

    def eval_reward(self, obs_emb, action):
        
        return self.policy.reward_head(obs_emb, action).squeeze(1)

 
    def get_combined_batch(self, policy_batch, expert_batch):
        
        args = self.args
        
        if args.only_expert_states:
            
            assert not args.offline, 'No access to policy actions in expert mode.'
    
            _, _, policy_action, _, _, _ = policy_batch
            expert_obs, expert_next_obs, _, expert_ep_id, expert_reward, expert_done = expert_batch
            
            # Use policy actions instead of experts actions for IL with only observations
            expert_batch = expert_obs, expert_next_obs, policy_action, expert_ep_id, expert_reward, expert_done
    
        if not args.offline:
            batch = get_concat_samples(policy_batch, expert_batch, args)
        else:
            # get_concat_samples (used in online case) adds is_expert to the batch.
            # So we need to add that here:
            is_expert = torch.ones(expert_batch[4].shape, dtype=torch.bool)
            batch = expert_batch + (is_expert,)
            
        return batch
   
    
    # Save model parameters
    def save(self, path, suffix=""):
        
        state_dict = self.actor_critic.vpt_policy.state_dict()
        
        output_path = f"{path}{suffix}.weights"
        print('Saving models to {}'.format(output_path))
        torch.save(state_dict, output_path)



class VPT_actor_critic_wrapper(torch.nn.Module):

    def __init__(self, vpt_policy, device, vpt_policy_target = None,
                 agent_state_target = False, batch_size = 1):
        
        super().__init__()
        
        self.vpt_policy = vpt_policy
        self.vpt_policy_target = vpt_policy_target
        
        self.device = device
        
        self.first = torch.ones(batch_size, 1, dtype = bool).to(device)
        self.prev_episode_id = None
        
        self.batch_size = batch_size
        
        self.episode_hidden_states = self.vpt_policy.initial_state(self.batch_size)
        
        self.agent_state_target = agent_state_target
        
        if agent_state_target:
            
            assert vpt_policy_target is not None, "Must provide target policy to use target policy agent states"
            
            self.episode_hidden_states_target = self.vpt_policy.initial_state(self.batch_size)
        
        
    def forward(self, obs, episode_id, action = None, update_agent_state = False,
                eval_target = False, eval_policy_head = True):
        
        # Set whether to use target or current policy:
        if not eval_target or self.vpt_policy_target is None:  
            policy = self.vpt_policy
        else:
            policy = self.vpt_policy_target
        
        # Whether to use hidden states from current or target VPT model
        target_hidden_state = eval_target and self.agent_state_target
        
        if not target_hidden_state:  # Use hidden states from current VPT model
            episode_hidden_states = self.episode_hidden_states        
    
        else:    # Use hidden states from target VPT model
            episode_hidden_states = self.episode_hidden_states_target 

        # first should be a batch_size x 1 Boolean tensor indicating whether
        # each batch element corresponds to the start of an episode.        
        first = self.get_first(episode_id)

        (pi_logits, q_prediction, _), new_agent_state, pi_h, q_h = \
            policy(obs, first, episode_hidden_states, action, eval_policy_head)            
        
        if update_agent_state:

            # Make sure we do not try to backprop through sequence
            # (fails with current accumulation)
            new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
            
            if not eval_target:
                self.episode_hidden_states = new_agent_state
                
                self.prev_episode_id = episode_id
            elif target_hidden_state:
                self.episode_hidden_states_target = new_agent_state

        return pi_logits, q_prediction, pi_h, q_h

    def actor(self, obs, episode_id, update_agent_state = False):
        
        pi_logits, _, _, _ = self(obs, episode_id, update_agent_state = update_agent_state)
        return pi_logits
    
    def critic(self, obs, action, episode_id, update_agent_state = False):
        
        _, q_pred, _, obs_emb = self(obs, episode_id, action = action, 
                               update_agent_state = update_agent_state,
                               eval_policy_head = False)
        return q_pred, obs_emb
    
    
    def sample(self, obs, episode_id, update_agent_state = False, stochastic: bool = True):

        # first should be a batch_size x 1 Boolean tensor indicating whether
        # each batch element corresponds to the start of an episode.        
        first = self.get_first(episode_id)    

        ac, new_agent_state, log_prob, Qpred, _, q_h, ac_emb, pd = \
            self.vpt_policy.sample(obs, first, self.episode_hidden_states)        

        if update_agent_state:
        
            # Make sure we do not try to backprop through sequence
            # (fails with current accumulation)
            new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
            
            self.episode_hidden_states = new_agent_state
            
            self.prev_episode_id = episode_id

        return ac, log_prob, Qpred, q_h, ac_emb, pd


    def get_first(self, episode_id):
        """ first should be a batch_size x 1 Boolean tensor indicating whether
            each batch element corresponds to the start of an episode.
        """
        
        if self.prev_episode_id is None:
            first = self.first    # Initial value at start
        else:
            
            # Get indices of current episode ids that are not in the previous episode
            # ids list. These are assumed to be newly-beginning episodes.
            # We also assume that each episode_id in a given batch is unique.
            first = np.in1d(episode_id, self.prev_episode_id)
            first = torch.from_numpy(np.reshape(first, (-1, 1))).to(self.device) 
            
        return first



class VPT_critic_target(torch.nn.Module):

    def __init__(self, vpt_policy, device):
        
        super().__init__()

        self.device = device        

        Q_head_critic = vpt_policy.Q_head
        self.Q_head_critic = Q_head_critic

        # Initialize critic target Q with the same network parameters as the 
        # critic Q head.
        self.Q_head_target = QHead(Q_head_critic.state_size, Q_head_critic.output_size,
                            Q_head_critic.action_size, Q_head_critic.hidden_size,
                            Q_head_critic.num_hidden_layers).to(device)
        
        # Bug fix: initialize Q-target weights to critic Q-head
        hard_update(self.Q_head_critic, self.Q_head_target)
        #self.Q_head_target.reset_parameters()

    
    def forward(self, q_h, action):
        """
        q_h is observation embedding from VPT network.
        """

        Qpred = self.Q_head_target(q_h, action).squeeze(2)
        return Qpred
        
        
class TrajectoryRewardEvaluator(torch.nn.Module):
    """
    Class for evaluating rewards of entire trajectories. Current intended use:
    calculate rewards for trajectories pairwise preference, and keep track of
    episode hidden states.
    """
    
    def __init__(self, vpt_policy, device):
        super().__init__()
        
        self.vpt_policy = vpt_policy

        self.device = device
        
    def reset(self, batch_size, t):
        """ 
        Reset hidden state information.
        
        first should be a batch_size x 1 Boolean tensor indicating whether
            each batch element corresponds to the start of an episode.
        """        
        self.episode_hidden_states = self.vpt_policy.initial_state(batch_size)
 
        self.first = torch.ones(batch_size, t, dtype = bool).to(self.device)   
        self.batch_size = batch_size
        self.t = t
 
    def forward(self, obs, action = None, calc_obs_embeddings = True, obs_embedding_only = False):    
        """
        Evaluate rewards for the given observations and actions.
        """
        
        if calc_obs_embeddings:
        
            with torch.no_grad():    
        
                _, new_agent_state, _, obs_emb = \
                    self.vpt_policy(obs, self.first, self.episode_hidden_states,
                                    eval_policy_head = False)
                
                # Update hidden state info.            
                self.episode_hidden_states = new_agent_state
                self.first = torch.zeros(self.batch_size, self.t, dtype = bool).to(self.device)   
            
        else:   # In this case, observations are already given in embedded form.
            
            obs_emb = obs

        # Evaluate reward head for the given observations and actions:
        if not obs_embedding_only:
            reward_pred = self.vpt_policy.reward_head(obs_emb, action)
        else:
            reward_pred = None

        return reward_pred, obs_emb
        
 
    def get_total_trajectory_reward(self, observations, actions = None, b_chunk = None, t_chunk = None,
                                    calc_obs_embeddings = True, obs_embedding_only = False):
        """
        Evaluate total reward prediction for a set of trajectories.
        
        observations: b x t x 128 x 128 x 3 if calc_obs_embeddings is True.
                      b x t x obs_embedding_dim if calc_obs_embeddings is False.
        actions: b x t x action_embedding_dimension
        
        b_chunk: maximum batch size (batching over trajectories) to put through a single model forward pass.
        t_chunk: maximum batch size (batching over time) to put through a single model forward pass.
                 t_chunk is required to be divisible by t.

        calc_obs_embeddings (bool): set to True if VPT embeddings need to
        be calculated for all observations. This can be set to False if all the 
        VPT model weights are frozen (such that the observation embeddings never
        change), and if we already have calculated the observations in embedded form.
                 
        obs_embedding_only (bool): if True, just return observation embeddings and
                                   don't calculate the reward,
        
        Outputs:
            1) total reward prediction: b x 1 tensor (we sum over the t dimension).
            2) observation embeddings: b x t x observation_embedding_dim
        """
        b, t = observations.shape[:2]
        print(f"COMPUTING REWARDS OF {b} SEGMENTS WITH {t} LENGTH...")
        
        if b_chunk is None:
            b_chunk = b
        if t_chunk is None:
            t_chunk = t
        
        assert(np.mod(t, t_chunk) == 0), 'Trajectory length must be divisible by t_chunk'
        
        if not obs_embedding_only:
            assert(actions is not None), 'Must supply actions if obs_embedding_only = False'
        
        # Get start and end of each "chunk" to pass though the model.
        b_starts = np.arange(0, b, b_chunk)
        t_starts = np.arange(0, t, t_chunk)

        b_ends = np.concatenate((b_starts[1:], [b]))
        t_ends = np.concatenate((t_starts[1:], [t]))

        reward_total = torch.zeros((b, 1), device = self.device)   # To store reward info
        
        # To store all observation embeddings. This is useful if the VPT model 
        # weights are frozen, in which case these can be re-used in future iterations.
        obs_emb_all = torch.empty((b, t, self.vpt_policy.output_latent_size), requires_grad=False, device = self.device)

        for b_start, b_end in zip(b_starts, b_ends):

            if calc_obs_embeddings:
                self.reset(b_end - b_start, t_chunk)   # Reset episode hidden state info
            
            for t_start, t_end in zip(t_starts, t_ends):
                
                if not obs_embedding_only:
                
                    rewards, obs_emb = self(observations[b_start:b_end, t_start:t_end, :],
                                   actions[b_start:b_end, t_start:t_end, :],
                                   calc_obs_embeddings = calc_obs_embeddings,
                                   obs_embedding_only = obs_embedding_only)
    
                    reward_total[b_start: b_end] += torch.sum(rewards, axis = 1)
                    
                    obs_emb_all[b_start:b_end, t_start:t_end, :] = obs_emb
                    
                else:

                    _, obs_emb = self(observations[b_start:b_end, t_start:t_end, :],
                                   calc_obs_embeddings = calc_obs_embeddings,
                                   obs_embedding_only = obs_embedding_only)
                    
                    obs_emb_all[b_start:b_end, t_start:t_end, :] = obs_emb                    

        return reward_total, obs_emb_all 
 
    
class PreferencesManager():   
    
    def __init__(self, args, prefs_path, policy, device):
        
        self.prefs_path = prefs_path
        
        self.prefs_file_size = 0

        self.args = args
        
        self.device = device

        # preference database used to get prefs saved on disk
        self.pref_db = PrefDB(maxlen=1000, args=args)
        
        self.max_prefs_batch = args.preferences.max_prefs_batch
        self.min_prefs_batch = args.preferences.min_prefs_batch
        
        self.num_prefs = 0
        
        # Determine whether or not to store observation embeddings:
        if args.freeze_vpt_parameters and not args.unfreeze_last_vpt_layer:
            self.store_obs_embeddings = True
        else:
            self.store_obs_embeddings = False
            
        # Length of segments:
        self.segment_len = args.preferences.trajectory_length
        
        # For calculating segment rewards:
        self.trajectory_reward_evaluator = TrajectoryRewardEvaluator(policy, device)
        
        # Size of observation and action embeddings:
        self.act_emb_size = policy.action_emb_size
        self.obs_embedding_dim = policy.output_latent_size
        
        if self.store_obs_embeddings:
            
            self.obs_embeddings = {}
            self.prev_segment_ids_unique = np.empty(0)
        

    def load(self):
        """
        Load new segments, preferences, etc.
        
        If we're storing observation embeddings for each segment, then we:
            a) Check for new segments, and calculate their observation embeddings.
            b) Check for any obselete segments and remove them.
        """
        # Load preferences file:
        self.pref_db = self.pref_db.load(path=self.prefs_path)
        self.num_prefs = len(self.pref_db.prefs)
        
        # Do observation embedding processing.
        if self.store_obs_embeddings:
            
            prefs = self.pref_db.prefs
            observations = self.pref_db.segments_fmt

            # Get all unique segment ids in the preference data:
            segment_ids = np.array([prefs[idx][:2] for idx in range(self.num_prefs)])
            segment_ids_unique = np.unique(segment_ids)

            """Process any new segment ids:"""
            new_segment_ids = np.setdiff1d(segment_ids_unique, self.prev_segment_ids_unique)
            num_new_segments = len(new_segment_ids)
            
            if num_new_segments > 0:
            
                # Gather all new observations.
                obs_new = torch.empty((num_new_segments, self.segment_len, 128, 128, 3), device = self.device)
                
                for i, new_segment_id in enumerate(new_segment_ids):
                
                    obs_new[i] = observations[new_segment_id]
    
                # Calculate embeddings of new observations:
                _, obs_embedding = self.trajectory_reward_evaluator.get_total_trajectory_reward(obs_new,
                                                                           obs_embedding_only = True)
                # Store new observation embeddings in our dictionary:
                for i, new_segment_id in enumerate(new_segment_ids):
                    
                    self.obs_embeddings[new_segment_id] = obs_embedding[i]
            
            """Remove any segments we're finished with:"""
            finished_segment_ids = np.setdiff1d(self.prev_segment_ids_unique, segment_ids_unique)
            
            for segment_id in finished_segment_ids:
                
                segment_embedding = self.obs_embeddings.pop(segment_id)
                del segment_embedding
            
            """Update list of unique segment ids:"""
            self.prev_segment_ids_unique = segment_ids_unique
        

    def get_prefs_file_size(self):
        '''
        Gets the file size of the human preferences saved on disk.
        '''
        try:
            prefs_file_size = os.stat(self.prefs_path).st_size
        except:
            # file might not exist
            prefs_file_size = 0
        return prefs_file_size

    def have_new_preferences(self):
        '''
        Check file size to detect if there are new human preferences saved on disk.
        '''
        curr_prefs_file_size = self.get_prefs_file_size()
        if curr_prefs_file_size != self.prefs_file_size:
            self.prefs_file_size = curr_prefs_file_size
            return True
        else:
            return False

    def has_enough_prefs(self):
        
        return self.num_prefs >= self.min_prefs_batch
        
    def get_batch_reward_predictions(self):
        """
        Sample a batch of preferences from self.pref_db and return the predicted
        total reward for each.
        
        Returns:
            1) Predicted total rewards: tensor of length batch_size
            2) Preferences: list of length batch_size, in which each element 
               is either (1, 0), (0, 1), or (0.5, 0.5). These respectively indicate
               a left preference, a right preference, and "no preference".
        """
        
        # Get data from pref_db:
        prefs = self.pref_db.prefs
        observations = self.pref_db.segments_fmt
        actions_emb = self.pref_db.actions_fmt
        
        # Determine which preferences will go in this batch:
        if self.num_prefs < self.max_prefs_batch:   # Use all preferences
            batch_size = self.num_prefs
            
            batch_pref_idxs = np.arange(self.num_prefs)
            
        else:      # Randomly sample idxs of preferences in this batch
            batch_size = self.max_prefs_batch
        
            batch_pref_idxs = np.random.choice(np.arange(self.num_prefs), batch_size, replace = False)
        
        # Get all segment ids corresponding to this batch:
        segment_ids_batch = np.array([prefs[idx][:2] for idx in batch_pref_idxs])
        segment_ids_unique = np.unique(segment_ids_batch)
        num_unique_segments = len(segment_ids_unique)
        
        # Stack observations and actions belonging to this batch:
        if self.store_obs_embeddings:
            obs_unique = torch.empty((num_unique_segments, self.segment_len, self.obs_embedding_dim), device = self.device)
        else:
            obs_unique = torch.empty((num_unique_segments, self.segment_len, 128, 128, 3), device = self.device)
            
        act_unique = torch.empty((num_unique_segments, self.segment_len, self.act_emb_size), device = self.device)
        
        for i, segment_id in enumerate(segment_ids_unique):
        
            if self.store_obs_embeddings:
                obs_unique[i] = self.obs_embeddings[segment_id]
            
            else:
                obs_unique[i] = observations[segment_id]
                
            act_unique[i] = actions_emb[segment_id]
        
        # Get reward predictions for all segments in the batch:
        rewards_unique, _ = self.trajectory_reward_evaluator.get_total_trajectory_reward(obs_unique, act_unique, 
                                            calc_obs_embeddings = not self.store_obs_embeddings)
            
        # Adjust rewards to expanded format, in which the first half of the tensor are rewards 
        # from the left trajectories and the last half is from the right trajectories.    
        # NOTE: having num_prefs instead of batch_size here was an obvious bug,
        # as it can result in run-time errors.
        rewards_expanded = torch.empty(2 * batch_size, device = self.device)
        
        # Segment ID corresponding to each index in rewards_expanded:
        segment_ids_batch = segment_ids_batch.transpose().flatten()

        # NOTE: having num_prefs instead of batch_size here was an obvious bug,
        # as it can result in run-time errors.        
        for i in range(2 * batch_size):
            
            segment_idx_unique = np.nonzero(segment_ids_batch[i] == segment_ids_unique)[0]
            
            rewards_expanded[i] = rewards_unique[segment_idx_unique]
        
        # Format preferences into a list of length batch_size, in which each element 
        # is (1, 0), (0, 1), or (0.5, 0.5).
        prefs_batch = [prefs[idx][-1] for idx in batch_pref_idxs]
        
        return rewards_expanded, prefs_batch
        