import numpy as np
from pickle import load
import copy

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from networks import *


class SAC(nn.Module):
    
    def __init__(self, config):
        super(SAC, self).__init__()
        self.name = 'sac'
        self.state_size = config.n_state
        self.action_size = config.n_action

        self.device = config.device
        
        self.gamma = 0.99
        self.tau = 1e-2
        hidden_size = 256
        learning_rate = 1e-4
        self.clip_grad_param = 1
        self.alpha_cql = 0.5

        self.target_entropy = -self.action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 
                
        # Actor Network 

        self.actor_local = Actor(self.state_size, self.action_size, hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = Critic(self.state_size, self.action_size, hidden_size, 2).to(self.device)
        self.critic2 = Critic(self.state_size, self.action_size, hidden_size, 1).to(self.device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(self.state_size, self.action_size, hidden_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(self.state_size, self.action_size, hidden_size).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 

    def get_action_prob(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action_prob = self.actor_local(state)
        return action_prob.detach().cpu().numpy()
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            action = self.actor_local.get_det_action(state)
        return action.numpy()
    
    def greedy_action(self, state):
        self.actor_local.eval()
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action = self.actor_local.get_argmax_action(state)
        return action.detach().cpu().numpy()


    def calc_policy_loss(self, states, alpha):
        self.actor_local.eval()
        self.critic1.eval()
        self.critic2.eval()
        _, action_probs, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states)   
        q2 = self.critic2(states)
        min_Q = torch.min(q1,q2)
        actor_loss = (action_probs * (alpha * log_pis - min_Q )).sum(1).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi
    
    def calc_q_value(self, states, action):
        self.critic1.eval()
        self.critic2.eval()

        state = torch.from_numpy(states).float().to(self.device)
        q1 = self.critic1(state)   
        q2 = self.critic2(state)
        min_Q = torch.min(q1,q2)
       
        q_value = torch.squeeze(min_Q, axis=0)[action]
    
        return q_value.detach().cpu().numpy()[0][0]


    def learn(self, samples, batches, cost_net, gamma, d=1):
        self.actor_local.train()
        self.critic1.train()
        self.critic2.train()
        cost_net.eval()

        states, actions, rewards, next_states, dones = batches
        fake_states, fake_actions, fake_rewards, fake_next_states, fake_dones = samples

        states_all = torch.cat([states, fake_states], dim=0)
        actions_all = torch.cat([actions, fake_actions], dim=0)
        next_states_all = torch.cat([next_states, fake_next_states], dim=0)
        dones_all = torch.cat([dones, fake_dones], dim=0)

        costs = cost_net(torch.cat((states_all, actions_all.reshape(-1, 1)), dim=-1))
        rewards_all = -costs


        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states_all, current_alpha.to(self.device))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_states_all)
            Q_target1_next = self.critic1_target(next_states_all)
            Q_target2_next = self.critic2_target(next_states_all)
            Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards_all + (gamma * (1 - dones_all) * Q_target_next.sum(dim=1).unsqueeze(-1)) 

        # Compute critic loss
        q1 = self.critic1(states_all).gather(1, actions_all.long())
        q2 = self.critic2(states_all).gather(1, actions_all.long())
        
        critic1_loss = 0.5 * F.mse_loss(q1, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2, Q_targets)

        ########################################
        # CQL equation
        ########################################
        q1_real = self.critic1(states).gather(1, actions.long())
        q2_real = self.critic2(states).gather(1, actions.long())
        q1_data_mean = q1_real.mean()
        q2_data_mean = q2_real.mean()

        # (2) fake 데이터 부분
        q1_fake = self.critic1(fake_states).gather(1, fake_actions.long())
        q2_fake = self.critic2(fake_states).gather(1, fake_actions.long())
        q1_fake_mean = q1_fake.mean()
        q2_fake_mean = q2_fake.mean()

        cql_loss_q1 = self.alpha_cql * (q1_fake_mean - q1_data_mean)
        cql_loss_q2 = self.alpha_cql * (q2_fake_mean - q2_data_mean)

        q1_loss = critic1_loss + cql_loss_q1
        q2_loss = critic2_loss + cql_loss_q2
    

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        q1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        q2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        # return actor_loss.item(), alpha_loss.item(), q1_loss.item(), q2_loss.item(), current_alpha

    def soft_update(self, local_model , target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
