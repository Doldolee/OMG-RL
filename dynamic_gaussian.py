# implement model to learn state transitions and rewards
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import warnings
import random
from pickle import load
import numpy as np

from buffer import BufferBatch
from torch.utils.data import DataLoader, TensorDataset

_device_ = 'cuda'

class MLPNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(MLPNetwork, self).__init__()

        self.state_dim = state_dim
        self.in_dim = state_dim + action_dim
        self.out_dim = 2 * (state_dim + 1)
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim)
        )
   
        # self.mean_head = nn.Linear(hidden_dim, self.out_dim)
        # self.log_std_head = nn.Linear(hidden_dim, self.out_dim)  

        # self.max_logvar = nn.Parameter((torch.ones((1, self.state_dim +1)).float() / 2).to(_device_), requires_grad=True)
        # self.min_logvar = nn.Parameter((-torch.ones((1, self.state_dim + 1)).float() * 10).to(_device_), requires_grad=True)
        # self.register_parameter("max_logvar", self.max_logvar)
        # self.register_parameter("min_logvar", self.min_logvar)

    def forward(self, input):
        out = self.net(input)
        s_dim_plus_one = self.state_dim + 1
        mean = out[:, :s_dim_plus_one]
        log_var = out[:, s_dim_plus_one:]

        # return mean, log_var

        # mean = self.mean_head(x)
        # logvar = self.log_std_head(x)
        # log_var = self.max_logvar - F.softplus(self.max_logvar - log_var)
        # log_var = self.min_logvar + F.softplus(log_var - self.min_logvar)

        log_var = torch.clamp(log_var, min=-5)

        return mean, log_var

    # decay loss?

class GaussianDynamic:
    def __init__(self, net, epochs, optimizer):
        self.net = net
        self.epochs = epochs

        self.optimizer = optimizer

    def train(self, loader):
        for epoch in range(self.epochs):
            self.net.train()
            loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(loader):
                state_batch, action_batch = inputs[:,:STATE_DIM], inputs[:,STATE_DIM:]
                next_state_batch, reward_batch = targets[:,:STATE_DIM], targets[:,STATE_DIM:]
                state_batch = state_batch.to(_device_)
                action_batch = action_batch.to(_device_)
                next_state_batch = next_state_batch.to(_device_)
                reward_batch = reward_batch.to(_device_)

                # delta_state_batch = next_state_batch - state_batch

                self.optimizer.zero_grad()

                model_input = torch.cat([state_batch, action_batch], dim=-1)
                mean, logvar = self.net(model_input)
                gt = torch.cat((next_state_batch, reward_batch), dim=-1)
                gaussian_loss = self.loss(mean, logvar, gt)

                gaussian_loss.backward()
                self.optimizer.step()
                
                # mse_loss, var_loss = self.loss(mean, logvar, gt)
                # mse_loss = torch.sum(mse_loss)
                # var_loss = torch.sum(var_loss)

                # transition_loss = mse_loss + var_loss
                # transition_loss += 0.01 * torch.sum(self.net.max_logvar) - 0.01 * torch.sum(self.net.min_logvar)


                loss += gaussian_loss.item()

            epoch_loss = loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}")
            

    def loss(self, mean, logvar, gt):

        loss = nn.GaussianNLLLoss()
        var = torch.exp(logvar)
        output = loss(mean, gt, var)

        return output
    
    def step(self, inputs, steps):

        terminated, truncated = False, False

        inputs = torch.from_numpy(inputs).float().to(_device_)
        mean, logvar = self.net(inputs)
        var = torch.exp(logvar)

        distribution = torch.distributions.Normal(mean, torch.sqrt(var))
        sampled = distribution.rsample() 

        next_state = sampled[:, :-1].squeeze(0)  # shape: [s_dim]
        reward = sampled[:, -1].squeeze(0)       # shape: []
        
        if reward[0][0] > 1:
            # reward[0][0] = 1
            print(f"model reward range upper 1: {reward[0][0]}")
        if reward[0][0] < -1:
            # reward[0][0] = -1
            print(f"model reward range lower -1: {reward[0][0]}")

        ## truncated rule
        if steps > 100:
            terminated = True
            truncated = True
    
        return reward[0][0].detach().cpu().numpy(), next_state[0].detach().cpu().numpy(), terminated, truncated



if __name__=='__main__':
    EPOCHS = 50
  
    STATE_DIM = 16
    ACTION_DIM = 1
    REWARD_DIM = 1
    NEXT_STATE_DIM = 16
    OUT_DIM = NEXT_STATE_DIM + REWARD_DIM

    HIDDEN_DIM = 64
    BATCH_SIZE = 64
    data_path = "./data/before_buffer/"

    # scaling completed
    train_replay_memory = BufferBatch(state_size =STATE_DIM)
    train_replay_memory.load(data_path + 'train')
    print("train memory data size:",train_replay_memory.crt_size)   
    test_replay_memory = BufferBatch(state_size =STATE_DIM)
    test_replay_memory.load(data_path + 'test')
    print("test memory data size:",test_replay_memory.crt_size)

    train_state, train_action, train_next_state, train_reward, train_done = train_replay_memory.random_sample_all_batch()
    print(train_state.shape, train_action.shape, train_next_state.shape, train_reward.shape, train_done.shape)
    test_state, test_action, test_next_state, test_reward, test_done = test_replay_memory.random_sample_all_batch()
    print(test_state.shape, test_action.shape, test_next_state.shape, test_reward.shape, test_done.shape)

    action_scaler = load(open("./data/scaler/action_scaler.pkl",'rb'))
    train_action = action_scaler.transform(train_action)
    test_action = action_scaler.transform(test_action)

    train_input = np.concatenate([train_state, train_action], axis=-1)
    train_target = np.concatenate([train_next_state, train_reward], axis=-1)
    print(train_input.shape, train_target.shape)

    test_input = np.concatenate([test_state, test_action], axis=-1)
    test_target = np.concatenate([test_next_state, test_reward], axis=-1)
    print(test_input.shape, test_target.shape)

    # Convert numpy arrays to PyTorch tensors
    train_input_tensor = torch.tensor(train_input, dtype=torch.float32)
    train_target_tensor = torch.tensor(train_target, dtype=torch.float32)

    test_input_tensor = torch.tensor(test_input, dtype=torch.float32)
    test_target_tensor = torch.tensor(test_target, dtype=torch.float32)

    # Create TensorDataset and DataLoader for training and test sets
    train_dataset = TensorDataset(train_input_tensor, train_target_tensor)
    test_dataset = TensorDataset(test_input_tensor, test_target_tensor)

    # Create DataLoader with batch size and shuffle options
    batch_size = 64  # You can adjust this value as needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Print DataLoader information
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")

    net = MLPNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(_device_)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    dynamic_model = GaussianDynamic(net, EPOCHS, optimizer)
    dynamic_model.train(train_loader)

    


    

