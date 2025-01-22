import numpy as np
import random
import torch
from collections import deque, namedtuple

class BufferSample:
    def __init__(self, config):

        self.device = config.device
        self.memory = deque(maxlen=config.buffer_size)  
      
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class BufferBatch:
    def __init__(self, config):
        self.buffer_capacity = config.buffer_size
        self.crt_size = 0
        
        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state = np.zeros((self.buffer_capacity, config.n_state))
        self.action = np.zeros((self.buffer_capacity, 1))
        self.reward = np.zeros((self.buffer_capacity, 1))
        self.done = np.zeros((self.buffer_capacity, 1))
        self.next_state = np.zeros((self.buffer_capacity, config.n_state))

    
    def load(self, path):
        rewards = np.load(path + "/reward.npy")
        actions = np.load(path + "/action.npy")
        states = np.load(path + "/state.npy")
        next_states = np.load(path + "/next_state.npy")
        dones = np.load(path + "/done.npy")
        
        self.crt_size = rewards.shape[0]

        self.state[:self.crt_size] = states[:self.crt_size]
        self.action[:self.crt_size] = actions[:self.crt_size]
        self.next_state[:self.crt_size] = next_states[:self.crt_size]
        self.done[:self.crt_size] = dones[:self.crt_size]
        self.reward[:self.crt_size] = rewards[:self.crt_size]
        print(f"Replay buffer loaded with {self.crt_size} elements")


    def sample(self, batch_size):
        idx = np.random.randint(0, self.crt_size, size=batch_size)
        return(
                self.state[idx],
                self.action[idx],
                self.reward[idx],
                self.next_state[idx],
                self.done[idx]
            )
    
    def random_sample_all_batch(self):
        idxes = np.random.randint(0, len(self.state), self.crt_size)
        return(
                self.state[idxes],
                self.action[idxes],
                self.next_state[idxes],
                self.reward[idxes],
                self.done[idxes]
              )
    
    def make_demo_trajs(self, demo_episodes, reward_scale):
        demo_trajs = []
        states, actions, rewards, probs = [], [], [], []
        
        episode_count=1
        for i in range(self.crt_size):
            state, action, reward, done = self.state[i], self.action[i], self.reward[i], self.done[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward[0]*reward_scale)
            ## action prob는 모르니까 0으로 할당
            probs.append([0,0,0,0,0,0])
            if done[0] == 0:
                demo_trajs.append([states, actions, rewards, probs])
                states, actions, rewards, probs = [], [], [], []
                if episode_count == demo_episodes:
                    break

                episode_count +=1
            
        return np.array(demo_trajs, dtype='object')