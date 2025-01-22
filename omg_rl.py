import numpy as np
import matplotlib.pyplot as plt
import torch

from agent import SAC
from util import preprocess_traj

class OMGRL:
    def __init__(self, config, agent):
        
        self.config = config
        self.action_scaler = {
            0:-1.46134109,
            1:-0.8774269,
            2:-0.2935127,
            3:0.29040149,
            4:0.87431569,
            5:1.45822988
        }

        self.n_action = config.n_action
        self.n_state = config.n_state

        self.agent = agent
        self.reward_scale = config.reward_scale

    def train(self, dynamic_model, buffer_expert, buffer_sample, cost_function, D_expert):
        cost_net, cost_optimizer = cost_function
        cost_net.train()
    
        ## ROLLOUT SAMPLE TRAJECTORY
        # also add generated trajectory to buffer_sample
        D_samp = np.array([])
        sample_trajs = [self.generate_session(dynamic_model, buffer_expert, buffer_sample) for _ in range(self.config.episodes_to_play)]
        D_samp = preprocess_traj(sample_trajs, D_samp)

        ## TRAIN REWARD FUNCTION
        loss_reward = 0.0
        for _ in range(self.config.reward_function_update):
            selected_samp = np.random.choice(len(D_samp), self.config.demo_batch)
            selected_expert = np.random.choice(len(D_expert), self.config.demo_batch)

            D_s_samp = D_samp[selected_samp]
            D_s_expert = D_expert[selected_expert]

            #D̂ samp ← D̂ demo ∪ D̂ samp
            D_s_samp = np.concatenate((D_s_expert, D_s_samp), axis = 0)

            states, probs, actions = D_s_samp[:,:-2], D_s_samp[:,-2], D_s_samp[:,-1]
            states_expert, actions_expert = D_s_expert[:,:-2], D_s_expert[:,-1]

            states = torch.from_numpy(states).float().to(self.config.device)
            actions = torch.from_numpy(actions).float().to(self.config.device)
            states_expert = torch.from_numpy(states_expert).float().to(self.config.device)
            actions_expert = torch.from_numpy(actions_expert).float().to(self.config.device)
            probs = torch.from_numpy(probs).float().to(self.config.device)

            costs_samp = cost_net(torch.cat((states, actions.reshape(-1, 1)), dim=-1))
            costs_expert = cost_net(torch.cat((states_expert, actions_expert.reshape(-1, 1)), dim=-1))

            # LOSS CALCULATION FOR IOC (COST FUNCTION)
            loss_IOC = torch.mean(costs_expert) + torch.log(torch.mean(torch.exp(-costs_samp)/(probs+1e-7)))
    
            # UPDATING THE COST FUNCTION
            cost_optimizer.zero_grad()
            loss_IOC.backward()
            cost_optimizer.step()

            loss_reward += loss_IOC.detach().cpu().numpy()
        print("cost function loss:", loss_reward/self.config.reward_function_update)
        reward_loss = 0.0

        ## TRAIN POLICY
        for _ in range(self.config.policy_function_update):
            # collect_random(env=dynamic_model, dataset=self.buffer_RL, buffer_ensemble=buffer_ensemble, reward_scale=5,num_samples=100)
            states_sample, actions_sample, rewards_sample, next_states_sample, dones_sample = buffer_sample.sample(batch_size = self.config.batch_size//2)
            states_expert, actions_expert, rewards_expert, next_states_expert, dones_expert = buffer_expert.sample(batch_size = self.config.batch_size//2)
            states_expert = torch.from_numpy(states_expert).float().to(self.config.device)
            actions_expert = torch.from_numpy(actions_expert).float().to(self.config.device)
            rewards_expert = torch.from_numpy(rewards_expert).float().to(self.config.device)
            next_states_expert = torch.from_numpy(next_states_expert).float().to(self.config.device)
            dones_expert = torch.from_numpy(dones_expert).float().to(self.config.device)
            
            samples = (states_sample, actions_sample, rewards_sample, next_states_sample, dones_sample)
            experts = (states_expert, actions_expert, rewards_expert, next_states_expert, dones_expert)

            self.agent.learn(samples, experts, cost_net, gamma=0.99)

    def evaluate(self, 
                 epoch, 
                 best_eval_score, 
                 test_returns, 
                 test_costs,
                 epoch_records, 
                 dynamic_model, 
                 buffer_expert, 
                 cost_net):
    
        num_episodes = 0
        total_return = 0.0
        total_cost = 0.0
        num_eval_episode = 5 # episode 5번의 평균 누적보상으로 모델 평가
        cost_net.eval()
        
        while True:
            state, _, _, _, _ = buffer_expert.sample(batch_size=1)

            episode_steps = 0
            episode_return = 0.0
            episode_cost = 0.0
            truncated = False
            while (not truncated):
                action = self.agent.greedy_action(state)

                scaled_a = np.expand_dims(np.expand_dims(np.array(self.action_scaler[action[0][0]]), axis=0),axis=0)
                inputs = np.concatenate([state, scaled_a], axis=-1)

                reward, next_state, terminated, truncated = dynamic_model.step(inputs, episode_steps)
        
                input = torch.cat([torch.Tensor(state), torch.Tensor(action)], dim=-1).to(self.config.device)
                cost = cost_net(input)
                # cost = torch.tensor(list(map(lambda x: 5 if x>5 else (-5 if x<-5 else x) ,cost.detach().cpu().numpy())), dtype=torch.float32)

                episode_steps += 1
                episode_return += reward
                episode_cost += cost.detach().cpu().numpy()[0][0]/25
                state = np.expand_dims(next_state, axis=0)
        
                # if truncated:
                    # print("truncated")
         
            num_episodes += 1
            total_return += episode_return
            total_cost += episode_cost

            if num_episodes == num_eval_episode:
                break

        mean_return = total_return / num_eval_episode
        mean_cost = total_cost / num_eval_episode
        if mean_return > best_eval_score:
            best_eval_score = mean_return
            # torch.save(cost_f.state_dict(), f"./model_weights/cost/{best_eval_score}cost.pth")

        print(f'Num epoch: {epoch}, return: {mean_return}, cost: {mean_cost}, best return:{best_eval_score}')
        test_returns.append(mean_return)
        test_costs.append(mean_cost)
        epoch_records.append(epoch)

        return test_returns, test_costs, epoch_records, best_eval_score

    ## one episode
    def generate_session(self, dynamic_model, buffer_expert, buffer_sample):
        states, actions, rewards, probs = [], [], [], []
        state, _, _, _, _ = buffer_expert.sample(batch_size=1)

        for t in range(20):
            action_probs = self.agent.get_action_prob(state)
            action = np.random.choice(self.n_action,  p = action_probs.squeeze(0))


            scaled_a = np.expand_dims(np.expand_dims(np.array(self.action_scaler[action]), axis=0),axis=0)
            inputs = np.concatenate([state, scaled_a], axis=-1)
   
            reward, next_state, terminated, truncated = dynamic_model.step(inputs, t)

            buffer_sample.add(state[0], [action], reward*self.reward_scale, next_state, terminated)

            states.append(np.squeeze(state,axis=0))
            actions.append(action)
            rewards.append(reward*self.reward_scale)
            probs.append(action_probs.squeeze(0))

 
            state = np.expand_dims(next_state,axis=0)
           
            if terminated or truncated:
                break
            
        return states, actions, rewards, probs