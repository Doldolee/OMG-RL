import torch
import torch.nn as nn
import json
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.stats import linregress
import seaborn as sns

from buffer import BufferBatch, BufferSample
from dynamic_deterministic import DeterministicDynamicEnsemble
from omg_rl import OMGRL
from agent import SAC
from networks import CostNet
from util import preprocess_traj
from plot import line_plot_rewards, square_plot, box_plot_treatment_success_rate

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--epochs", type=int, default=500, help="Number of episodes, default: 500")
    parser.add_argument("--print_interval", type=int, default=5, help="")
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size, default: 256")
    parser.add_argument("--ensemble_size", type=int, default=7, help="")
    parser.add_argument("--elite_size", type=int, default=5, help="")
    parser.add_argument("--hidden_size", type=int, default=128, help="")

    parser.add_argument("--ensemble_state_size", type=int, default=16, help="")
    parser.add_argument("--ensemble_action_size", type=int, default=1, help="")

    parser.add_argument("--dynamic_weight_path", type=str, default="./model_weights/ensemble.pth", help="")
    parser.add_argument("--elite_model_idx_path", type=str, default="./model_weights/elite_model.pkl", help="")

    parser.add_argument("--n_action", type=int, default=6, help="")
    parser.add_argument("--n_state", type=int, default=16, help="")

    parser.add_argument("--data_path", type=str, default="./data/before_buffer/", help="")
    parser.add_argument("--demo_episodes", type=int, default=500, help="")
    parser.add_argument("--episodes_to_play", type=int, default=10, help="")
    parser.add_argument("--reward_scale", type=int, default=5, help="")
    parser.add_argument("--reward_function_update", type=int, default=10, help="")
    parser.add_argument("--policy_function_update", type=int, default=2, help="")
    parser.add_argument("--demo_batch", type=int, default=100, help="")

    args = parser.parse_args()
    return args


def main(config):
    buffer_expert = BufferBatch(config)
    buffer_expert.load(config.data_path + 'train')

    buffer_sample = BufferSample(config)

    # dynamic model
    dynamic_model = DeterministicDynamicEnsemble(config.ensemble_size, 
                                                 config.elite_size, 
                                                 config.ensemble_state_size, 
                                                 config.ensemble_action_size, 
                                                 reward_size=1,
                                                 hidden_size=config.hidden_size, 
                                                 use_decay=False)
    dynamic_model.load_weight(config.dynamic_weight_path, config.elite_model_idx_path)

    # omg_rl
    agent = SAC(config)
    omg_rl = OMGRL(config, agent)

    # reward network
    cost_net = CostNet(config.n_state + 1).to(config.device)
    cost_optimizer = torch.optim.Adam(cost_net.parameters(), 1e-3, weight_decay=1e-5)
    cost_function = [cost_net, cost_optimizer]

    # generate expert trajectory
    D_expert = np.array([])
    expert_trajs = buffer_expert.make_demo_trajs(config.demo_episodes, config.reward_scale)
    D_expert = preprocess_traj(expert_trajs, D_expert, is_Demo=True)


    best_eval_score = -10
    test_returns, test_costs, epoch_records = [], [], []
    for n_epi in range(config.epochs):
        omg_rl.train(dynamic_model, 
                    buffer_expert, 
                    buffer_sample, 
                    cost_function,
                    D_expert)
        
        if n_epi%config.print_interval==0 and n_epi!=0:
            test_returns, test_costs, epoch_records, best_eval_score = omg_rl.evaluate(n_epi, 
                                                                      best_eval_score, 
                                                                      test_returns, 
                                                                      test_costs, 
                                                                      epoch_records,
                                                                      dynamic_model,
                                                                      buffer_expert,
                                                                      cost_net)
            # plot_episode(test_returns, test_costs, epoch_records)
    torch.save(cost_net.state_dict(), "./model_weights/cost_net.pth")
    return test_returns, test_costs, epoch_records, omg_rl

if __name__ == "__main__":
    config = get_config()
    ################################train##########################################

    learned_reward_max_per_episode = []
    predefined_reward_max_per_episode = []

    test_leanred_reward_set, test_predefined_reward_set = [], []
    for _ in range(2):
        test_returns, test_costs, epoch_records, omg_rl = main(config)

        test_predefined_reward_set.append(test_returns)
        test_leanred_reward_set.append(test_costs)
        learned_reward_max_per_episode.append(np.max(test_costs))
        predefined_reward_max_per_episode.append(np.max(test_returns))


