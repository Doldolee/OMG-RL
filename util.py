import numpy as np
import torch
from buffer import *
from dynamic_deterministic import *

def get_action_on_trajectory(agent, config):
    
    device = "cuda"
    buffer = BufferBatch(config)
    buffer.load(config.data_path + 'test')

    clinician_action_set = []
    ai_action_set = []

    pt_set = []
    inr_set = []

    done_forward=0
    with torch.no_grad():

        all_done = torch.FloatTensor(buffer.done[:buffer.crt_size])
        done_index = torch.where(all_done==0)[0]
        
        count = 0
        treatement_success = 0.0
        for i in done_index:
            state = buffer.state[done_forward:i+1] #trajectory 1개 done_forward ~ done index
            action = buffer.action[done_forward:i+1]
            reward = buffer.reward[done_forward:i+1]
            reward = reward.squeeze(-1)
            ai_action = agent.greedy_action(state)
            # ai_action = agent.get_action(state)
            # ai_action = np.expand_dims(ai_action, axis=-1)
            clinician_action_set.extend(action.astype(int).squeeze(-1).tolist())
            ai_action_set.extend(ai_action.astype(int).squeeze(-1).tolist())


            pt = state[:,8]
            inr = state[:,12]
            pt_set.extend(pt.tolist())
            inr_set.extend(inr.tolist())
           

            if any(num > 0.9 for num in reward):
                treatement_success+=1
            count+=1

            
            done_forward = i+1
        print("clinician treatement success rate:", treatement_success/count)
    return clinician_action_set, ai_action_set, pt_set, inr_set

def get_treatment_success_rate(agent, config):
    buffer_expert = BufferBatch(config)
    buffer_expert.load(config.data_path + 'train')

    action_scaler = {
            0:-1.46134109,
            1:-0.8774269,
            2:-0.2935127,
            3:0.29040149,
            4:0.87431569,
            5:1.45822988
        }
    
    dynamic_model = DeterministicDynamicEnsemble(config.ensemble_size, 
                                                 config.elite_size, 
                                                 config.ensemble_state_size, 
                                                 config.ensemble_action_size, 
                                                 reward_size=1,
                                                 hidden_size=config.hidden_size, 
                                                 use_decay=False)
    dynamic_model.load_weight(config.dynamic_weight_path, config.elite_model_idx_path)
    print("start")

    success = 0
    for _ in range(500):
        # 1 episode start
        episode_steps = 0
        state, _, _, _, _ = buffer_expert.sample(batch_size=1)
        truncated = False
        while (not truncated):
            
            action = agent.greedy_action(state)[0][0]

            scaled_a = np.expand_dims(np.expand_dims(np.array(action_scaler[action]), axis=0),axis=0)
            inputs = np.concatenate([state, scaled_a], axis=-1)

            reward, next_state, terminated, truncated = dynamic_model.step(inputs, episode_steps)
            if reward >=0.2:
                success +=1
                break

            state = np.expand_dims(next_state,axis=0)
            episode_steps += 1

        
    print("OMG-RL success rate: ",success/500)

def preprocess_traj(traj_list, step_list, is_Demo = False):
    step_list = step_list.tolist()
    for traj in traj_list:

        states = np.array(traj[0])
        if is_Demo:
            probs = np.ones((states.shape[0], 1))
        else:
            # print(np.array(traj[1]).squeeze().shape)
            s_prob = []
            for prob, action in zip(traj[3], traj[1]):
                s_prob.append(prob[action])
            probs = np.array(s_prob).reshape(-1, 1)

        actions = np.array(traj[1]).reshape(-1,1)

        x = np.concatenate((states, probs, actions), axis=1) 
        step_list.extend(x)
    return np.array(step_list)