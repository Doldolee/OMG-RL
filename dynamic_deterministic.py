import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
from pickle import load
import itertools
import pickle
import matplotlib.pyplot as plt

from buffer import BufferBatch

_device_ = 'cuda'

def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x
    
class EnsembleFC(nn.Module):
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features:int, out_features:int, ensemble_size:int, weight_decay:float=0., bias: bool=True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class EnsembleModel(nn.Module):
    def __init__(self, state_size, action_size, reward_size, ensemble_size, hidden_size, learning_rate=1e-3, use_decay=False):
        super(EnsembleModel, self).__init__()
        self.hidden_size = hidden_size
        self.use_decay = use_decay
        self.output_dim = state_size + reward_size

        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size)
        # self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size)
        self.nn4 = EnsembleFC(hidden_size, hidden_size//2, ensemble_size)
        self.nn5 = EnsembleFC(hidden_size//2, self.output_dim, ensemble_size)



        self.optimizer = torch.optim.Adam([
                {'params': self.nn1.parameters()},
                # {'params': self.nn2.parameters()},
                {'params': self.nn3.parameters()},
                {'params': self.nn4.parameters()},
                {'params': self.nn5.parameters()}
                ], lr=learning_rate)
        
        self.apply(init_weights)
        self.swish = Swish()

        self.relu = nn.ReLU()

    def forward(self, x):
        nn1_output = self.relu(self.nn1(x))
        # nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.relu(self.nn3(nn1_output))
        nn4_output = self.relu(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)

        return nn5_output

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss

    def batch_loss(self, output, labels):
        
        mse_loss = torch.mean(torch.pow(output-labels,2),dim=(1,2))
        total_loss = torch.sum(mse_loss)

        return total_loss, mse_loss

    def batch_train(self, loss):
        self.optimizer.zero_grad()

        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        self.optimizer.step()

class DeterministicDynamicEnsemble():
    def __init__(self, ensemble_size, elite_size, state_size, action_size, reward_size=1, hidden_size=128, use_decay=False):
        self.ensemble_size = ensemble_size
        self.elite_size = elite_size
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.elite_model_idxes = []
        self.ensemble_model = EnsembleModel(state_size, action_size, reward_size, ensemble_size, hidden_size, use_decay=use_decay).to(_device_)
        

    def train(self, inputs, labels, batch_size, max_epochs_since_update=15):
        epoch_count=[]
        test_loss_count1=[]
        test_loss_count2=[]
        test_loss_count3=[]
        test_loss_count4=[]
        test_loss_count5=[]
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.ensemble_size)}


        train_inputs , test_inputs = inputs
        train_labels, test_labels = labels

        
        train_inputs = torch.from_numpy(train_inputs).float().to(_device_)
        train_labels = torch.from_numpy(train_labels).float().to(_device_)
        train_inputs = train_inputs[None, :, :].repeat([self.ensemble_size, 1, 1])
        train_labels = train_labels[None, :, :].repeat([self.ensemble_size, 1, 1])

        test_inputs = torch.from_numpy(test_inputs).float().to(_device_)
        test_labels = torch.from_numpy(test_labels).float().to(_device_)
        test_inputs = test_inputs[None, :, :].repeat([self.ensemble_size, 1, 1])
        test_labels = test_labels[None, :, :].repeat([self.ensemble_size, 1, 1])


        for epoch in itertools.count():
            epoch_count.append(epoch)
            losses = []

            self.ensemble_model.train()
            for start_pos in range(0, train_inputs.shape[1], batch_size):
                train_input = train_inputs[:,start_pos:start_pos+batch_size,:].to(_device_)
                train_label = train_labels[:,start_pos:start_pos+batch_size,:].to(_device_)

                train_output = self.ensemble_model(train_input)#(7,batch,17)
                train_total_loss, train_mse_loss = self.ensemble_model.batch_loss(train_output, train_label)
                self.ensemble_model.batch_train(train_total_loss)
                losses.append(np.mean(train_mse_loss.detach().cpu().numpy()))
      
            print('epoch: {}, train_mse_loss: {}'.format(epoch, np.mean(losses)))
            test_loss_count1.append(losses[0])
            test_loss_count2.append(losses[1])
            test_loss_count3.append(losses[2])
            test_loss_count4.append(losses[3])
            test_loss_count5.append(losses[4])

            plt.figure()
            plt.title(f"Mse_loss")
            plt.plot(epoch_count,test_loss_count1, label='model1')
            plt.plot(epoch_count,test_loss_count2, label='model2')
            plt.plot(epoch_count,test_loss_count3, label='model3')
            plt.plot(epoch_count,test_loss_count4, label='model4')
            plt.plot(epoch_count,test_loss_count5, label='model5')

            plt.grid()
            plt.legend()

            plt.savefig('./plot/ensemble.png', dpi=400)
            plt.close()

            self.ensemble_model.eval()
            with torch.no_grad():
                test_outputs = self.ensemble_model(test_inputs)
                _, test_mse_loss = self.ensemble_model.batch_loss(test_outputs, test_labels)
                test_mse_loss = test_mse_loss.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(test_mse_loss)
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()

                # break_train = self._save_best(epoch, test_mse_loss)
                # if break_train:
                #     break
            if epoch == 1000:
                break
            
         
            
            print('epoch: {}, test_mse_loss: {}'.format(epoch, np.mean(test_mse_loss)))
            print("------------------------------------------")
        
        # torch.save(self.ensemble_model.state_dict(), "./model_weights/ensemble.pth")
        # with open("./model_weights/elite_model.pkl",'wb') as f:
        #     pickle.dump(self.elite_model_idxes, f)


    def _save_best(self, epoch, test_losses):
        updated = False
        for i in range(len(test_losses)):
            current = test_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.001:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def load_weight(self, weight_path, elite_model_idx_path):
        self.ensemble_model.load_state_dict(torch.load(weight_path))
        with open(elite_model_idx_path,"rb") as f:
            self.elite_model_idxes = pickle.load(f)



    def step(self, inputs, steps):

        terminated, truncated = False, False

        inputs = torch.from_numpy(inputs).float().to(_device_)
        inputs = inputs[None, :, :].repeat([self.ensemble_size, 1, 1])
        output = self.ensemble_model(inputs)
        
        #randomly pick dynamic model
        elite_model_idx = np.random.choice(self.elite_model_idxes,1)[0]
        output = output[elite_model_idx,:,:]
        reward = output[:,:1]
        next_state = output[:,1:]

        if reward[0][0] > 1:
            reward[0][0] = 1
            # print(f"ensemble model reward range upper 1: {reward[0][0]}")
        if reward[0][0] < -1:
            reward[0][0] = -1
            # print(f"ensemble model reward range lower -1: {reward[0][0]}")

        ## truncated rule
        if steps > 100:
            terminated = True
            truncated = True

        # if reward[0][0] > 0.9:
        #     terminated = False
        #     truncated = False

    
        return reward[0][0].detach().cpu().numpy(), next_state[0].detach().cpu().numpy(), terminated, truncated


if __name__=='__main__':
    # EPOCHS = 50
    ENSEMBLE_SIZE = 7
    ELITE_SIZE = 5
    STATE_SIZE = 16
    ACTION_SIZE = 1
    HIDDEN_SIZE = 128
    BATCH_SIZE = 128

    NUM_STATES = 16
    data_path = "./data/before_buffer/"

    # scaling 완료됨
    train_replay_memory = BufferBatch(state_size =STATE_SIZE)
    train_replay_memory.load(data_path + 'train')
    print("train memory data size:",train_replay_memory.crt_size)   
    test_replay_memory = BufferBatch(state_size =STATE_SIZE)
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
    train_label = np.concatenate([train_reward, train_next_state], axis=-1)
    print(train_input.shape, train_label.shape)

    test_input = np.concatenate([test_state, test_action], axis=-1)
    test_label = np.concatenate([test_reward, test_next_state], axis=-1)
    print(test_input.shape, test_label.shape)

    inputs = (train_input, test_input)
    labels = (train_label, test_label)

    dynamic_model = DeterministicDynamicEnsemble(ENSEMBLE_SIZE, ELITE_SIZE, STATE_SIZE, ACTION_SIZE, reward_size=1, hidden_size=HIDDEN_SIZE, use_decay=False)
    dynamic_model.train(inputs, labels, BATCH_SIZE)







    

    


