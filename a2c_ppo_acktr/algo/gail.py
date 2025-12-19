import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd

from stable_baselines3.common.running_mean_std import RunningMeanStd


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, ratios, temp, device):
        super(Discriminator, self).__init__()

        self.ratios = ratios

        self.device = device

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()

        self.optimizer = torch.optim.Adam(self.trunk.parameters(), lr=1e-4)

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)

        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]


            policy_d = self.trunk(
                torch.cat([policy_state, policy_action], dim=1))


            (strict_expert_state, strict_expert_action),\
            (venture_expert_state, venture_expert_action),\
            (incorrect_expert_state, incorrect_expert_action) = expert_batch

            strict_expert_state = obsfilt(strict_expert_state.numpy(), update=False)
            venture_expert_state = obsfilt(venture_expert_state.numpy(), update=False)
            incorrect_expert_state = obsfilt(incorrect_expert_state.numpy(), update=False)

            strict_expert_state = torch.FloatTensor(strict_expert_state).to(self.device)
            venture_expert_state = torch.FloatTensor(venture_expert_state).to(self.device)
            incorrect_expert_state = torch.FloatTensor(incorrect_expert_state).to(self.device)

            strict_expert_action = strict_expert_action.to(self.device)
            venture_expert_action = venture_expert_action.to(self.device)
            incorrect_expert_action = incorrect_expert_action.to(self.device)

            strict_expert_d = self.trunk(
                torch.cat([strict_expert_state, strict_expert_action], dim=1))
            venture_expert_d = self.trunk(
                torch.cat([venture_expert_state, venture_expert_action], dim=1))
            incorrect_expert_d = self.trunk(
                torch.cat([incorrect_expert_state, incorrect_expert_action], dim=1))

            strict_expert_loss = F.binary_cross_entropy_with_logits(
                strict_expert_d,
                torch.ones(strict_expert_d.size()).to(self.device))
            
            venture_expert_loss = F.binary_cross_entropy_with_logits(
                venture_expert_d,
                torch.ones(venture_expert_d.size()).to(self.device))
            
            incorrect_expert_loss = F.binary_cross_entropy_with_logits(
                incorrect_expert_d,
                torch.ones(incorrect_expert_d.size()).to(self.device)) 
            
            t = (np.array(self.ratios) / (self.ratios[0] + self.ratios[1] - self.ratios[2])) 
            
            expert_loss = t[0] * strict_expert_loss + t[1] * venture_expert_loss - t[2] * incorrect_expert_loss
            
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss

            strict_grad_pen = self.compute_grad_pen(strict_expert_state, strict_expert_action,
                                             policy_state, policy_action)
            
            venture_grad_pen = self.compute_grad_pen(venture_expert_state, venture_expert_action,
                                             policy_state, policy_action)

            incorrect_grad_pen = self.compute_grad_pen(incorrect_expert_state, incorrect_expert_action,
                                                     policy_state, policy_action)
            
            grad_pen = t[0] * strict_grad_pen + t[1] * venture_grad_pen - t[2] * incorrect_grad_pen 

            loss += (gail_loss + grad_pen ).item()

            n += 1

            self.optimizer.zero_grad()

            (gail_loss + grad_pen).backward()

            self.optimizer.step()

        return loss / n

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)

            s_clipped = torch.clamp(s, 1e-8, 1 - 1e-8)
            reward = s_clipped.log() - (1 - s_clipped).log()
                
            if torch.isnan(reward).any() or torch.isinf(reward).any():
                reward = torch.logit(s, eps=1e-6)

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)


def subset_parts_by_ratio_and_count(parts, ratio, total_trajs, strict_trajs):

    prefixes = ["incorrect", "strict", "venture"]
    assert len(ratio) == len(prefixes)
    assert sum(ratio) == 1

    categorized_indices = {prefix: [] for prefix in prefixes}
    for i, infos in enumerate(parts['infos']):
        if infos:
            prefix = next((p for p in prefixes if p in infos[0]['cid']), None)
            if prefix:
                categorized_indices[prefix].append(i)

    for prefix in prefixes:
        random.shuffle(categorized_indices[prefix])

    strict_selected = categorized_indices['strict'][:strict_trajs]
    total_trajs = int(len(strict_selected) / ratio[0])    

    remaining_trajs = total_trajs - len(strict_selected)

    other_ratio_sum = ratio[1] + ratio[2] 
    items_to_select = {
        'incorrect': int(ratio[2] / other_ratio_sum * remaining_trajs),
        'venture': remaining_trajs - int(ratio[2] / other_ratio_sum * remaining_trajs)
    }

    if items_to_select['venture'] == 0:
        items_to_select['venture'] = 1
    if items_to_select['incorrect'] == 0:
        items_to_select['incorrect'] = 1

    venture_selected = categorized_indices['venture'][:items_to_select['venture']]
    incorrect_selected = categorized_indices['incorrect'][:items_to_select['incorrect']]

    subsets = {prefix: {key: [] for key in parts.keys()} for prefix in prefixes}

    for i in strict_selected:
        for key in parts.keys():
            subsets['strict'][key].append(parts[key][i])

    for i in venture_selected:
        for key in parts.keys():
            subsets['venture'][key].append(parts[key][i])

    for i in incorrect_selected:
        for key in parts.keys():
            subsets['incorrect'][key].append(parts[key][i])

    return subsets


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, all_trajectories, num_trajectories=10, ratios=(0.5,0.3,0.2)):
      
        all_subsets = subset_parts_by_ratio_and_count(all_trajectories, ratios, int(num_trajectories / ratios[0]), num_trajectories)
        
        self.trajectories = {
            'strict': {},
            'venture': {},
            'incorrect': {}
        }

        for subset_name in ["incorrect", "strict", "venture"]: 
            subset_data = all_subsets[subset_name]

            idx = list(range(len(subset_data['obs'])))
            random.shuffle(idx)
            
            for k, v in subset_data.items():
                data = [v[i] for i in idx]

                if k not in ['lengths', 'infos']:
                    samples = []
                    for i in range(min(num_trajectories, len(data))):
                        tmp = torch.tensor(data[i])
                        samples.append(tmp)
                    samples = torch.cat(samples, dim=0)
                    self.trajectories[subset_name][k] = samples
                else:
                    self.trajectories[subset_name][k] = data

            self.trajectories[subset_name]['length'] = sum(self.trajectories[subset_name]['lengths'])
            

        self.length = sum(self.trajectories['strict']['lengths'])
            
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        strict_obs = self.trajectories["strict"]['obs'][i]
        strict_acts = torch.tensor([self.trajectories["strict"]['acts'][i]])

        venture_idx = random.randint(0, len(self.trajectories["venture"]['obs']) - 1)
        venture_obs = self.trajectories["venture"]['obs'][venture_idx]
        venture_acts = torch.tensor([self.trajectories["venture"]['acts'][venture_idx]])

        incorrect_idx = random.randint(0, len(self.trajectories["incorrect"]['obs']) - 1)
        incorrect_obs = self.trajectories["incorrect"]['obs'][incorrect_idx]
        incorrect_acts = torch.tensor([self.trajectories["incorrect"]['acts'][incorrect_idx]])

        return (strict_obs, strict_acts), (venture_obs, venture_acts), (incorrect_obs, incorrect_acts)