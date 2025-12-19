import pickle
import numpy as np
import torch

from data_utils import set_seed, timeline_convert_merge_post


class ENV(object):

    def __init__(self, normalizer, seed, data_path, cache_key, expert_cids):

        with open(data_path, 'rb') as f:
            cached_results = pickle.load(f)

        all_data = cached_results[cache_key]


        self.normalizer = normalizer

        self.trajectories = {}

        for k, value in all_data.items():
            if k in expert_cids:
                continue

            value["cid"] = k

            timelines = value["seq"]["timeline"]
            seqs_list = value["seq"]["texts"]
            label = value["label"]

            tids = value["seq"]["tids"]

            _, sub_seqs, _ = timeline_convert_merge_post(timelines, seqs_list, tids, 1)


            assert len(sub_seqs) != 0

            if len(sub_seqs) == 1:
                continue

            states = value["states"]
            states_vec_stop = value["obs_stop"]
            states_vec_con = value["obs_continue"]
            
            self.trajectories[k] = {"cid":k, "label":label, "seq_len": len(seqs_list)}
            if len(states) == 1:
                self.trajectories[k]["obs"] = [states,states_vec_stop,states_vec_con]
                self.trajectories[k]["next_obs"] = [states,states_vec_stop,states_vec_con]
            else:
                self.trajectories[k]["obs"] = [states[:-1],states_vec_stop[:-1],states_vec_con[:-1]]
                self.trajectories[k]["next_obs"] = [states[1:],states_vec_stop[1:],states_vec_con[1:]]

            dones = np.zeros(len(self.trajectories[k]["obs"][0]), dtype=bool)
            dones[-1] = True
            self.trajectories[k]["dones"] = dones

            assert len(self.trajectories[k]["obs"][2]) == len(self.trajectories[k]["next_obs"][1]) == len(self.trajectories[k]["dones"])
        
        self.trajectories_ids = list(self.trajectories.keys())

        self.current_traj = None
        self.traj_step = None
            
        set_seed(seed=seed)

    
    def reset(self,num_env=1):

        np.random.shuffle(self.trajectories_ids)
        self.current_traj = self.trajectories_ids[0]

        first_obs = self.trajectories[self.current_traj]["obs"][1][0]

        self.traj_step = 0
        self.current_traj_len = len(self.trajectories[self.current_traj]["obs"][0])
        first_obs = self.normalizer._obfilt(first_obs)
        return torch.tensor(first_obs)

    
    def step(self, action):

        current_act = action[0][0].item()
        reward = 0.0 # placeholder only
        terminated = self.trajectories[self.current_traj]["dones"][self.traj_step]

        next_obs_idx = None
        if current_act == 1:
            next_obs_idx = 1
        else:
            next_obs_idx = 2

        next_obs = self.trajectories[self.current_traj]["next_obs"][next_obs_idx][self.traj_step]
        next_obs = self.normalizer._obfilt(next_obs)
        next_obs = torch.tensor(next_obs)
        self.traj_step += 1

        return next_obs, torch.tensor([reward]), [terminated], [{}]

class ENV_EVAL(object):

    def __init__(self, normalizer, seed, data_path, cache_key):

        with open(data_path, 'rb') as f:
            cached_results = pickle.load(f)
        
        all_data = cached_results[cache_key]

        self.normalizer = normalizer

        self.trajectories = {}

        for k, value in all_data.items():

            value["cid"] = k
            seqs_list = value["seq"]["texts"]
            label = value["label"]

            states = value["states"]
            states_vec_stop = value["obs_stop"]
            states_vec_con = value["obs_continue"]
            
            self.trajectories[k] = {"cid":k, "label":label, "seq_len": len(seqs_list)}
            if len(states) == 1:
                self.trajectories[k]["obs"] = [states,states_vec_stop,states_vec_con]
                self.trajectories[k]["next_obs"] = [states,states_vec_stop,states_vec_con]
            else:
                self.trajectories[k]["obs"] = [states[:-1],states_vec_stop[:-1],states_vec_con[:-1]]
                self.trajectories[k]["next_obs"] = [states[1:],states_vec_stop[1:],states_vec_con[1:]]

            dones = np.zeros(len(self.trajectories[k]["obs"][0]), dtype=bool)
            dones[-1] = True
            self.trajectories[k]["dones"] = dones

            assert len(self.trajectories[k]["obs"][2]) == len(self.trajectories[k]["next_obs"][1]) == len(self.trajectories[k]["dones"])
        
        self.trajectories_ids = list(self.trajectories.keys())

        self.current_traj = None
        self.traj_step = None
            

        set_seed(seed=seed)

    
    def reset(self,trajectory_id, num_env=1):

        self.current_traj = trajectory_id

        first_obs = self.trajectories[self.current_traj]["obs"][1][0]
       
        self.traj_step = 0
        self.current_traj_len = len(self.trajectories[self.current_traj]["obs"][0])

        first_obs = self.normalizer._obfilt(first_obs)
        return torch.tensor(first_obs)
    
    def step(self, action):
       
        current_act = action[0][0].item()
        reward = 0.0
        terminated = self.trajectories[self.current_traj]["dones"][self.traj_step]
        
        next_obs_idx = None

        if current_act == 1:
            next_obs_idx = 1
        else:
            next_obs_idx = 2


        current_obs_text = self.trajectories[self.current_traj]["obs"][0][self.traj_step]

        next_obs = self.trajectories[self.current_traj]["next_obs"][next_obs_idx][self.traj_step]
        next_obs = self.normalizer._obfilt(next_obs)
        next_obs = torch.tensor(next_obs)
        self.traj_step += 1
        return current_obs_text, next_obs, torch.tensor([reward]), [terminated], [{}]
