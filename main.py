import os
import json
import pickle
import datetime
import torch

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import VecNormalize
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from inference import infer_mistral, infer_llama, infer_gpt

from data_utils import *
from env import ENV


def main(config):

    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Start Training: {TIMESTAMP}")

    args = get_args()

    torch.manual_seed(config['training']['seed'])
    torch.cuda.manual_seed_all(config['training']['seed'])

    if config['training']['cuda'] and torch.cuda.is_available() and config['training']['cuda_deterministic']:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda" if config['training']['cuda'] else "cpu")
    
    actor_critic = Policy(
        config['model']['OBS_DIM'],
        config['model']['ACT_DIM'],
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    agent = algo.PPO(
        actor_critic,
        config['training']['clip_param'],
        config['training']['ppo_epoch'],
        config['training']['num_mini_batch'],
        config['training']['value_loss_coef'],
        config['training']['entropy_coef'],
        lr=config['training']['lr'],
        eps=config['training']['eps'],
        max_grad_norm=config['training']['max_grad_norm'])

    if config['training']['gail']:
        discr = gail.Discriminator(
            config['model']['OBS_DIM']+1, 100,
            config['training']['trajs_ratios'], config['training']['temp'], 
            device)

        set_seed(config['training']['seed'])
        cache_file = config['paths']['cache_file'].format(
            dataset_split=config['dataset']['dataset_split'],
            dataset_type=config['dataset']['dataset_type'],
            expert_llm = config['training']['expert_llm'].split("/")[-1]
        )
        
        cache_key = (f"d{config['dataset']['dataset_split']}_train_{config['dataset']['dataset_type']}", 
                     f"seed{config['training']['seed']}", f"n_sample{config['dataset']['n_sample']}", "expert", "formatted")
       
        
        with open(cache_file, 'rb') as f:
            cached_results = pickle.load(f)

        all_trajectories = cached_results[cache_key]["trajectories"]

        cache_key_vec = cache_key[:-1] + ("vec",)
        
        exp_cids = set([k.split("_")[0] for k in cached_results[cache_key_vec].keys()])
    
        expert_dataset = gail.ExpertDataset(
            all_trajectories, num_trajectories=config['training']['num_trajectories'],\
            ratios=config['training']["trajs_ratios"])
        
        drop_last = len(expert_dataset) > config['training']['gail_batch_size']
        gail_bs = min([len(expert_dataset), config['training']['gail_batch_size']])

        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=gail_bs,
            shuffle=True,
            drop_last=drop_last)

    normalizer = VecNormalize((config['model']['OBS_DIM'],),norm_obs=config['training']['norm_obs'])
    cache_key_env = (f"d{config['dataset']['dataset_split']}_train_{config['dataset']['dataset_type']}", 
                     f"seed{config['training']['seed']}", f"n_sample{config['dataset']['n_sample']}", "env", "vec")
    env = ENV(normalizer, config['training']['seed'], cache_file, cache_key_env, exp_cids)

    num_iteration = config['training']['num_iteration']
    num_step = config['training']['num_steps']
    total_num_steps = []
    for j in range(num_iteration):

        obs = env.reset()

        rollouts = RolloutStorage(num_step, args.num_processes,
                                (config['model']['OBS_DIM'],), config['model']['ACT_DIM'],
                                actor_critic.recurrent_hidden_state_size)

        rollouts.obs[0].copy_(obs)
        rollouts.to(device)


        if config['training']['use_linear_lr_decay']:
            utils.update_linear_schedule(
                agent.optimizer, j, num_iteration,
                agent.optimizer.lr if config['training']['algo'] == "acktr" else config['training']['lr'])

        for step in range(num_step):
          
            with torch.no_grad():

                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step]) 

            obs, reward, done, infos = env.step(action)

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            
            
            if done[0]:
                obs = env.reset()
                rollouts.obs[rollouts.step].copy_(obs)


        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if config['training']['lr']:
            if j >= config['training']["num_warmup_iteration"]:
                normalizer.eval() 

            gail_epoch = config['training']["gail_epoch"]
            if j < config['training']["num_warmup_iteration"]:
                gail_epoch = config['training']["gail_warm_up_epoch"]
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             normalizer._obfilt)

            for step in range(num_step):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], config['training']['gamma'],
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, config['training']['use_gae'], config['training']['gamma'],
                                 config['training']['gae_lambda'], config['training']['use_proper_time_limits'])

        agent.num_mini_batch = config['training']['num_mini_batch']

        agent.update(rollouts)
        total_num_steps.append(num_step)

        if (j % config['training']['save_interval'] == 0 
                or j == num_iteration - 1 and j != 0) and config['paths']['save_dir'] != "":
            mode_save_dir = config['paths']['save_dir'].format(
                expert_llm = config['training']['expert_llm'].split("/")[-1],
            )
            model_save_path = os.path.join(mode_save_dir, config['training']['model_type'],config['dataset']['dataset_type'],f"d{config['dataset']['dataset_split']}",str(TIMESTAMP))
            cache_key_save = tuple([ck.split(".")[0] for ck in cache_key[:-1]]) + (f"iteration{j}",)
            save_prefix = "_".join(cache_key_save)
            try:
                os.makedirs(model_save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(normalizer, 'obs_rms', None)
            ], os.path.join(model_save_path, save_prefix + ".pt"))

        if j==num_iteration-1:

            obs_rms = normalizer.obs_rms

            test_data_path = config['paths']["test_data_vec_path"].format(
                dataset_type=config['dataset']['dataset_type'],
                dataset_split=config['dataset']['dataset_split']
            )
            
            save_path = config['paths']['save_path'].format(
                dataset_split=config['dataset']['dataset_split'],
                dataset_type=config['dataset']['dataset_type'],
                model_type = config['training']['model_type'],
                TIMESTAMP=TIMESTAMP,
                expert_llm = config['training']['expert_llm'].split("/")[-1],
            )
            save_dir = "/".join(save_path.split("/")[:-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print("Save path: {}".format(save_path))
            
            cache_key_env = (f"d{config['dataset']['dataset_split']}_test_{config['dataset']['dataset_type']}", 
                    f"seed{config['dataset']['dataset_split']}", "eval_vec")
            
            num_processes = 1

            if "mistral" in config['training']['expert_llm'].lower():
                infer_mistral(test_data_path, save_path, cache_key_env,
                        actor_critic, obs_rms, config['training']['seed'], num_processes, device, model_id=config['training']['expert_llm'], norm_obs=config['training']['norm_obs'])
            if "gpt" in config['training']['expert_llm'].lower():
                infer_gpt(test_data_path, save_path, cache_key_env,
                        actor_critic, obs_rms, config['training']['seed'], num_processes, device, model_id=config['training']['expert_llm'], norm_obs=config['training']['norm_obs'])
            else:
                infer_llama(test_data_path, save_path, cache_key_env,
                        actor_critic, obs_rms, config['training']['seed'], num_processes, device, model_id=config['training']['expert_llm'], norm_obs=config['training']['norm_obs'])

if __name__ == "__main__":
    dataset_name = DATASET_NAME
    seed = SEED
    llm_id = LLM_ID
    config = {
        "model": {
        "OBS_DIM": 1024,
        "ACT_DIM": 2
        },
        "training": {
        "expert_llm": llm_id,
        "seed": seed,
        "num_iteration": 1000,
        "num_warmup_iteration": 10,
        "num_steps":200,
        "norm_obs":True,
        "algo": "ppo",
        "model_type": "gail-ppo",
        "gail": True,
        "gail_batch_size": 64,
        "gail_epoch": 5,
        "gail_warm_up_epoch": 10,
        "num_trajectories":50,
        "trajs_ratios":[0.7,0.15,0.15],
        "temp":0.05,
        "lr": 3e-4,
        "eps": 1e-5,
        "alpha": 0.99,
        "gamma": 0.99,
        "use_gae": True,
        "gae_lambda": 0.97,
        "entropy_coef": 0.01,
        "value_loss_coef": 0.5,
        "max_grad_norm": 0.5,
        "cuda": True,
        "cuda_deterministic": False,
        "num_processes": 1,
        "ppo_epoch": 4,
        "num_mini_batch": 4,
        "clip_param": 0.1,
        "save_interval": 1000,
        "use_linear_lr_decay": True,
        "use_proper_time_limits": False
    },
        "dataset": {
        "dataset_type": dataset_name,
        "dataset_split": seed,
        "n_sample": 50
        },
        "paths": {
        "cache_file": "data/d{dataset_split}_train_{dataset_type}_{expert_llm}.pkl",
        "save_dir": "trained_models/{expert_llm}",
        "test_data_vec_path": "data/d{dataset_split}_test_{dataset_type}_vec.pkl",
        "save_path": "results/{model_type}/{dataset_type}/d{dataset_split}_{dataset_type}_{model_type}_{expert_llm}_{TIMESTAMP}.pkl"
        }
    }

    json_formatted_str = json.dumps(config, indent=2)

    print(json_formatted_str)
    
    main(config)