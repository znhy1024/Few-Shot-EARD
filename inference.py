import torch
import pickle
import tiktoken
from openai import OpenAI

from a2c_ppo_acktr.envs import VecNormalize

from env import ENV_EVAL

from base_llm import get_llama3_pred, get_mistral3_pred, get_gpt_pred
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

OBS_DIM = 1024
ACT_DIM = 2
HF_TOKEN = ""
openai_api_key = ""


def infer_llama(data_path, save_path, cache_key_env, actor_critic, obs_rms, seed, num_processes, device, model_id="", norm_obs=True):

    prediction_results = {}
            
    try:
        model = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},
            device="cuda",
        )
    except:
        from huggingface_hub import snapshot_download
        snapshot_download(
                model_id,
                repo_type="model",
                ignore_patterns="*/*",
                token=HF_TOKEN,
        )
        model = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},
            device="cuda",
        )

    actor_critic = actor_critic.to(device)

    vec_norm = VecNormalize(observation_space_shape=(OBS_DIM,),norm_obs=norm_obs) 
    
    eval_envs = ENV_EVAL(vec_norm, seed, data_path, cache_key_env)

    with torch.no_grad():
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.obs_rms = obs_rms
        
        eval_ids = eval_envs.trajectories_ids

      
        for trajectory_id in eval_ids:

            obs = eval_envs.reset(trajectory_id).to(device).unsqueeze(0)
            eval_recurrent_hidden_states = torch.zeros(
                num_processes, actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(num_processes, 1, device=device)

            while True:
                
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs.to(device),
                        eval_recurrent_hidden_states.to(device),
                        eval_masks.to(device),
                        deterministic=True)
                

                current_obs_text, obs, _, done, _ = eval_envs.step(action)
                obs = obs.unsqueeze(0)

                eval_masks = torch.tensor(
                                [[0.0] if done_ else [1.0] for done_ in done],
                                dtype=torch.float32,
                                device=device)

                if action[0][0].item() == 1 or done[0]:
                    break

            label = eval_envs.trajectories[trajectory_id]["label"]
            pred_label = get_llama3_pred(model, current_obs_text, model_id)

            prediction_results[trajectory_id] = {
                "cid": trajectory_id,
                "label": label,
                "pred_label": pred_label,
                "obs_text": current_obs_text
            }

    with open(save_path, 'wb') as f:
        pickle.dump(prediction_results, f)

    del model
    del eval_envs


def infer_mistral(data_path, save_path, cache_key_env, actor_critic, obs_rms, seed, num_processes,
             device, model_id="", norm_obs=True):
    
    prediction_results = {}

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    except:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, 
                      token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    
    actor_critic = actor_critic.to(device)

    vec_norm = VecNormalize(observation_space_shape=(OBS_DIM,),norm_obs=norm_obs) 
    
    eval_envs = ENV_EVAL(vec_norm, seed, data_path, cache_key_env)

    with torch.no_grad():
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.obs_rms = obs_rms
        
        eval_ids = eval_envs.trajectories_ids

        for trajectory_id in eval_ids:

            obs = eval_envs.reset(trajectory_id).to(device).unsqueeze(0)
            eval_recurrent_hidden_states = torch.zeros(
                num_processes, actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(num_processes, 1, device=device)

            while True:
                
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs.to(device),
                        eval_recurrent_hidden_states.to(device),
                        eval_masks.to(device),
                        deterministic=True)

                current_obs_text, obs, _, done, _ = eval_envs.step(action)
                obs = obs.unsqueeze(0) 

                eval_masks = torch.tensor(
                                [[0.0] if done_ else [1.0] for done_ in done],
                                dtype=torch.float32,
                                device=device)

                if action[0][0].item() == 1 or done[0]:
                    break

            label = eval_envs.trajectories[trajectory_id]["label"]
            pred_label = get_mistral3_pred(model,tokenizer, current_obs_text, model_id)

            prediction_results[trajectory_id] = {
                "cid": trajectory_id,
                "label": label,
                "pred_label": pred_label,
                "obs_text": current_obs_text
            }

    with open(save_path, 'wb') as f:
        pickle.dump(prediction_results, f)

    del model
    del eval_envs


def infer_gpt(data_path, save_path, cache_key_env, actor_critic, obs_rms, seed, num_processes,
             device, model_id="", norm_obs=True):
    
    prediction_results = {}
            
    model = OpenAI(api_key=openai_api_key)   
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    actor_critic = actor_critic.to(device)

    vec_norm = VecNormalize(observation_space_shape=(OBS_DIM,),norm_obs=norm_obs) 
    
    eval_envs = ENV_EVAL(vec_norm, seed, data_path, cache_key_env)

    with torch.no_grad():
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.obs_rms = obs_rms
        
        eval_ids = eval_envs.trajectories_ids

        for trajectory_id in eval_ids:

            obs = eval_envs.reset(trajectory_id).to(device).unsqueeze(0)
            eval_recurrent_hidden_states = torch.zeros(
                num_processes, actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(num_processes, 1, device=device)

            while True:
                
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs.to(device),
                        eval_recurrent_hidden_states.to(device),
                        eval_masks.to(device),
                        deterministic=True)

                current_obs_text, obs, _, done, _ = eval_envs.step(action)
                obs = obs.unsqueeze(0) 

                eval_masks = torch.tensor(
                                [[0.0] if done_ else [1.0] for done_ in done],
                                dtype=torch.float32,
                                device=device)

                if action[0][0].item() == 1 or done[0]:
                    break

            label = eval_envs.trajectories[trajectory_id]["label"] 
            pred_label = get_gpt_pred(model, tokenizer, current_obs_text, model_id)

            prediction_results[trajectory_id] = {
                "cid": trajectory_id,
                "label": label,
                "pred_label": pred_label,
                "obs_text": current_obs_text
            }

    with open(save_path, 'wb') as f:
        pickle.dump(prediction_results, f)

    del model
    del eval_envs