import os
import json
import copy
import torch
import pickle
import numpy as np

from data_utils import set_seed, timeline_convert_merge_post
from prompts import construct_prompt_detection, TOKEN_LIMITS, normalize_response
from enc_state import get_state_vec_batch, get_env_state_vec_batch2

from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForCausalLM, logging
logging.set_verbosity_error()

from openai import OpenAI
import tiktoken

HF_TOKEN = ""
openai_api_key =""

from typing import (
    Any,
    List,
    Mapping
)

def count_flippings(input_list):
    if not input_list:
        return []
    
    flip_counts = [0] 
    current_value = input_list[0]
    total_flips = 0
    
    for i in range(1, len(input_list)):
        if input_list[i] != current_value:
            total_flips += 1
            current_value = input_list[i]
        flip_counts.append(total_flips)
    
    return flip_counts

def generate_counts_indicator(input_list):
    if not input_list:
        return []

    n = len(input_list)
    result = [0] * n
    last_change_index = 0

    for i in range(n - 1, 0, -1):
        if input_list[i] != input_list[i-1]:
            last_change_index = i
            break

    for i in range(last_change_index, n):
        result[i] = 1

    return result

def get_actions(input_list):
    flip_cnt = count_flippings(input_list)
    actions = generate_counts_indicator(flip_cnt)
    return actions

def find_first_one(lst):
    try:
        return lst.index(1)
    except ValueError:
        return -1 
    
def find_first_label(lst, label): 
    if label not in ["0", "1"]:
        raise ValueError("Label must be 0 or 1")
    
    try:
        return lst.index(label)
    except ValueError:
        return -1 


def get_llama3_actions_batch(model, sub_seqs, sub_tids, max_step, model_id, batch_size = 2):

    terminators = [
            model.tokenizer.eos_token_id,
            model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    answer_seqs = []
    prefix_seqs = []
    states = []

    prefix_tids = []
    original_tids = []

    all_messages = []

    for seq,tid_seq in zip(sub_seqs[:max_step],sub_tids[:max_step]):
        prefix_seqs.extend(seq)
        all_messages.append(prefix_seqs.copy())
        states.append(prefix_seqs.copy())
        prefix_tids.extend(tid_seq)
        original_tids.append(prefix_tids.copy())
    
    current_batch_size = batch_size
    outputs_list = []
    i = 0

    while i < len(all_messages):
        minibatch = all_messages[i:i+current_batch_size]
        batch_messages = [construct_prompt_detection(prefix_seqs, model.tokenizer, token_limit=TOKEN_LIMITS[model_id],truncate=True) for prefix_seqs in minibatch]
        batch_outputs = model(
                batch_messages,
                max_new_tokens=3,
                eos_token_id=terminators,
                do_sample=False,
                pad_token_id=model.tokenizer.eos_token_id,
                top_p=None,
                temperature=None
            )
        outputs_list.extend(batch_outputs)
        i += len(minibatch)
                  
    assert len(outputs_list) == len(all_messages), (len(outputs_list), len(all_messages))  

    for j, outputs in enumerate(outputs_list):
        response = outputs[0]["generated_text"][-1]["content"]
        pred_label = normalize_response(response)

        answer_seqs.append(pred_label)

    return answer_seqs, states, original_tids


def get_mistral_actions_batch(model, tokenizer, sub_seqs, sub_tids, max_step, model_id, batch_size = 2):

    tokenizer.pad_token = tokenizer.eos_token

    answer_seqs = []
    prefix_seqs = []
    states = []

    prefix_tids = []
    original_tids = []

    all_messages = []

    for seq,tid_seq in zip(sub_seqs[:max_step],sub_tids[:max_step]):
        prefix_seqs.extend(seq)
        all_messages.append(prefix_seqs.copy())
        states.append(prefix_seqs.copy())
        prefix_tids.extend(tid_seq)
        original_tids.append(prefix_tids.copy())
    
    current_batch_size = batch_size
    outputs_list = []
    i = 0

    while i < len(all_messages):
        minibatch = all_messages[i:i+current_batch_size]
        batch_messages = [construct_prompt_detection( prefix_seqs, tokenizer, token_limit=TOKEN_LIMITS[model_id],truncate=True,model_id=model_id) for prefix_seqs in minibatch]
        inputs = tokenizer(batch_messages, padding='longest', truncation=True, return_tensors="pt").to(model.device)
        batch_outputs = model.generate(**inputs, 
                                    max_new_tokens=3, 
                                    do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id,
                                    top_p=None,temperature=None)

        outputs_list.extend(batch_outputs)
        i += len(minibatch)
                   
    assert len(outputs_list) == len(all_messages), (len(outputs_list), len(all_messages))  

    for j, outputs in enumerate(outputs_list):
        response = tokenizer.decode(outputs[-3:], skip_special_tokens=True)
        pred_label = normalize_response(response)

        answer_seqs.append(pred_label)
    
    return answer_seqs, states, original_tids


def get_gpt_actions_batch(model, tokenizer, sub_seqs, sub_tids, max_step, model_id):

    answer_seqs = []
    prefix_seqs = []
    states = []

    prefix_tids = []
    original_tids = []

    all_messages = []

    for seq,tid_seq in zip(sub_seqs[:max_step],sub_tids[:max_step]):
        prefix_seqs.extend(seq)
        all_messages.append(prefix_seqs.copy())
        states.append(prefix_seqs.copy())
        prefix_tids.extend(tid_seq)
        original_tids.append(prefix_tids.copy())
    
    for mi, prefix_seqs in enumerate(all_messages):
        input_message = construct_prompt_detection(prefix_seqs, tokenizer, 
                                                token_limit=TOKEN_LIMITS[model_id],truncate=True,model_id=model_id)

        responses = model.chat.completions.create(
            model=model_id,
            messages=input_message,
            max_tokens=5,
        )
    
        response = responses.choices[0].message.content
        
        pred_label = normalize_response(response)
        
        answer_seqs.append(pred_label)
    
    return answer_seqs, states, original_tids


def get_llama3_actions(expert_data, cache_file, cached_results, cache_key_action, model_id=""):

    if cache_key_action in cached_results and "update_traj" in cached_results[cache_key_action]:
        update_traj = cached_results[cache_key_action]["update_traj"]
    else:
        update_traj = {}

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

    for ti, traj in enumerate(expert_data):
        value = traj

        if value["cid"] in update_traj:
            continue    


        timelines = value["seq"]["timeline"]
        seqs_list = value["seq"]["texts"]
        label = value["label"]

        tids = value["seq"]["tids"]

        _, sub_seqs, sub_tids = timeline_convert_merge_post(timelines, seqs_list, tids, 1)


        assert len(sub_seqs) != 0

        if len(sub_seqs) < 3: # skip too short trajs
            continue

        answer_seqs, states,original_tids = get_llama3_actions_batch(model,copy.deepcopy(sub_seqs),sub_tids, 500, model_id)       

        traj["pred_seqs"] = answer_seqs
        actions = get_actions(answer_seqs)
        traj["actions"] = actions
        traj["states"] = states
        traj["tids"] = original_tids

        update_traj[traj["cid"]] = traj

        cached_results[cache_key_action] = update_traj
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_results, f)

    
    cached_results[cache_key_action] = update_traj
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_results, f)

    save_data = {
        "update_traj": update_traj,
    }

    cached_results[cache_key_action] = save_data
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_results, f)
  
    return update_traj


def get_mistral_actions(expert_data, cache_file, cached_results, cache_key_action, model_id):

    if cache_key_action in cached_results and "update_traj" in cached_results[cache_key_action]:
        update_traj = cached_results[cache_key_action]["update_traj"]
    else:
        update_traj = {}

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    except:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, 
                      token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    for ti, traj in enumerate(expert_data):
        value = traj
        if value["cid"] in update_traj:
            continue    

        timelines = value["seq"]["timeline"]
        seqs_list = value["seq"]["texts"]
        label = value["label"]

        tids = value["seq"]["tids"]

        _, sub_seqs, sub_tids = timeline_convert_merge_post(timelines, seqs_list, tids, 1)


        assert len(sub_seqs) != 0

        if len(sub_seqs) < 3:
            continue

        answer_seqs, states,original_tids = get_mistral_actions_batch(model,tokenizer, copy.deepcopy(sub_seqs),sub_tids, 500,model_id)          

        traj["pred_seqs"] = answer_seqs
        actions = get_actions(answer_seqs)
        traj["actions"] = actions
        traj["states"] = states
        traj["tids"] = original_tids

        update_traj[traj["cid"]] = traj

        cached_results[cache_key_action] = update_traj
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_results, f)
        

    cached_results[cache_key_action] = update_traj
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_results, f)

    save_data = {
        "update_traj": update_traj
    }

    cached_results[cache_key_action] = save_data
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_results, f)
  
    return update_traj


def get_gpt_actions(expert_data, cache_file, cached_results, cache_key_action, model_id=""):

    if cache_key_action in cached_results and "update_traj" in cached_results[cache_key_action]:
        update_traj = cached_results[cache_key_action]["update_traj"]
    else:
        update_traj = {}
    
    
    tokenizer = tiktoken.get_encoding("cl100k_base")

    model = OpenAI(api_key=openai_api_key)

    for ti, traj in enumerate(expert_data):

        value = traj
        if value["cid"] in update_traj:
            continue    

        timelines = value["seq"]["timeline"]
        seqs_list = value["seq"]["texts"]
        label = value["label"]

        tids = value["seq"]["tids"]

        _, sub_seqs, sub_tids = timeline_convert_merge_post(timelines, seqs_list, tids, 1)


        assert len(sub_seqs) != 0

        if len(sub_seqs) < 3:
            continue

        answer_seqs, states,original_tids = get_gpt_actions_batch(model,tokenizer, copy.deepcopy(sub_seqs),sub_tids, 500,model_id)              

        traj["pred_seqs"] = answer_seqs
        actions = get_actions(answer_seqs)
        traj["actions"] = actions
        traj["states"] = states
        traj["tids"] = original_tids

        update_traj[traj["cid"]] = traj

        cached_results[cache_key_action] = update_traj
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_results, f)

    cached_results[cache_key_action] = update_traj
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_results, f)


    save_data = {
        "update_traj": update_traj
    }

    cached_results[cache_key_action] = save_data
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_results, f)
  
    return update_traj


def get_state_emb_eval(embedder, tokenizer, data_path, save_path, cache_key):

    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            cached_results = pickle.load(f)
    else:
        cached_results = {}
    
    if cache_key in cached_results:
        prediction_results = cached_results[cache_key]
    else:
        prediction_results = {}
            
    embedder = embedder.to("cuda")

    all_data = json.load(open(data_path))

    for k, value in all_data.items():
        if k in prediction_results:
            continue

        value["cid"] = k
        timelines = value["seq"]["timeline"]
        seqs_list = value["seq"]["texts"]
        tids = value["seq"]["tids"]

        sub_times, sub_seqs, sub_tids = timeline_convert_merge_post(timelines, seqs_list, tids, 1)

        assert len(sub_seqs) != 0


        prefix_seqs = []
        original_states = []
        prefix_tids = []
        original_tids = []
        prefix_times = []
        original_times = []
        for seq,tid_seq,time_seq in zip(sub_seqs[:1000],sub_tids[:1000],sub_times[:1000]):
            prefix_seqs.extend(seq)
            original_states.append(prefix_seqs.copy())
            prefix_tids.extend(tid_seq)
            original_tids.append(prefix_tids.copy())
            prefix_times.extend(time_seq)
            original_times.append(prefix_times.copy())

        with torch.no_grad():
            env_states_vec_stop, env_states_vec_con = get_env_state_vec_batch2(original_states, original_tids, embedder, tokenizer)

        all_data[k]["obs_stop"] = env_states_vec_stop
        all_data[k]["obs_continue"] = env_states_vec_con
        all_data[k]["states"] = original_states
        all_data[k]["timestamps"] = original_times
        prediction_results[k] = copy.deepcopy(all_data[k])


    cached_results[cache_key] = prediction_results
    with open(save_path, 'wb') as f:
        pickle.dump(cached_results, f)
         

def get_exp_trajs(expert_file_path, env_file_path, cache_file, embedder, tokenizer, n_sample, seed, model_id):

    cache_key = (expert_file_path.split("/")[-1].split(".")[0], f"seed{seed}", f"n_sample{n_sample}")
    
    exp_cache_key = cache_key + ("expert",)
    env_cache_key = cache_key + ("env",)

    expert_data = []
    for k, v in json.load(open(expert_file_path)).items():
        v["cid"] = k
        expert_data.append(v)
    
    env_data = []
    for k, v in json.load(open(env_file_path)).items():
        v["cid"] = k
        env_data.append(v)

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
                cached_results = pickle.load(f)
    else:
        cached_results = {}
    cached_results[exp_cache_key] = expert_data
    cached_results[env_cache_key] = env_data

    llm_id = model_id

    cache_key_action = exp_cache_key + ("action",)

    if "llama" in llm_id:
        update_traj1 = get_llama3_actions(
            copy.deepcopy(expert_data), cache_file, cached_results, cache_key_action,
            model_id=llm_id
        )
    if "mistral" in llm_id:
        update_traj1 = get_mistral_actions(
            copy.deepcopy(expert_data), cache_file, cached_results, cache_key_action,
            model_id=llm_id
        )
    if "gpt" in llm_id:
        update_traj1 = get_gpt_actions(
            copy.deepcopy(expert_data), cache_file, cached_results, cache_key_action, 
            model_id=llm_id
        )

    update_traj = copy.deepcopy(update_traj1)

    cache_key_vec = exp_cache_key + ("vec",)
    
    if cache_key_vec in cached_results:
        vec_trajs = cached_results[cache_key_vec]
    else:
        vec_trajs = {}
    
    embedder = embedder.to("cuda")

    for k, v in update_traj.items():
        if k in vec_trajs:
            continue
        acts = v["actions"]
        states = v["states"]
        label = v["label"]

        all_seq = v["seq"]['texts']
        all_preds = v['pred_seqs']
        all_tids = v['tids']

        sub_trajs = {
            f"{k}_incorrect": None,
            f"{k}_venture": None,
            f"{k}_strict": None
        }

        stop_idx = find_first_label(all_preds, label)

        if stop_idx == -1:
            trunc_idx = len(acts) + 3
            states_tmp = copy.deepcopy(states[:trunc_idx+1])
            acts_tmp = copy.deepcopy(acts[:trunc_idx+1])
            tids_tmp = copy.deepcopy(all_tids[:trunc_idx+1])

            states_vec = get_state_vec_batch(states_tmp, tids_tmp, embedder, tokenizer, acts_tmp[1:])
            sub_trajs[f"{k}_incorrect"] = {
                "obs": copy.deepcopy(states_vec),
                "acts": acts_tmp[1:],
                "rews": [0.0 for _ in acts_tmp[1:]],
                "infos": [{"label":label,"all_len":len(all_seq)} for _ in all_preds]
            }
        else:
            if all_preds[-1] == label:
                trunc_idx = len(acts) + 3
                states_tmp = copy.deepcopy(states[:trunc_idx+1])
                acts_tmp = copy.deepcopy(acts[:trunc_idx+1])
                tids_tmp = copy.deepcopy(all_tids[:trunc_idx+1])
                states_vec = get_state_vec_batch(states_tmp, tids_tmp, embedder, tokenizer, acts_tmp[1:])
                sub_trajs[f"{k}_strict"] = {
                    "obs": copy.deepcopy(states_vec),
                    "acts": acts_tmp[1:],
                    "rews": [0.0 for _ in acts_tmp[1:]],
                    "infos": [{"label":label,"all_len":len(all_seq)} for _ in all_preds]
                }
            else:
                trunc_idx = len(acts) + 3
                states_tmp = copy.deepcopy(states[:trunc_idx+1])
                acts_tmp = copy.deepcopy(acts[:trunc_idx+1])
                tids_tmp = copy.deepcopy(all_tids[:trunc_idx+1])
                states_vec = get_state_vec_batch(states_tmp, tids_tmp, embedder, tokenizer, acts_tmp[1:])
                sub_trajs[f"{k}_incorrect"] = {
                    "obs": copy.deepcopy(states_vec),
                    "acts": acts_tmp[1:],
                    "rews": [0.0 for _ in acts_tmp[1:]],
                    "infos": [{"label":label,"all_len":len(all_seq)} for _ in all_preds]
                }


            strict_stop_idx = find_first_one(acts)
            if strict_stop_idx != stop_idx:
                tmp_acts = [0 for _ in range(stop_idx)] + [1 for _ in range(len(acts) - stop_idx)]
                trunc_idx = len(tmp_acts) + 3
                tmp_states = copy.deepcopy(states[:trunc_idx+1])
                tids_tmp = copy.deepcopy(all_tids[:trunc_idx+1])
                states_vec = get_state_vec_batch(tmp_states, tids_tmp, embedder, tokenizer, tmp_acts[1:])

                sub_trajs[f"{k}_venture"] = {
                    "obs": copy.deepcopy(states_vec),
                    "acts": tmp_acts[:trunc_idx+1][1:],
                    "rews": [0.0 for _ in tmp_acts[:trunc_idx+1][1:]],
                    "infos": [{"label":label,"all_len":len(all_seq)} for _ in all_preds]
                }
        
        for k_traj in sub_trajs:
            if sub_trajs[k_traj]:
                if len(sub_trajs[k_traj]["acts"]) > 1:
                    vec_trajs[k_traj] = sub_trajs[k_traj]

    cached_results[cache_key_vec] = vec_trajs
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_results, f)
    
    cache_key_env_vec = env_cache_key + ("vec",)
    env_traj = copy.deepcopy(env_data)

    if cache_key_env_vec in cached_results:
        env_vec_trajs = cached_results[cache_key_env_vec]
    else:
        env_vec_trajs = {}


    for ti, traj in enumerate(env_traj):

        value = traj
        k = value["cid"]

        if value["cid"] in env_vec_trajs:
            continue    

        timelines = value["seq"]["timeline"]
        seqs_list = value["seq"]["texts"]
        label = value["label"]

        tids = value["seq"]["tids"]

        _, sub_seqs, sub_tids = timeline_convert_merge_post(timelines, seqs_list, tids, 1)

        assert len(sub_seqs) != 0


        if len(sub_seqs) < 3:
            continue

        prefix_seqs = []
        original_states = []
        prefix_tids = []
        original_tids = []
        for seq,tid_seq in zip(sub_seqs[:1000],sub_tids[:1000]):
            prefix_seqs.extend(seq)
            original_states.append(prefix_seqs.copy())
            prefix_tids.extend(tid_seq)
            original_tids.append(prefix_tids.copy())

        with torch.no_grad():
            env_states_vec_stop, env_states_vec_con = get_env_state_vec_batch2(original_states, original_tids, embedder, tokenizer)
        
        env_vec_trajs[k] = copy.deepcopy(value)
        env_vec_trajs[k]["obs_stop"] = env_states_vec_stop
        env_vec_trajs[k]["obs_continue"] = env_states_vec_con
        env_vec_trajs[k]["states"] = original_states

        cached_results[cache_key_env_vec] = env_vec_trajs
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_results, f)

    
    cached_results[cache_key_env_vec] = env_vec_trajs
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_results, f)
    
    cache_key_format = cache_key_vec[:-1] + ("formatted",)
    
    keys = ["obs", "next_obs", "acts", "dones", "lengths", "infos"]
    parts: Mapping[str, List[Any]] = {key: [] for key in keys}

    for ti, traj in vec_trajs.items():

        parts["acts"].append(np.array(traj["acts"]))

        obs = traj["obs"]
        parts["obs"].append(obs[:-1])
        parts["next_obs"].append(obs[1:])

        dones = np.zeros(len(traj["acts"]), dtype=bool)
        dones[-1] = True
        parts["dones"].append(dones)

        if traj['infos'] is None:
            infos = np.array([{}] * len(traj))
        else:
            infos = []
            for tinfo in traj['infos']:
                tinfo["cid"] = ti
                infos.append(tinfo)
        parts["infos"].append(infos[:-1])
        parts["lengths"].append(len(traj["acts"]))

        assert len(traj["acts"]) == len(obs[:-1])
    
    save_data = {
        "trajectories": parts,
    }
    
    cached_results[cache_key_format] = save_data
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_results, f)
    
    return save_data


if __name__ == "__main__":

    dataset_type = DATASET_NAME
    model_id = LLM_ID
    seed = SEED

    set_seed(seed)
        
    n_sample = 50
    emb_model = "vinai/bertweet-large"
    embedder = AutoModel.from_pretrained(emb_model)
    tokenizer = AutoTokenizer.from_pretrained(emb_model)

    expert_file_path = f"data/{dataset_type}/d{seed}_train_expert_{dataset_type}.json"
    env_file_path = f"data/{dataset_type}/d{seed}_train_env_{dataset_type}.json"
    model_name = model_id.split("/")[-1]
    cache_file = f"data/d{seed}_train_{dataset_type}_{model_name}.pkl"

    get_exp_trajs(expert_file_path, env_file_path, cache_file, embedder, tokenizer, n_sample=n_sample, seed=seed, model_id=model_id)


    file_path = f"data/{dataset_type}/d{seed}_test_{dataset_type}.json"
    cache_file = f"data/d{seed}_test_{dataset_type}_vec.pkl"
    cache_key = (f"d{seed}_test_{dataset_type}", f"seed{seed}", "eval_vec")
    get_state_emb_eval(embedder, tokenizer, file_path, cache_file, cache_key)

    print("Done")

