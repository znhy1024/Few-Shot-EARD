import torch
import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def timeline_convert_merge_post(timeLine, texts, tids, interval=10):
    if len(timeLine) != len(texts):
        raise ValueError("timeLine and texts must have the same length")
    
    merge_index = list(range(len(timeLine)))[0::interval]
    merge_texts,merge_times,merge_tids = [],[],[]
    for i,index in enumerate(merge_index):
        try:
            next_index = merge_index[i+1]
        except:
            next_index = index+len(timeLine)+2
        assert next_index != index
        merge_text = [x for x in texts[index:next_index]]
        merge_time = [x for x in timeLine[index:next_index]]
        merge_tid = [x for x in tids[index:next_index]]

        merge_texts.append(merge_text)
        merge_times.append(merge_time)
        merge_tids.append(merge_tid)

    return merge_times, merge_texts, merge_tids



    