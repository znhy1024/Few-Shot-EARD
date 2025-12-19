import torch
from emoji import demojize
from nltk.tokenize import TweetTokenizer

Tweettokenizer = TweetTokenizer()

def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return token
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token

def normalizeTweet(tweet):
    tokens = Tweettokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())

def get_env_state_vec_batch2(states, tids, embedder, tokenizer, device="cuda", batch_size=16):
      
    def process_batch(batch, embedder, initial_batch_size):
        def embed_minibatch(minibatch):
            inputs = tokenizer(minibatch, max_length=512, padding='longest', truncation=True, return_tensors="pt").to(device)

            with torch.no_grad():
                features = embedder(**inputs)['last_hidden_state'][:,0,:]
            return features.cpu()

        current_batch_size = initial_batch_size
        features = []
        i = 0

        while i < len(batch):
            try:
                minibatch = batch[i:i+current_batch_size]
                minibatch_features = embed_minibatch(minibatch)
                features.append(minibatch_features)
                i += len(minibatch)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    current_batch_size = max(1, current_batch_size // 2)
                    if current_batch_size == 1:
                        raise ValueError(f"Unable to process item at index {i} even with batch size of 1.")
                else:
                    raise
                
        len_feat = sum(f.size(0) for f in features)
        assert len_feat == len(batch), (len_feat, len(batch))            
        return torch.cat(features, dim=0)

    features_stop = []
    features_continue = []
    post_features = {-1:{},-2:{}}
    all_lines = []
    all_tids = []

    for si, (state,tid) in enumerate(zip(states[-1],tids[-1])):
            if tid not in post_features:
                post_features[tid] = {}
                post_features[tid]["text"] = state
    post_features[-1]["text"] = "Stop."
    post_features[-2]["text"] = "Continue."
    for k, post in post_features.items():
        line = post["text"]
        all_lines.append(normalizeTweet(line))
        all_tids.append(k)
    features = []
    for i in range(0, len(all_lines), batch_size):
        batch = all_lines[i:i+batch_size]
        features.append(process_batch(batch, embedder, batch_size))
    features = torch.cat(features, dim=0)
    assert features.shape[0] == len(all_lines) ==len(all_tids)
    for i in range(len(features)):
        if len(features[i].shape) == 1:
            post_features[all_tids[i]]["feats"] = features[i].unsqueeze(0)
        else:
            post_features[all_tids[i]]["feats"] = features[i]
        assert len(post_features[all_tids[i]]["feats"].shape)


    def process_states(states_batch,tids_batch, add_stop=False, add_continue=False):
        all_tids = []
        lengths = []
        for si, (_,tid_seq) in enumerate(zip(states_batch,tids_batch)):
            tids = [t for t in tid_seq]
            if add_stop and si!=0:
                tids.append(-1)
            elif add_continue and si!=0:
                tids.append(-2)
            lengths.append(len(tids))
            all_tids.extend(tids)
        
        features = []
        for i in range(len(all_tids)):
            features.append(post_features[all_tids[i]]["feats"])
        features = torch.cat(features, dim=0)
        
        start = 0
        state_features = []
        for length in lengths:
            state_features.append(features[start:start+length].sum(dim=0))
            start += length
        
        return torch.stack(state_features)
    

    features_stop = process_states(states, tids, add_stop=True).numpy()
    features_continue = process_states(states, tids, add_continue=True).numpy()

    return features_stop, features_continue

def get_state_vec_batch(states, tids, embedder, tokenizer, all_acts=None, device="cuda",batch_size=16):
    
    def process_batch(batch, embedder, initial_batch_size):
        def embed_minibatch(minibatch):
            inputs = tokenizer(minibatch, max_length=512, padding='longest', truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                features = embedder(**inputs)['last_hidden_state'][:,0,:]
            return features.cpu()

        current_batch_size = initial_batch_size
        features = []
        i = 0

        while i < len(batch):
            try:
                minibatch = batch[i:i+current_batch_size]
                minibatch_features = embed_minibatch(minibatch)
                features.append(minibatch_features)
                i += len(minibatch)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    current_batch_size = max(1, current_batch_size // 2)
                    if current_batch_size == 1:
                        raise ValueError(f"Unable to process item at index {i} even with batch size of 1.")
                else:
                    raise
                
        len_feat = sum(f.size(0) for f in features)
        assert len_feat == len(batch), (len_feat, len(batch))            
        return torch.cat(features, dim=0)
    
    post_features = {-1:{},-2:{}}
    all_lines = []
    all_tids = []

    for si, (state,tid) in enumerate(zip(states[-1],tids[-1])):
        if tid not in post_features:
            post_features[tid] = {}
            post_features[tid]["text"] = state
    post_features[-1]["text"] = "Stop."
    post_features[-2]["text"] = "Continue."
    for k, post in post_features.items():
        line = post["text"]
        all_lines.append(normalizeTweet(line))
        all_tids.append(k)
    features = []
    for i in range(0, len(all_lines), batch_size):
        batch = all_lines[i:i+batch_size]
        features.append(process_batch(batch, embedder, batch_size))
    features = torch.cat(features, dim=0)
    assert features.shape[0] == len(all_lines) ==len(all_tids)
    for i in range(len(features)):
        if len(features[i].shape) == 1:
            post_features[all_tids[i]]["feats"] = features[i].unsqueeze(0)
        else:
            post_features[all_tids[i]]["feats"] = features[i]
        assert len(post_features[all_tids[i]]["feats"].shape)

    all_tids = []
    lengths = []
    for si, (state,tid_seq) in enumerate(zip(states,tids)):
        tids = [t for t in tid_seq]
        if si != 0:
            current_pred = int(all_acts[si-1])
            if current_pred == 1:
                tids.append(-1)
            else:
                tids.append(-2)
        lengths.append(len(tids))
        all_tids.extend(tids)
    
    features = []
    for i in range(len(all_tids)):
        features.append(post_features[all_tids[i]]["feats"])
    features = torch.cat(features, dim=0)
    
    start = 0
    state_features = []
    for length in lengths:
        state_features.append(features[start:start+length].sum(dim=0))
        start += length
    
    return torch.stack(state_features).numpy()