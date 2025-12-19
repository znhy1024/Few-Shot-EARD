
TOKEN_LIMITS = {
    "meta-llama/Meta-Llama-3-8B-Instruct": 8000,
    "mistralai/Mistral-7B-Instruct-v0.3": 8000,
    "gpt-4o-mini": 128000,
}

def cap_len(text,tokenizer,token_limit,model_id="llama"):

    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)

    if "llama" in model_id.lower():
        new_text = tokenizer.decode(tokens[-token_limit:][1:])
    if "mistral" in model_id.lower():
        new_text = tokenizer.decode(tokens[-token_limit:])
    if "gpt" in model_id.lower():
        new_text = tokenizer.decode(tokens[-token_limit:])
    return new_text, num_tokens

def normalize_response(response):
    response = response.lower().strip()
    
    positive_indicators = ['yes', 'y', 'true', '1']
    negative_indicators = ['no', 'n', 'false', '0']
    
    if any(indicator in response for indicator in positive_indicators):
        return '1'
    elif any(indicator in response for indicator in negative_indicators):
        return '0'
    else:
        return 'Unknown'

def construct_prompt_detection(seqs, tokenizer=None,token_limit = 8000, truncate=False, model_id="",return_token_len=False):
    posts = []
    
    for pi,post in enumerate(seqs):
        posts.append(f"Post {pi+1}: {post}\n")


    posts.append(f"Post 1: {seqs[0]}\nAnalyze the given sequence of social media posts, determine if it is a rumor. Respond Yes or No only.")

    posts = "".join(posts)

    if truncate:
        cap_message, token_len = cap_len(posts,tokenizer,token_limit,model_id)
    else:
        cap_message = posts
        
    message = f"Analyze the given sequence of social media posts, determine if it is a rumor. Respond Yes or No only.\n\nPosts:\n{cap_message}\n\n"
    if "llama" in model_id.lower():
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": f"{message}"},
        ]
    if "mistral" in model_id.lower():
        conversation =  [{"role": "user", "content": f"{message}"}]
        messages = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True)
    if "gpt" in model_id.lower():
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": f"{message}"}
            ]
    if return_token_len:
        return messages, token_len
    else:
        return messages

