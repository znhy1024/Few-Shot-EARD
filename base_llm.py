from prompts import construct_prompt_detection, normalize_response, TOKEN_LIMITS


def get_mistral3_pred(model, tokenizer, state_text, model_id):

    tokenizer.pad_token = tokenizer.eos_token

    messages = construct_prompt_detection(state_text, 
                                        tokenizer, token_limit=TOKEN_LIMITS[model_id],
                                        truncate=True, model_id=model_id)

    
    inputs = tokenizer(messages, return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs,  
                            max_new_tokens=3,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            top_p=None,
                            temperature=None)
    response = tokenizer.decode(outputs[0][-3:], skip_special_tokens=True)
    
    pred_label = normalize_response(response)
  
    return pred_label


def get_gpt_pred(model, tokenizer, state_text, model_id):

    messages = construct_prompt_detection(state_text, 
                                        tokenizer, token_limit=TOKEN_LIMITS[model_id],
                                        truncate=True, model_id=model_id)

    
    responses = model.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=5,
        )
    
    response = responses.choices[0].message.content
    
    pred_label = normalize_response(response)
    
    return pred_label


def get_llama3_pred(model, state_text, model_id):

    terminators = [
            model.tokenizer.eos_token_id,
            model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    messages = construct_prompt_detection(state_text,model.tokenizer, token_limit=TOKEN_LIMITS[model_id],truncate=True)

    model_outputs = model(
        messages,
        max_new_tokens=3,
        eos_token_id=terminators,
        do_sample=False,
        pad_token_id=model.tokenizer.eos_token_id,
        top_p=None,
        temperature=None
    )
    
    response = model_outputs[0]["generated_text"][-1]["content"]
    
    pred_label = normalize_response(response)
  
    return pred_label
