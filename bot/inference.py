from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_model(checkpoint):
    model_name = "DeepPavlov/rudialogpt3_medium_based_on_gpt2_v2"   
    tokenizer =  AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
    model.eval()
    model.cuda()
    
    return tokenizer, model

def get_prediction(query, tokenizer, model, params_dict=None):
    # INFERENCE

    chat_history_ids = torch.zeros((1, 0), dtype=torch.int)
    new_user_input_ids = tokenizer.encode(f"|0|1|{query}{tokenizer.eos_token}", return_tensors="pt")
    chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    new_user_input_ids = tokenizer.encode(f"|1|-|", return_tensors="pt")
    chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    input_len = chat_history_ids.shape[-1]

    temperature = 0.5
    if params_dict is not None:
        temperature = params_dict['temperature']
    
    chat_history_ids = model.generate(
        chat_history_ids.cuda(),
        num_return_sequences=1,                     # use for more variants, but have to print [i]
        max_length=512,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=temperature,                          # 0 for greedy
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
        
    decoded = tokenizer.decode(chat_history_ids[0].cpu().detach(), skip_special_tokens=True)\
            .split('|1|-|')[-1].replace('\"', '').replace('|', '')
        
    result = f"{decoded}"
    
    return result
    
# LSTM

import numpy as np
import torch
import string
import sys

# !git clone https://github.com/dariush-bahrami/character-tokenizer.git
from charactertokenizer import CharacterTokenizer

from lstm_class import *

def get_lstm_model(checkpoint):
    with open('tokenizer_chars.txt', 'r') as fin:
        chars = fin.read()

    model_max_length = 512
    tokenizer = CharacterTokenizer(chars, model_max_length, padding_size='right')
    
    model = LSTM(tokenizer.vocab_size)
    model = model.load_from_checkpoint(checkpoint_path=checkpoint, \
                                       n_vocab=tokenizer.vocab_size)
    model.eval()
    model.cuda()
    
    return tokenizer, model

def get_lstm_prediction(text, tokenizer, model, params_dict=None):
    softmax = nn.Softmax(dim=1)
    pattern = torch.tensor(tokenizer.encode(text))[:-1]
    result = text
    
    k = 2
    if params_dict is not None:
        k = params_dict['k']
    
    with torch.no_grad():
        for _ in range(1000):
            x = pattern / float(tokenizer.vocab_size)
            prediction = model(x.unsqueeze(0).cuda().float())
            p = softmax(prediction * k).cpu().flatten().detach().numpy()
            index = np.random.choice(np.arange(len(prediction.flatten())), 1, p=p)
            result += tokenizer.decode(index)
            if index == 1:
                break
            pattern = torch.concatenate((pattern.cuda(), torch.tensor(index).cuda()))
            pattern = pattern[1:]
  
    return result
    
    
   