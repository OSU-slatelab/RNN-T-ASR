import torch
import math
import random
import json
import ast
import re
import os
import time
import pandas as pd
import pdb
import string
import numpy as np
from transformers import BertTokenizer

TOK = BertTokenizer.from_pretrained("bert-base-uncased")
ASR_ID2TOK = ["<blank>"]+list(string.ascii_lowercase) + [str(x) for x in range(10)] + ["'","-","<space>"]
ASR_TOK2ID = {x:i for i, x in enumerate(ASR_ID2TOK)}

def pad_fac(input, factor=4):
    add_size = input.size(1) % factor
    if add_size != 0:
        rem_size = factor - add_size
        return torch.cat([input, torch.zeros(input.size(0), rem_size, input.size(2))], dim=1)
    else:
        return input

def roll_in(x, lens):
    lens = [math.ceil(1.*k/4) for k in lens]
    x = pad_fac(x, 4)
    full = torch.cat([x, x.roll(-1, dims=1), x.roll(-2, dims=1), x.roll(-3, dims=1)], dim=2)
    extract = list(range(0, x.size(1), 4))
    return full[:, extract, :], lens

def list_batch(X, lens):
    sbatch_ = []
    for i, l in enumerate(lens):
        sbatch_.append(X[i,:l,:])
    return sbatch_

def convert_tok2id(text):
    lst = []
    for ch in text:
        if ch == " ":
            lst.append(ASR_TOK2ID["<space>"])
        else:
            lst.append(ASR_TOK2ID[ch])
    return lst

def convert_id2tok(ids):
    text = ''
    for x in ids:
        if x == ASR_TOK2ID["<space>"]:
            text = text + ' '
        else:
            text = text + ASR_ID2TOK[x]
    return text

def clean4asr(text):
    text = re.sub('[^A-Za-z0-9\s\-\']+','',text)
    return text.lower().strip()

def my_shuffle(x):
    if len(x) == 1:
        raise Exception
    for i in reversed(range(1, len(x))):
        # pick an element in x[:i] with which to exchange x[i]
        j = int(random.random() * i)
        x[i], x[j] = x[j], x[i]

def inject_seqn(X):
    if X.size(0) == 1:
        return X
    mask_sn = torch.ones(X.shape[0],1,1)
    zr = random.sample(list(range(X.shape[0])), int(0.2*X.shape[0]))
    mask_sn[zr] = 0.
    Z = list(range(X.shape[0]))
    my_shuffle(Z)
    X = torch.log(torch.exp(X) + 0.4 * torch.exp(X[Z]) * mask_sn)
    return X

def load_dict(model, ptdict, loc='cuda:0', ddp=True):
    #pretrained_dict = torch.load(dict_path, map_location=loc)
    pretrained_dict = ptdict
    model_dict = model.state_dict()
    new_pt_dict = {}
    for k, v in pretrained_dict.items():
        k_new = k
        if not ddp and k[:7] == 'module.':
            k_new = k[7:]
        elif ddp and k[:7] != 'module.':
            k_new = f'module.{k}'
        new_pt_dict[k_new] = v
    pretrained_dict = {k: v for k, v in new_pt_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def save_pick(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pick(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save(model, path):
    torch.save(model.state_dict(), path)

def load(model, path):
    model.load_state_dict(torch.load(path))

def save_checkpoint(state, filename):
    torch.save(state, filename)
