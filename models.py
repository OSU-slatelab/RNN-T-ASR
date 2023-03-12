import pdb
import time
import numpy as np
import random
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from transformers import BertForMaskedLM, BertTokenizer
from tqdm import tqdm
from util import *
from encoders import *

NEG = -10000000
TOK_NC = BertTokenizer.from_pretrained("bert-base-uncased")

def logsumexp(a, b):
    return np.log(np.exp(a) + np.exp(b))

def get_mask(lens):
    mask = torch.ones(len(lens), max(lens))
    for i, l in enumerate(lens):
        mask[i][:l] = 0.
    return mask

def extract(tens, mask, offset=0):
    out_lens = (1-mask).sum(dim=1).tolist()
    out = []
    for i, ten in enumerate(tens):
        out.append(ten[:out_lens[i]-offset])
    return torch.cat(out, dim=0)

class Attention(nn.Module):
    def __init__(self, input_dim, nhead, dim_feedforward=2048, dropout=0.1):
        super(Attention, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout) 

    def forward(self, Q, K, mask):
        src, attn = self.self_attn(Q, K, K, key_padding_mask=mask)
        ## Add and norm
        src = Q + self.dropout(src)
        src = self.norm1(src)
        ## MLP
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        ## Add and norm
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src, attn

class BERTNC(nn.Module):
    def __init__(self):
        super(BERTNC, self).__init__()
        self.encoder = BertForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True).bert.embeddings
        
    def forward(self, inputs):
        return self.encoder(inputs.input_ids).permute(1,0,2), 1. - inputs.attention_mask.float()

class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        model = BertForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.encoder = model.bert
        
    def forward(self, inputs):
        output = self.encoder(**inputs)
        return output.last_hidden_state.permute(1,0,2), 1. - inputs.attention_mask

    def forward_full(self, inputs):
        output = self.encoder(**inputs)
        return output.hidden_states, 1. - inputs.attention_mask

class RNNT(nn.Module):
    def __init__(self, args):
        super(RNNT, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)

        if args.enc_type == 'lstm':
            if not args.unidirectional:
                self.tNet = LstmEncoder(args.n_layer, args.in_dim, int(args.hid_tr/2), dropout0=args.dropout, dropout=args.dropout, spec=args.deep_spec, bidirectional=True)
            else:
                self.tNet = LstmEncoder(args.n_layer, args.in_dim, args.hid_tr, dropout0=args.dropout, dropout=args.dropout, spec=args.deep_spec, bidirectional=False)
        else:
            self.inScale = nn.Linear(args.in_dim, args.hid_tr)
            self.tNet = ConformerEncoder(args.n_layer, args.hid_tr, args.head_dim, args.nhead, dropout=args.dropout, spec=args.deep_spec)
        self.bottle = nn.Linear(args.hid_tr, 768)
        self.prEmb = nn.Embedding(args.vocab_size, 10)
        self.pNet = LstmEncoder(1, 10, args.hid_pr, dropout0=0.01, dropout=args.dropout, spec=args.deep_spec, bidirectional=False)

        self.projTr = nn.Linear(768, 256)
        self.projPr = nn.Linear(args.hid_pr, 256)
    
        self.clsProj = nn.Linear(256, args.vocab_size)

    def forward_tr(self, x): # x is a list of sequences
        #pack = pack_sequence(x, enforce_sorted=False)
        #x, lens = pad_packed_sequence(pack, batch_first=True)
        if self.args.enc_type != 'lstm':
            x = self.inScale(x)
        x, _ = self.tNet(x)
        x = self.bottle(self.dropout(x))
        return x

    def forward_pr(self, x): # x --> bsz, max_seq_len
        x = self.prEmb(x)
        x, _ = self.pNet(x)
        return x

    def forward_jnt(self, x, y):
        x = self.projTr(self.dropout(x)).unsqueeze(2)
        y = self.projPr(self.dropout(y)).unsqueeze(1)
        z = x + y
        return self.clsProj(torch.tanh(z))

    def forward(self, speech, tokens):
        x = self.forward_tr(speech)
        y = self.forward_pr(tokens)
        pred = self.forward_jnt(x, y)
        return pred

    def beam_search(self, speech, beam_size=1):
        raise NotImplementedError
