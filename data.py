import torch
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
import torchaudio
import torchaudio.transforms as AT
import copy
from util import *
from tqdm import tqdm
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from speechbrain.processing.features import STFT, spectral_magnitude, Filterbank, Deltas, InputNormalization, ContextWindow

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, win_len=25, hop_length=10, n_fft=400, n_mels=80, sample_rate=16000):
        self.df = pd.read_csv(csv_path)
        self.compute_stft = STFT(sample_rate=sample_rate, win_length=win_len, hop_length=hop_length, n_fft=n_fft)
        self.compute_fbanks = Filterbank(n_mels=n_mels)
        self.sr = sample_rate

    def get_filterbanks(self, signal):
        features = self.compute_stft(signal)
        features = spectral_magnitude(features)
        features = self.compute_fbanks(features)
        return features

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        wav, org_sr = torchaudio.load(row['audio_file'])
        if org_sr > self.sr:
            wav = AT.Resample(org_sr, self.sr)(wav)
        return self.get_filterbanks(wav)

class ASRDataset(SpeechDataset):
    def __init__(self, args, csv_path, win_len=25, hop_length=10, n_fft=400, n_mels=80, sample_rate=16000):
        super(ASRDataset, self).__init__(csv_path, win_len, hop_length, n_fft, n_mels, sample_rate)
        self.args = args

    def fix_path(self, path):
        return path.replace('/data/corpora2/librispeech/LibriSpeech/', '/users/PAS1939/vishal/datasets/librispeech/audio/')

    def __getitem__(self, index):
        row = self.df.iloc[index]
        wav, org_sr = torchaudio.load(self.fix_path(row['audio_file']))
        if org_sr > self.sr:
            wav = AT.Resample(org_sr, self.sr)(wav)
        return self.get_filterbanks(wav), clean4asr(row['utterance'])

class CollatorASR(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, lst):
        speechL = [x[0].squeeze(0) for x in lst if x[0].size(1) > 2]
        pack1 = pack_sequence(speechL, enforce_sorted=False)
        speechB, logitLens = pad_packed_sequence(pack1, batch_first=True)
        lmax = speechB.size(1)

        text = [x[1] for x in lst if x[0].size(1) > 2]
        textOut = [torch.tensor(convert_tok2id(x)) for x in text]
        textIn = [torch.cat([torch.tensor([0]), x]) for x in textOut]

        pack2 = pack_sequence(textOut, enforce_sorted=False)
        targetB, targetLens = pad_packed_sequence(pack2, batch_first=True)
        
        pack3 = pack_sequence(textIn, enforce_sorted=False)
        textB, _ = pad_packed_sequence(pack3, batch_first=True)

        return speechB, textB, targetB, logitLens, lmax, targetLens

