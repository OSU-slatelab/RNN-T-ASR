from util import *
from models import *
from train import *
from data import *
from logging.handlers import RotatingFileHandler
from tokenizers import Tokenizer
from speechbrain.processing.features import InputNormalization
import torch.nn as nn
import torch
import pdb
import logging
import copy
import argparse
import time
import random
import sys
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

class CollatorDec(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, lst):
        speechL = [x[0].squeeze(0) for x in lst if x[0].size(1) > 2]
        pack1 = pack_sequence(speechL, enforce_sorted=False)
        speechB, logitLens = pad_packed_sequence(pack1, batch_first=True)
        lmax = speechB.size(1)

        text = [x[1] for x in lst if x[0].size(1) > 2]

        return speechB, text, logitLens, lmax

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0, help='')
    parser.add_argument('--fmask', type=int, default=27, help='for specaug') #15
    parser.add_argument('--nspeech-feat', type=int, default=80, help='# logmels')
    parser.add_argument('--sample-rate', type=int, default=16000, help='speech sampling rate to use')
    parser.add_argument('--n-layer', type=int, default=6, help='')
    parser.add_argument('--in-dim', type=int, default=320, help='')
    parser.add_argument('--hid-tr', type=int, default=1280, help='transcription hidden layer dim')
    parser.add_argument('--hid-pr', type=int, default=1024, help='prediction hidden layer dim')
    parser.add_argument('--head-dim', type=int, default=64, help='')
    parser.add_argument('--nhead', type=int, default=8, help='')
    parser.add_argument('--beam-size', type=int, default=16, help='')
    parser.add_argument('--dropout', type=float, default=0.25, help='')
    parser.add_argument('--test-path', type=str, default='', help='')
    parser.add_argument('--ckpt-path', type=str, default='', help='')
    parser.add_argument('--decode-path', type=str, default='', help='')
    parser.add_argument('--enc-type', type=str, default='lstm', help='')
    parser.add_argument('--deep-spec', action='store_true', help='')
    parser.add_argument('--unidirectional', action='store_true', help='')
    parser.add_argument('--dont-fix-path', action='store_true', help='')
    
    args = parser.parse_args()
    args.vocab_size = len(ASR_ID2TOK)

    # Data init
    csv_path = os.path.join(args.test_path, f'{args.rank}.csv')
    data = ASRDataset(args, csv_path,  n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
    collator = CollatorDec(args)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=1, collate_fn=collator)

    # Load model
    print(f'Loading model.')
    model = RNNT(args)
    checkpoint = torch.load(args.ckpt_path, map_location=f'cpu')
    load_dict(model, checkpoint['state_dict'], ddp=False)
    normalizer = checkpoint['normalizer'].to('cpu')
    model.eval()

    # Decode
    print(f'Decoding.')
    if args.rank==0 and not os.path.exists(args.decode_path):
        os.makedirs(args.decode_path)
    if args.rank != 0:
        while not os.path.exists(args.decode_path):
            continue
    write_path = os.path.join(args.decode_path, f'{args.rank}.txt')
    with open(write_path, 'w') as dP:
        for speechB, text, logitLens, lmax in tqdm(loader, file=sys.stdout):
            GT = text[0]
            lens_norm = [1.*(x/lmax) for x in logitLens]
            speechB = normalizer(speechB, torch.tensor(lens_norm).float(), epoch=1000) # mean-var normalize
            speechB = SpecDel(speechB, logitLens, args.fmask, train=False) # del+ddel
            speechB, logitLens = roll_in(speechB, logitLens) # lower sequence length
            hyp, score = model.beam_search(speechB, beam_size=args.beam_size)
            hypText = convert_id2tok(hyp)
            dP.write(f'{GT} ----> {hypText}\n')

if __name__ == '__main__':
    main()
