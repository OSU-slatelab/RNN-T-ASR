from models import *
from util import *
from data import *
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import f1_score, accuracy_score
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.augment import SpecAugment
from torchaudio.functional import rnnt_loss
import torch.distributed as dist
import numpy as np
import copy
import pdb
import random
import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F

def load2gpu(x, device):
    if x is None:
        return x
    if isinstance(x, dict):
        t2 = {}
        for key, val in x.items():
            t2[key] = val.to(device)
        return t2
    if isinstance(x, list):
        y = []
        for v in x:
            y.append(v.to(device))
        return y
    return x.to(device)

class Trainer(object):
    def __init__(self, args, data, device, optimizer, normalizer, sampler=None, rank=0):
        if sampler is not None:
            shuffle = False
        else:
            shuffle = True
        self.args = args
        self.sampler = sampler
        self.rank = rank
        self.data = data
        self.device = device
        self.aug = SpecAugment(time_warp=False, freq_mask_width=(0, self.args.fmask), time_mask_width=(0, self.args.tmask))
        self.normalizer = normalizer
        self.optimizer = optimizer

        eff_bsz = args.batch_size / args.world_size
        self.update_after = math.ceil(eff_bsz / args.bsz_small)
        collator = CollatorASR(args)
        self.loader = torch.utils.data.DataLoader(data, batch_size=args.bsz_small, shuffle=shuffle, num_workers=4, collate_fn=collator, pin_memory=True, sampler=sampler)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.nepochs, steps_per_epoch=math.ceil(1. * len(self.loader) / self.update_after), anneal_strategy='linear', pct_start=0.3)
        if args.checkpoint is not None:
            self.scheduler.load_state_dict(args.checkpoint['scheduler'])

    def asr(self, model, logger):
        loss_list = []
        for epoch in range(self.args.epochs_done+1, self.args.nepochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            print(f'Running epoch {epoch}.')
            step = 0
            self.optimizer.zero_grad()
            for speechB, textB, targetB, logitLens, lmax, targetLens in tqdm(self.loader):
                model.train()
                step += 1
                lens_norm = [1.*(x/lmax) for x in logitLens]
                speechB = self.normalizer(speechB, torch.tensor(lens_norm).float(), epoch=epoch-1) # mean-var normalize
                speechB = inject_seqn(speechB) # sequence noise injection
                speechB = self.aug(speechB) # specaugment
                speechB, logitLens = roll_in(speechB, logitLens) # lower sequence length
                speechB, textB, targetB, logitLens, targetLens = load2gpu(speechB, self.device), load2gpu(textB, self.device), load2gpu(targetB, self.device), load2gpu(torch.tensor(logitLens), self.device), load2gpu(torch.tensor(targetLens),self.device)

                pred = model(speechB, textB)
                loss = rnnt_loss(pred, targetB.int(), logitLens.int(), targetLens.int(), 0)
                loss.backward()

                if step % self.update_after == 0 or step == len(self.loader):
                    nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                    self.optimizer.step() 
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                loss_rec = loss.detach()
                if self.args.ddp:
                    dist.all_reduce(loss_rec)
                    loss_list.append(loss_rec.item() / dist.get_world_size())
                else:
                    loss_list.append(loss_rec.item())

            if self.rank==0:
                log = f'| epoch = {epoch} | loss_asr = {np.mean(loss_list)} | lr = {self.scheduler.get_last_lr()} |'
                logger.info(log)
            loss_list = []
            if epoch % self.args.checkpoint_after == 0 and self.rank==0:
                checkpoint = {'state_dict':model.state_dict(), 'normalizer':self.normalizer, 'optimizer':self.optimizer.state_dict(), 'scheduler':self.scheduler.state_dict, 'epochs_done':epoch}
                save_checkpoint(checkpoint, f'{self.args.save_path}')
