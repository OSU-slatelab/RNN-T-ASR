import pdb
import torch
import random
import torch.nn as nn
from conformer import ConformerBlock
from speechbrain.lobes.augment import SpecAugment

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ConformerLayer(nn.Module):
    def __init__(self, hidDim, headDim, nhead, dropout=0.25):
        super(ConformerLayer, self).__init__()
        self.encoder = ConformerBlock(dim = hidDim, dim_head=headDim, heads=nhead, ff_mult = 4, conv_expansion_factor = 2, conv_kernel_size = 32, attn_dropout = dropout, ff_dropout = dropout, conv_dropout = dropout)
        
    def forward(self, x):
        return self.encoder(x)

class LstmLayer(nn.Module):
    def __init__(self, inDim, hidDim, dropout=0.25, bidirectional=True):
        super(LstmLayer, self).__init__()
        self.encoder = nn.LSTM(inDim, hidDim, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(x)
        return self.encoder(x)

    def step(self, x, hidden):
        return self.encoder(x, hidden)
 
class ConformerEncoder(nn.Module):
    def __init__(self, nLayer, hidDim, headDim, nhead, dropout=0.25, spec=False):
        super(ConformerEncoder, self).__init__()
        self.nLayer = nLayer
        for i in range(self.nLayer):
            setattr(self, 'Conf'+str(i), ConformerLayer(hidDim, headDim, nhead, dropout=dropout))
        self.spec = spec
        self.aug = SpecAugment(time_warp=False, freq_mask_width=(0, 100), time_mask_width=(0, 30))

    def augment(self, x):
        if not self.training:
            return x
        return self.aug(x)

    def forward(self, x):
        full = []
        regLyr = random.sample(list(range(self.nLayer)), 1)
        for i in range(self.nLayer):
            x = getattr(self, 'Conf'+str(i))(x)
            if self.spec and i in regLyr:
                x = self.augment(x)
            full.append(x)
        return x, full

class LstmEncoder(nn.Module):
    def __init__(self, nLayer, inDim, hidDim, dropout0=0.25, dropout=0.25, spec=False, bidirectional=True):
        super(LstmEncoder, self).__init__()
        self.nLayer = nLayer
        self.Lstm0 = LstmLayer(inDim, hidDim, dropout=dropout0, bidirectional=bidirectional)
        for i in range(1, self.nLayer):
            setattr(self, 'Lstm'+str(i), LstmLayer(2*hidDim, hidDim, dropout=dropout, bidirectional=bidirectional))
        self.spec = spec
        self.aug = SpecAugment(time_warp=False, freq_mask_width=(0, 100), time_mask_width=(0, 20))

    def augment(self, x):
        if not self.training:
            return x
        return self.aug(x)

    def forward(self, x):
        full = []
        regLyr = random.sample(list(range(self.nLayer)), 1)
        for i in range(self.nLayer):
            x, _ = getattr(self, 'Lstm'+str(i))(x)
            if self.spec and i in regLyr:
                x = self.augment(x)
            full.append(x)
        return x, full

    def step(self, x, hidden):
        return self.Lstm0.step(x, hidden)
        #hiddenO = []
        #for i in range(self.nLayer):
        #    x, hidden = getattr(self, 'Lstm'+str(i)).step(x, hiddenI[i])
        #    hiddenO.append(hidden)
        #return x, hiddenO

