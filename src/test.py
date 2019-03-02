import os
import gc
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from models import *
from dataset import SpaceDataset
from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Creating network..')
net = SpaceLSTM()
net = net.to(device)

def test():
    global net
    net.load_state_dict(torch.load('../save/network.ckpt'))
    vdataset = SpaceDataset('/home/nevronas/dataset/', download=False)
    dataloader = DataLoader(vdataset, batch_size=1)
    sequences, predictions = next(iter(dataloader))
    out = net(sequences)
    out = out[0].detach().cpu().numpy()
    sequences = sequences[0].cpu().numpy()
    '''
    matplotlib.image.imsave('../save/plots/input/sequences.png', sequences[0])
    matplotlib.image.imsave('../save/plots/output/stylized_sequences.png', out[0])
    aud_res = reconstruction(sequences[0], phase)
    out_res = reconstruction(out[0], phase[:, :-3])
    librosa.output.write_wav("../save/plots/input/raw_sequences.wav", aud_res, fs)
    librosa.output.write_wav("../save/plots/output/raw_output.wav", out_res, fs)
    '''
    print("Testing Finished")