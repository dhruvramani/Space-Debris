import os
import gc
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import *
from dataset import DebrisDataset
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch Space Orbital Path Prediction')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate') # NOTE change for diff models
parser.add_argument('--batch_size', default=30, type=int)
parser.add_argument('--resume', '-r', type=int, default=1, help='resume from checkpoint')
parser.add_argument('--epochs', '-e', type=int, default=4, help='Number of epochs to train.')
parser.add_argument('--momentum', '-lm', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-ld', type=float, default=1e-5, help='Weight decay (L2 penalty).')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, epoch, step = 0, 0, 0
loss_fn = torch.nn.MSELoss()
print('==> Preparing data..')

# To get logs of current run only
with open("../save/logs/train_loss.log", "w+") as f:
    pass 

print('==> Creating network..')
net = SpaceLSTM()
net = net.to(device)

if(args.resume):
    if(os.path.isfile('../save/network.ckpt')):
        net.load_state_dict(torch.load('../save/network.ckpt'))
        print('==> SpaceNet : loaded')

    if(os.path.isfile("../save/info.txt")):
        with open("../save/info.txt", "r") as f:
            epoch, step = (int(i) for i in str(f.read()).split(" "))
        print("=> SpaceNet : prev epoch found")

def train(epoch):
    global step
    print('==> Preparing data..')
    dataset = DebrisDataset(n_rows=30, steps= 10)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)

    print('\n=> Loss Epoch: {}'.format(epoch))
    train_loss, total = 0, 0
    params = list(net.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.decay)
    
    for i in range(step, len(dataloader)):
        sequences, predictions = next(dataloader)
        sequences, predictions = sequences.permute([1, 0, 2]), predictions.permute([1, 0, 2])
        sequences, predictions = sequences.to(device), predictions.to(device)
        output = net(sequences)
        optimizer.zero_grad()
        loss = loss_fn(output[-1], predictions[0]) # Last LSTM output, Prediction
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss += loss.item()

        with open("../save/logs/train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / (i - step +1)))

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(net.state_dict(), '../save/network.ckpt')

        with open("../save/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, i))

        progress_bar(i, len(dataloader), 'Loss: %.3f' % (train_loss / (i - step + 1)))

    step = 0
    del dataloader
    del dataset
    print('=> Loss Network : Epoch [{}/{}], Loss:{:.4f}'.format(epoch + 1, 5, train_loss / len(dataloader)))


for epoch in range(epoch, epoch + args.epochs):
    train(epoch)
