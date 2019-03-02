import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(1)

class SpaceNet(torch.nn.Module):
    def __init__(self, inp_dim=7, op_dim=7, ):
        super(SpaceNet, self).__init__()
        # Hidden I/P : 2 
        self.inp_dim, self.op_dim = inp_dim, op_dim
        self.hidden = (torch.randn(1, 1, self.inp_dim), torch.randn(1, 1, self.op_dim)) 
        self.lstm = nn.LSTM(self.inp_dim, self.op_dim)

    def forward(self, inp_seq):
        out, self.hidden = self.lstm(inp_seq, self.hidden)
        return out