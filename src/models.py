import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(1)

class SpaceLSTM(torch.nn.Module):
    def __init__(self, inp_dim=6, hidden_dim=4, op_dim=6, batch_size=128):
        super(SpaceLSTM, self).__init__()
        # Hidden I/P : 2 
        self.batch_size, self.hidden_dim = batch_size, hidden_dim
        self.inp_dim, self.op_dim = inp_dim, op_dim
        self.hidden = (torch.randn(1, self.batch_size, self.hidden_dim), torch.randn(1, self.batch_size, self.op_dim)) 
        self.lstm = nn.LSTM(self.inp_dim, self.op_dim)

    def forward(self, inp_seq):
        out, self.hidden = self.lstm(inp_seq, self.hidden)
        return out
