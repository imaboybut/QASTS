import torch
import torch.nn as nn

class LearnableBias(nn.Module):
    def __init__(self, out_chn, device=None):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn, device=device), requires_grad=True)

    def forward(self, x):

        out = x + self.bias.expand_as(x)
        return out

class LearnableBias4img(nn.Module):
    def __init__(self, out_chn, device=None):
        super(LearnableBias4img, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn, device=device), requires_grad=True)

    def forward(self, x):
        bias_reshaped = self.bias.reshape(1, 1, x.shape[-2], x.shape[-1])
        out = x + bias_reshaped
        return out