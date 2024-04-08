import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class L2O_g(nn.Module):
    def __init__(self, metric_fn, preproc=True, hidden_sz=30,
                 preproc_factor=10.0, lamb_a=0.01, lamb_b=0.01,
                 device='cpu', abl_g=False):
        super().__init__()
        self.hidden_sz = hidden_sz
        if preproc:
            self.recurs1 = nn.LSTMCell(2, hidden_sz)
        else:
            self.recurs1 = nn.LSTMCell(1, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 2)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)
        self.metric_fn = metric_fn

        self.switch = nn.Linear(hidden_sz, 1)
        self.lamb_a = lamb_a
        self.lamb_b = lamb_b
        # self.reg = 0.001
        self.reg = 0.0
        self.device = device
        self.abl_g = abl_g

    def forward(self, p, inp, hidden, cell):
        if self.preproc:
            inp = inp.data
            inp2 = torch.zeros(inp.size()[0], 2).to(self.device)
            keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0][keep_grads] = (torch.log(torch.abs(inp[keep_grads]) + torch.tensor(1e-8).to(self.device)) / torch.tensor(self.preproc_factor).to(self.device)).squeeze()
            inp2[:, 1][keep_grads] = torch.sign(inp[keep_grads]).squeeze()

            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = (float(torch.exp(torch.tensor(self.preproc_factor).to(self.device))) * inp[~keep_grads]).squeeze()
            inp = Variable(inp2)
        hidden0, cell0 = self.recurs1(inp, (hidden[0].to(self.device), cell[0].to(self.device)))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1].to(self.device), cell[1].to(self.device)))

        ak, dk = np.split(self.output(hidden1), 2, axis=1)
        sk = self.lamb_a*torch.exp(self.lamb_b*ak.view(*p.size()))*dk.view(*p.size())
        gamma = torch.relu(self.switch(hidden1)).squeeze()

        if self.abl_g == False:
            if torch.sum(gamma) != 0:
                Bk = torch.linalg.pinv(self.metric_fn(p.cpu()).float().to(self.device) + torch.eye(*p.size()).to(self.device) * self.reg).diag()
                out = (1.0 - gamma) * sk + gamma * torch.exp(self.lamb_b*ak.view(*p.size())) * Bk.detach()
            else: out = sk
        else: out = sk

        return out, (hidden0, hidden1), (cell0, cell1)
