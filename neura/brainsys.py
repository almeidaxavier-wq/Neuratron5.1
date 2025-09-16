import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

import os
import random
import math
import typing

from neuratron import Neuratron

class BrainSys(nn.Module):
    def __init__(self, neura:Neuratron, learning_rate:float, size:typing.Tuple, n_layers:int):
        self.neura = neura
        self.lr = learning_rate
        self.G = nx.DiGraph()
        self.history = []

        for i in range(n_layers):
            self.neura.allocate("Neura-"+i, size[0], size[1])

        for i, pos in self.neura.allocations:
            for j, pos_ in self.neura.allocations:
                if i != j and pos[1][1] - pos[1][0] == pos_[0][1] - pos_[0][0]:
                    self.G.add_edge(i, j, prob=1)

    def forward(self, X:torch.Tensor, initial_alloc:str, depth:int):
        idx = self.neura.allocations[initial_alloc]

        w = self.neura.total_weight[idx[0][0]:idx[0][1], idx[1][0]:idx[1][1]]
        b = self.neura.total_bias[idx[0][0]:idx[0][1]]

        res = F.linear(X, w, b)
        next_alloc = initial_alloc

        for i in range(depth):
            weights = [node['prob'] for node in self.G.neighbors(next_alloc)]
            chosed = random.choices([node for node in self.G.neighbors(next_alloc)], weights=weights, k=1)

            idx = self.neura.allocations[chosed]
            w = self.neura.total_weight[idx[0][0]:idx[0][1], idx[1][0]:idx[1][1]]
            b = self.neura.total_bias[idx[0][0]:idx[0][1]]

            res = F.linear(res, w, b)

        return res

