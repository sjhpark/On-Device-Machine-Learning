import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Feed Forward Neural Network
class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_hidden, bias=True):
        super(FFNN, self).__init__()
        self.bias = bias
        dim_list = [input_dim] + [hidden_dim] * (num_hidden)

        # blocks of linear layers
        ll_blocks = [self.ll_block(input_dim, hidden_dim) for input_dim, hidden_dim in zip(dim_list[:-1], dim_list[1:])]
        self.model = nn.Sequential(*ll_blocks,
                                    nn.Linear(dim_list[-1], out_dim, bias=bias))

    def ll_block(self, input_dim, hidden_dim):
        # linear layer block
        return nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=self.bias),
                nn.ReLU()
                )
    
    def forward(self, x):
        return self.model(x)
