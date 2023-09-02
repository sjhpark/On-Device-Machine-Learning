import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Feed Forward Neural Network
class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_hidden):
        super(FFNN, self).__init__()
        input_dim_list = np.linspace(input_dim, hidden_dim, num_hidden, dtype=int)[:-1]
        hidden_dim_list = np.linspace(input_dim, hidden_dim, num_hidden, dtype=int)[1:]

        # blocks of linear layers
        ll_blocks = [self.ll_block(input_dim, hidden_dim) 
                    for input_dim, hidden_dim in zip(input_dim_list, hidden_dim_list)]

        self.model = nn.Sequential(*ll_blocks,
                                    nn.Linear(hidden_dim_list[-1], out_dim))

    def ll_block(self, input_dim, hidden_dim):
        # linear layer block
        return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
                )
    
    def forward(self, x):
        return self.model(x)