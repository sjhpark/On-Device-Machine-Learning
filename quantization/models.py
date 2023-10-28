import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
torch.backends.quantized.engine = 'fbgemm'

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

# Quantized Linear Layer
class QuantizedLinear(nn.Module):
  """ Adapted from: https://github.com/pytorch/pytorch/blob/main/torch/ao/nn/quantized/modules/linear.py"""
  def __init__(self, in_features, out_features, scale=1.0, zero_point=0, bias=True, dtype=torch.qint8):
    super().__init__()

    self.in_features = in_features
    self.out_features = out_features
    bias = None

    self.scale = 1.0
    self.zero_point = zero_point

    bias = torch.zeros(out_features, dtype=torch.float)
    weight_q8 = torch._empty_affine_quantized(
        [out_features, in_features], scale=1, zero_point=0, dtype=torch.qint8)

    self._packed_params = torch.ops.quantized.linear_prepack(weight_q8, bias)

  def set_weight_bias(self, weight_q8, bias):
    self._packed_params = torch.ops.quantized.linear_prepack(weight_q8, bias)

  def forward(self, x_quint8):
    return torch.ops.quantized.linear(
        x_quint8, self._packed_params, self.scale, self.zero_point)

  ### Nothing to See Here
  def _weight_bias(self):
      return self._packed_params._weight_bias()

  def weight(self):
      return self._weight_bias()[0]

  def bias(self):
      return self._weight_bias()[1]

  def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
      self._packed_params.set_weight_bias(w, b)

  def _save_to_state_dict(self, destination, prefix, keep_vars):
      super()._save_to_state_dict(destination, prefix, keep_vars)
      destination[prefix + 'scale'] = torch.tensor(self.scale)
      destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)

  def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):
      self.scale = float(state_dict[prefix + 'scale'])
      state_dict.pop(prefix + 'scale')

      self.zero_point = int(state_dict[prefix + 'zero_point'])
      state_dict.pop(prefix + 'zero_point')

      super()._load_from_state_dict(
          state_dict, prefix, local_metadata, False,
          missing_keys, unexpected_keys, error_msgs)

# Feed Forward Neural Network with Quantized (qint8) Linear Layers
class FFNN_qint8(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_hidden, bias=True):
        super(FFNN_qint8, self).__init__()
        self.bias = bias
        dim_list = [input_dim] + [hidden_dim] * (num_hidden)

        # blocks of linear layers
        ll_blocks = [self.ll_block(input_dim, hidden_dim) for input_dim, hidden_dim in zip(dim_list[:-1], dim_list[1:])]
        self.model = nn.Sequential(*ll_blocks,
                                    QuantizedLinear(dim_list[-1], out_dim, bias=bias))

    def ll_block(self, input_dim, hidden_dim):
        # linear layer block
        return nn.Sequential(
                QuantizedLinear(input_dim, hidden_dim, bias=self.bias),
                nn.ReLU()
                )
    
    def forward(self, x):
        return self.model(x)
