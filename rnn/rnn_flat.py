import torch
import torch.nn as nn
import torch.optim as optim



class RNN_Flat(nn.Module):
    def __init__(self):
        self.input_feature_size = 128
        self.hidden_feature_size = 512
        self.weight_input_hidden = nn.Parameter(
            torch.Tensor(self.hidden_feature_size, self.input_feature_size)
        )
        self.weight_input_bias = nn.Parameter(
            torch.Tensor(self.hidden_feature_size)
        )
        self.weight_hidden_hidden = nn.Parameter(
            torch.Tensor(self.hidden_feature_size, self.hidden_feature_size)
        )
        self.weight_hidden_bias = nn.Parameter(
            torch.Tensor(self.hidden_feature_size)
        )
        std = 1.0 / (self.hidden_feature_size) ** 0.5
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
        

        cell = 