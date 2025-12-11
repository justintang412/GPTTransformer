import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


class RNN_Flat(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        """
        Args:
            input_size: Feature dimension per time step.
            hidden_size: Feature dimension of the hidden state.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(
            torch.Tensor(self.hidden_size, self.input_size))
        self.bias_ih = nn.Parameter(torch.Tensor(self.hidden_size))
        self.weight_hh = nn.Parameter(
            torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias_bh = nn.Parameter(torch.Tensor(self.hidden_size))
        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

        self.tanh = lambda x, h: torch.tanh(
            torch.mm(x, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(h, self.weight_hh.t())
            + self.bias_bh
        )

    def forward(
        self, x: torch.Tensor, h: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x batch size, input size
            h batch size, hidden size
        Returns:

        """
        # x shape: (batch_size, sequnce_size, input_size)
        sequence_output = []
        for t in range(x.shape[1]):
            x_t = x[:, t, :]  # time t step: batch_size, input_size
            # 6 layers
            for layer_id in range(6):
                if h is None:
                    h = torch.zeros(
                        6, self.batch_size, self.hidden_size, device=x.device
                    )
                    h[0] = self.tanh(x_t, h=h[0])
                else:
                    h[layer_id] = self.cell(h[layer_id - 1], h[layer_id])
            sequence_output.append(h[-1].unsqueeze(1))
        return torch.cat(sequence_output, dim=1), h
