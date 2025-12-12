import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import numpy as np


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
        self.weight_hh = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias_bh = nn.Parameter(torch.Tensor(self.hidden_size))
        self.output_layer = nn.Linear(self.hidden_size, 1)

        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

        self.first_cell = lambda x, h: torch.tanh(
            torch.mm(x, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(h, self.weight_hh.t())
            + self.bias_bh
        )

        self.subsequence_cell = lambda xt, h: torch.tanh(
            torch.mm(xt, self.weight_hh.t())
            + self.bias_bh
            + torch.mm(h, self.weight_hh.t())
            + self.bias_bh
        )

    # output: (batch_size, sequence_size, hidden_size)
    # h: (num_layers, batch_size, hidden_size)
    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x batch size, input size
            h batch size, hidden size
        Returns:

        """
        # x shape: (batch_size, sequence_size, input_size)
        batch_size, seq_len, _ = x.shape
        num_layers = 6
        sequence_output = []

        # initialize hidden if not provided
        if h is None:
            h = torch.zeros(num_layers, batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)

            # compute next hidden states without in-place writes
            h_next = []
            # layer 0 uses input x_t
            h0 = self.first_cell(x_t, h=h[0])
            h_next.append(h0)

            # higher layers use the previous layer's hidden at the same time step
            for layer_id in range(1, num_layers):
                h_next.append(self.subsequence_cell(h_next[layer_id - 1], h[layer_id]))

            # stack to form new hidden state tensor for all layers
            h = torch.stack(h_next, dim=0)

            # collect top/last layer hidden state for this time step
            sequence_output.append(h[-1].unsqueeze(1).contiguous())
        
        output = self.output_layer(torch.cat(sequence_output, dim=1))
        return output, h


class SineWaveDataset(Dataset):
    def __init__(self) -> None:
        self.data = []
        self.targets = []
        self.frequence = 0.1
        self.sequence_length = 20
        for _ in range(1000):
            start = np.random.uniform(0, 10)
            x = np.arange(start, start + self.sequence_length, 1)  # 20 time steps
            sequence = np.sin(self.frequence * x).astype(np.float32)
            target = np.sin(self.frequence * (start + self.sequence_length)).astype(np.float32)
            self.data.append(
                sequence.reshape(-1, 1)
            )  # (sequence_length, 1)
            self.targets.append(target)

    def __len__(self) -> int:
        return 1000

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.data[index], self.targets[index]

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    dataset = SineWaveDataset()
    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    model = RNN_Flat(1, 2048).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    train_losses = []
    evaluation_losses = []

    for epoch in range(100): #10 epochs
        model.train()
        total_loss = 0.0
        for batch_index, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output[:, -1, :], y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_losses.append(loss.item())
            print(f"Epoch {epoch+1}, Batch {batch_index+1}, Loss: {loss.item()}")
        print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_loader)}")


train()
