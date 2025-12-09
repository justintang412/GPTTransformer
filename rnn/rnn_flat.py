import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import numpy as np


class RNN_Flat(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 128
        self.hidden_size = 512
        self.batch_size = 10
        self.weight_ih = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.bias_ih = nn.Parameter(torch.Tensor(self.hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias_bh = nn.Parameter(torch.Tensor(self.hidden_size))
        self.output_layer = nn.Linear(self.hidden_size, 1)

        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

        # x:(batch_size, input_size), h:(batch_size, hidden_size)
        # For the first layer, input_size=128, hidden_size=512s
        def cell_ih(x, h):
            return torch.tanh(
                torch.mm(x, self.weight_ih.t())
                + self.bias_ih
                + torch.mm(h, self.weight_hh.t())
                + self.bias_bh
            )

        self.cell_ih = cell_ih

        # xt:(batch_size, hidden_size), h:(batch_size, hidden_size)
        # From the second layer
        def cell_hh(xt, h):
            return torch.tanh(
                torch.mm(xt, self.weight_hh.t())
                + self.bias_ih
                + torch.mm(h, self.weight_hh.t())
                + self.bias_bh
            )

        self.cell_hh = cell_hh

    # output: (batch_size, sequence_size, hidden_size)
    # h: (num_layers, batch_size, hidden_size)
    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, sequnce_size, input_size)
        sequence_output = []
        if h is None:
            h = torch.zeros(6, self.batch_size, self.hidden_size, device=x.device)
        for t in range(x.shape[1]):
            x_t = x[:, t, :]  # time t step: batch_size, input_size
            # 6 layers
            for layer_id in range(6):
                if layer_id == 0:
                    h[0] = self.cell_ih(x_t, h=h[0])
                else:
                    h[layer_id] = self.cell_hh(h[layer_id - 1], h[layer_id])
            sequence_output.append(h[-1].unsqueeze(1))
        
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
            sequence = np.sin(self.frequence * x)
            target = np.sin(self.frequence * (start + self.sequence_length))
            self.data.append(
                sequence.astype(np.float32).reshape(-1, 1)
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
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    model = RNN_Flat().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_losses = []
    evaluation_losses = []

    for epoch in range(1): #10 epochs
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
