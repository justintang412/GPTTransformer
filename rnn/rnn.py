"""
RNN Implementation from Scratch using PyTorch
This module contains a basic RNN cell and a complete RNN model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class RNNCell(nn.Module):
    """
    Basic RNN Cell - implements one time step of the RNN
    
    Equations:
    h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        """
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
        """
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input to hidden weights
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        # Hidden to hidden weights
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # Biases
        self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through one time step
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            h: Hidden state of shape (batch_size, hidden_size)
        
        Returns:
            h_new: New hidden state of shape (batch_size, hidden_size)
        """
        h_new = torch.tanh(
            torch.mm(x, self.weight_ih.t()) + self.bias_ih +
            torch.mm(h, self.weight_hh.t()) + self.bias_hh
        )
        return h_new


class RNN(nn.Module):
    """
    RNN Model that processes sequences
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        """
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of stacked RNN layers
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create multiple RNN cells for stacking
        self.rnn_cells = nn.ModuleList([
            RNNCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RNN
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            h: Initial hidden state of shape (num_layers, batch_size, hidden_size)
        
        Returns:
            output: Output tensor of shape (batch_size, seq_length, hidden_size)
            h: Final hidden state of shape (num_layers, batch_size, hidden_size)
        """
        batch_size, seq_length, _ = x.size()
        
        # Initialize hidden state if not provided
        if h is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        output = []
        
        # Process each time step
        for t in range(seq_length):
            x_t = x[:, t, :]  # Shape: (batch_size, input_size)
            h_new = []
            # Process through each layer
            for layer in range(self.num_layers):
                h_layer_new = self.rnn_cells[layer](x_t, h[layer])
                x_t = h_layer_new
                h_new.append(h_layer_new)
            h = torch.stack(h_new)
            output.append(h[-1].unsqueeze(1))  # Use output from last layer
        
        output = torch.cat(output, dim=1)  # Shape: (batch_size, seq_length, hidden_size)
        
        return output, h


class SimpleRNNWithOutput(nn.Module):
    """
    RNN with output layer for sequence-to-value prediction
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        """
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            output_size: Size of output
            num_layers: Number of RNN layers
        """
        super(SimpleRNNWithOutput, self).__init__()
        self.rnn = RNN(input_size, hidden_size, num_layers)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
        
        Returns:
            output: Predictions of shape (batch_size, output_size)
        """
        rnn_output, _ = self.rnn(x)
        # Use the last hidden state for prediction
        last_hidden = rnn_output[:, -1, :]
        output = self.output_layer(last_hidden)
        return output


class LSTMCell(nn.Module):
    """Basic LSTM cell implementing one time step.

    Equations (per time step):
        i_t = sigmoid(W_ii x_t + W_hi h_{t-1} + b_i)
        f_t = sigmoid(W_if x_t + W_hf h_{t-1} + b_f)
        g_t = tanh   (W_ig x_t + W_hg h_{t-1} + b_g)
        o_t = sigmoid(W_io x_t + W_ho h_{t-1} + b_o)

        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # We pack all four gates into single matrices for efficiency
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        self._reset_parameters()

    def _reset_parameters(self):
        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One LSTM step.

        Args:
            x: (batch_size, input_size)
            h: previous hidden state (batch_size, hidden_size)
            c: previous cell state (batch_size, hidden_size)

        Returns:
            h_new, c_new: updated hidden and cell states
        """
        gates = (
            torch.mm(x, self.weight_ih.t()) + self.bias_ih +
            torch.mm(h, self.weight_hh.t()) + self.bias_hh
        )

        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)

        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * torch.tanh(c_new)
        return h_new, c_new


class LSTM(nn.Module):
    """Multi-layer LSTM built from LSTMCell, similar to the RNN class above."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_cells = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through time.

        Args:
            x: (batch_size, seq_length, input_size)
            h: (num_layers, batch_size, hidden_size) or None
            c: (num_layers, batch_size, hidden_size) or None

        Returns:
            output: (batch_size, seq_length, hidden_size) from last layer
            (h_n, c_n): final hidden and cell states
        """
        batch_size, seq_length, _ = x.size()

        if h is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        if c is None:
            c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        outputs: List[torch.Tensor] = []

        for t in range(seq_length):
            x_t = x[:, t, :]
            new_h_layers, new_c_layers = [], []
            for layer in range(self.num_layers):
                new_h_layer, new_c_layer = self.lstm_cells[layer](x_t, h[layer], c[layer])
                new_h_layers.append(new_h_layer)
                new_c_layers.append(new_c_layer)
                x_t = new_h_layer
            h = torch.stack(new_h_layers)
            c = torch.stack(new_c_layers)
            outputs.append(h[-1].unsqueeze(1))

        output = torch.cat(outputs, dim=1)
        return output, (h, c)


class SimpleLSTMWithOutput(nn.Module):
    """LSTM-based sequence-to-value model (parallel to SimpleRNNWithOutput)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super(SimpleLSTMWithOutput, self).__init__()
        self.lstm = LSTM(input_size, hidden_size, num_layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_output, _ = self.lstm(x)
        last_hidden = lstm_output[:, -1, :]
        output = self.output_layer(last_hidden)
        return output


class SinewaveDataset(Dataset):
    """
    Dataset that generates sine wave sequences
    Task: Given a sequence of sine values, predict the next value
    """
    
    def __init__(self, num_samples: int = 1000, seq_length: int = 20, freq: float = 0.1):
        """
        Args:
            num_samples: Number of samples to generate
            seq_length: Length of each sequence
            freq: Frequency of sine wave
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.freq = freq
        
        self.data = []
        self.targets = []
        
        # Generate sine wave data
        for i in range(num_samples):
            start = np.random.uniform(0, 10)
            x = np.arange(start, start + seq_length, 1)
            sequence = np.sin(self.freq * x)
            target = np.sin(self.freq * (start + seq_length))
            
            self.data.append(sequence.astype(np.float32).reshape(-1, 1))
            self.targets.append(np.array([target], dtype=np.float32))
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.data[idx], self.targets[idx]


class Trainer:
    """
    Training class for the RNN model
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Args:
            model: The RNN model to train
            device: Device to train on (cpu or cuda)
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate on validation data
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output, y)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        """
        Full training loop
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
        """
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('rnn_losses1.png')
        print("Loss plot saved as 'rnn_losses.png'")
        plt.show()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.autograd.set_detect_anomaly(True)
    # Configuration
    INPUT_SIZE = 1
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 1
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    EPOCHS = 10
    SEQ_LENGTH = 20
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset and dataloaders
    print("\nCreating dataset...")
    dataset = SinewaveDataset(num_samples=1000, seq_length=SEQ_LENGTH)
    
    # Split into train and validation (80-20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    print("Creating RNN model...")
    model = SimpleLSTMWithOutput(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        num_layers=NUM_LAYERS
    )
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train the model
    print("\nStarting training...")
    trainer = Trainer(model, device)
    trainer.train(train_loader, val_loader, EPOCHS)
    
    # Plot results
    print("\nPlotting results...")
    trainer.plot_losses()
    
    # Test on a sample
    print("\nTesting on a sample sequence...")
    model.eval()
    with torch.no_grad():
        sample_x, sample_y = dataset[0]
        sample_x_tensor = torch.FloatTensor(sample_x).unsqueeze(0).to(device)
        prediction = model(sample_x_tensor)
        print(f"Target: {sample_y[0]:.4f}")
        print(f"Prediction: {prediction.item():.4f}")
