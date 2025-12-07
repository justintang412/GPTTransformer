import torch
import torch.nn as nn

torch.manual_seed(0)

# small RNN that will definitely vanish gradients
input_size = 10
hidden_size = 20
seq_len = 200

rnn = nn.RNN(input_size, hidden_size, nonlinearity="tanh", batch_first=True)

# random input
x = torch.randn(1, seq_len, input_size)

# initial hidden
h0 = torch.zeros(1, 1, hidden_size)

# forward pass
out, hn = rnn(x, h0)

# loss depends only on the final output (time step 199)
loss = out[:, -1, :].sum()
loss.backward()

# inspect gradient magnitude for earlier timesteps inside W_hh
print("Gradient L2-norm of recurrent weight:")
print(torch.norm(rnn.weight_hh_l0.grad))

# Inspect how small gradients get at the input of an early step.
# (Vanishing gradient hits the gradient flowing into earlier x's)
print("Grad norm at x[:, 0, :]:", torch.norm(x.grad[:, 0, :]) if x.grad is not None else "None")
