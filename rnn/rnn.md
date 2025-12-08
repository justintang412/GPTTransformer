
1. Why RNNs add h @ W_hh

Because each hidden state depends on the previous hidden state.
This lets the RNN remember information over time.

⸻

2. Why bias is 1-D (not hidden×hidden)

Bias is added per feature dimension, not transformed by a matrix.
Weights do transformations; bias just shifts.

⸻

3. What is tanh

It is a nonlinear activation mapping values to [-1, 1].
Its derivative shrinks near ±1 → causes vanishing gradients.

⸻

4. Vanishing gradients

During backprop through time:

\frac{\partial L}{\partial h_0} \sim
\prod_{t=1}^{T} \tanh'(z_t)\;W_h

Small derivatives × many steps → gradient goes to 0.

Causes:
	•	tanh/sigmoid saturation
	•	long sequences
	•	weight matrices with norms < 1
	•	repeated multiplications through time

RNNs cannot learn long dependencies because gradients die after 5–20 steps.

⸻

5. Exploding gradients

Opposite of vanishing.
Occurs when weight matrices (or derivatives) are > 1, so repeated multiplications blow gradients up:

1.2^{50} \approx 9000

Symptoms:
	•	huge gradient norms
	•	NaNs
	•	unstable training

Fixed by gradient clipping.

⸻

6. How to detect vanishing/exploding

Check gradient norms:

param.grad.norm()

Tiny → vanishing
Huge → exploding

Norm gives the overall magnitude of a whole gradient tensor.

⸻

7. Saturation

tanh/sigmoid output near ±1 → derivative ≈ 0 → kills gradients.
Saturation is a main cause of vanishing gradients.

⸻

8. RNNs and sequence length

Vanilla RNNs do not work for long sequences because gradients either vanish or explode across many steps.

LSTM/GRU fix this.
Transformers eliminate the problem completely.