# scratch-lstm-language-model

A **from-scratch character-level LSTM language model** implemented in PyTorch, without using `nn.LSTM`.

This repository is designed to expose the **full internal mechanics** of Long Short-Term Memory networks — including gate equations, cell-state dynamics, truncated backpropagation through time (BPTT), gradient clipping, perplexity-based evaluation, and temperature-controlled text generation.

The goal is **mechanistic clarity** rather than abstraction, making the code suitable for:
- deep learning education,
- research prototyping,
- and extension to stacked / deep recurrent models.

---

## Key Features

- Explicit implementation of **LSTM gates** (input, forget, output, candidate cell)
- Manual **hidden state and cell state propagation**
- **Truncated BPTT** with hidden-state detachment
- Stable optimization via **gradient clipping**
- Correct **token-normalized perplexity** computation
- **Train / validation split** with generalization monitoring
- **Temperature-based probabilistic text generation**
- Real-time batch-level training diagnostics

---

## Model Overview

### LSTM Cell Equations

At each time step \( t \), the LSTM computes:

$$
\begin{aligned}
I_t &= \sigma(X_t W_{xi} + H_{t-1} W_{hi} + b_i) \\
F_t &= \sigma(X_t W_{xf} + H_{t-1} W_{hf} + b_f) \\
O_t &= \sigma(X_t W_{xo} + H_{t-1} W_{ho} + b_o) \\
\tilde{C}_t &= \tanh(X_t W_{xc} + H_{t-1} W_{hc} + b_c) \\
C_t &= F_t \odot C_{t-1} + I_t \odot \tilde{C}_t \\
H_t &= O_t \odot \tanh(C_t)
\end{aligned}
$$

All parameters are learned explicitly, without relying on PyTorch’s high-level recurrent modules.

---

## Repository Structure

```text
scratch-lstm-language-model/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── pride_and_prejudice.txt
│
├── src/
│   ├── model.py        # LSTMScratch implementation
│   ├── train.py        # training & validation loops
│   ├── generate.py     # text generation
│   └── utils.py        # vocabulary + helpers
│
├── notebooks/
│   └── exploration.ipynb
│
└── scripts/
    └── run_train.py    # training entry point
