import torch
from torch import nn

class LSTMScratch(nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.num_hiddens = num_hiddens

        def init_weights_and_bias():
            W_x = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
            W_h = nn.Parameter(torch.randn(num_hiddens, num_hiddens) * sigma)
            b = nn.Parameter(torch.zeros(num_hiddens))
            return W_x, W_h, b

        self.W_xi, self.W_hi, self.b_i = init_weights_and_bias()
        self.W_xf, self.W_hf, self.b_f = init_weights_and_bias()
        self.W_xo, self.W_ho, self.b_o = init_weights_and_bias()
        self.W_xc, self.W_hc, self.b_c = init_weights_and_bias()

    def forward(self, inputs, H_C=None):
        if H_C is None:
            B = inputs.shape[1]
            device = inputs.device
            H = torch.zeros(B, self.num_hiddens, device=device)
            C = torch.zeros(B, self.num_hiddens, device=device)
        else:
            H, C = H_C

        outputs = []
        for X in inputs:
            I = torch.sigmoid(X @ self.W_xi + H @ self.W_hi + self.b_i)
            F = torch.sigmoid(X @ self.W_xf + H @ self.W_hf + self.b_f)
            O = torch.sigmoid(X @ self.W_xo + H @ self.W_ho + self.b_o)
            C_tilde = torch.tanh(X @ self.W_xc + H @ self.W_hc + self.b_c)

            C = F * C + I * C_tilde
            H = O * torch.tanh(C)
            outputs.append(H)

        return outputs, (H, C)
