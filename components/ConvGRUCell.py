from torch import nn


class ConvGRUCell(nn.Module):
    def __int__(self, input_dim, hidden_dim, kernel_size, n_layer=1):