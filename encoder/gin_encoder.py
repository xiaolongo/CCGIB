import torch
from torch.nn import ELU, BatchNorm1d, Linear, Sequential, Tanh
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.nn.conv import MessagePassing


class EGIN(MessagePassing):
    def __init__(self, nn, eps=0., train_eps=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_weight, size=None):
        out = self.propagate(edge_index,
                             edge_weight=edge_weight,
                             x=x,
                             size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j):
        return x_j


class GINEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=5):
        super().__init__()

        self.conv1 = GINConv(Sequential(Linear(in_dim, hidden_dim), ELU(),
                                        Linear(hidden_dim, hidden_dim), ELU(),
                                        BatchNorm1d(hidden_dim)),
                             train_eps=True)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(
                GINConv(Sequential(Linear(hidden_dim, hidden_dim), ELU(),
                                   Linear(hidden_dim, hidden_dim), ELU(),
                                   BatchNorm1d(hidden_dim)),
                        train_eps=True))
        self.convs.append(
            GINConv(Sequential(Linear(hidden_dim, hidden_dim), ELU(),
                               Linear(hidden_dim, hidden_dim), ELU(),
                               BatchNorm1d(hidden_dim)),
                    train_eps=True))

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)

        for conv in self.convs:
            h = conv(h, edge_index)

        h_g = global_mean_pool(h, batch)
        return h_g


class VGINEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=5):
        super().__init__()

        self.conv1 = GINConv(Sequential(Linear(in_dim, hidden_dim), ELU(),
                                        Linear(hidden_dim, hidden_dim), ELU(),
                                        BatchNorm1d(hidden_dim)),
                             train_eps=True)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(
                GINConv(Sequential(Linear(hidden_dim, hidden_dim), ELU(),
                                   Linear(hidden_dim, hidden_dim), ELU(),
                                   BatchNorm1d(hidden_dim)),
                        train_eps=True))

        self.conv_mu = GINConv(Sequential(Linear(hidden_dim,
                                                 hidden_dim), ELU(),
                                          Linear(hidden_dim, hidden_dim)),
                               train_eps=False)
        self.conv_logvar = GINConv(Sequential(Linear(hidden_dim, hidden_dim),
                                              ELU(),
                                              Linear(hidden_dim, hidden_dim),
                                              Tanh()),
                                   train_eps=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv_mu.reset_parameters()
        self.conv_logvar.reset_parameters()

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)

        for conv in self.convs:
            h = conv(h, edge_index)

        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)

        h_g_mu = global_mean_pool(mu, batch)
        h_g_logvar = global_mean_pool(logvar, batch)
        return h_g_mu, h_g_logvar
