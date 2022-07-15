import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   remove_self_loops)

from data import load_data
from encoder import FNN, GINEncoder, VGINEncoder
from utils import evaluate_embedding, logger, view_generator  # attack_edge


class CLUB(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x, edge_index, batch, h):
        mu, logvar = self.encoder(x, edge_index, batch)

        positive = -(mu - h)**2 / 2. / (logvar.exp() + 1e-7)

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        h_samples_1 = h.unsqueeze(0)  # shape [1,nsample,dim]

        negative = -((h_samples_1 - prediction_1)**2).mean(dim=1) / 2. / (
            logvar.exp() + 1e-7)

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x, edge_index, batch, h):
        mu, logvar = self.encoder(x, edge_index, batch)
        return (-(mu - h)**2 / logvar.exp() - logvar +
                1e-7).sum(dim=1).mean(dim=0)

    def learning_loss(self, x, edge_index, batch, h):
        return -self.loglikeli(x, edge_index, batch, h)


class MVGIB(torch.nn.Module):
    def __init__(self, encoder_c1, encoder_h1, encoder_c2, encoder_h2,
                 encoder_f, club1, club2):
        super().__init__()
        self.encoder_c1 = encoder_c1  # v1 view-common
        self.encoder_h1 = encoder_h1  # v1 view-specific
        self.encoder_c2 = encoder_c2  # v2 view-common
        self.encoder_h2 = encoder_h2  # v2 view-specific

        self.encoder_f = encoder_f  # feature
        self.club1 = club1
        self.club2 = club2

    def decoder(self, z, edge_index):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value)

    def forward(self, data):
        x, batch = data.x, data.batch
        edge_index1, edge_index2 = data.edge_index1, data.edge_index2

        c1 = self.encoder_c1(x, edge_index1, batch)
        c2 = self.encoder_c2(x, edge_index2, batch)

        h1 = self.encoder_h1(x, edge_index1, batch)
        h2 = self.encoder_h2(x, edge_index2, batch)

        if self.training:
            return c1, c2, h1, h2
        else:
            return torch.cat([c1, c2, h1, h2], dim=-1)

    def recon_f_loss(self, x, c1, c2, h1, h2, batch):
        z = torch.cat([c1, c2, h1, h2], dim=-1)
        z = z[batch]
        re_f = self.encoder_f(z)
        f_loss = F.mse_loss(re_f, x)
        return f_loss

    def recon_loss(self, z, edge_index, batch):
        z = z[batch]
        pos_loss = -torch.log(self.decoder(z, edge_index) + 1e-7).mean()

        pos_edge_index, _ = remove_self_loops(edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        neg_edge_index = negative_sampling(pos_edge_index)

        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index) +
                              1e-7).mean()
        return pos_loss + neg_loss

    def common_loss(self, c1, c2, edge_index1, edge_index2, batch, beta=1):
        loss_t1 = self.recon_loss(c1, edge_index1, batch) + self.recon_loss(
            c2, edge_index2, batch)
        loss_t2 = self.recon_loss(c1, edge_index2, batch) + self.recon_loss(
            c2, edge_index1, batch)
        return loss_t1 + beta * loss_t2

    def specific_loss(self,
                      x,
                      h1,
                      h2,
                      edge_index1,
                      edge_index2,
                      batch,
                      gamma=1):
        loss_t1 = self.recon_loss(h1, edge_index1, batch) + self.recon_loss(
            h2, edge_index2, batch)

        loss_t2_p1 = self.club1(x, edge_index1, batch,
                                h2) + self.club1.learning_loss(
                                    x, edge_index1, batch, h2)
        loss_t2_p2 = self.club2(x, edge_index2, batch,
                                h1) + self.club2.learning_loss(
                                    x, edge_index2, batch, h1)

        return loss_t1 + gamma * (loss_t2_p1 + loss_t2_p2)

    def loss(self,
             x,
             c1,
             h1,
             c2,
             h2,
             edge_index1,
             edge_index2,
             batch,
             beta=1,
             gamma=1):
        c_loss = self.common_loss(c1, c2, edge_index1, edge_index2, batch,
                                  beta)
        v_loss = self.specific_loss(x, h1, h2, edge_index1, edge_index2, batch,
                                    gamma)
        f_loss = self.recon_f_loss(x, c1, c2, h1, h2, batch)
        return c_loss + v_loss + f_loss


def train(model, optimizer, loader, device, beta, gamma):
    model.train()

    total_loss = []
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        c1, c2, h1, h2 = out[0], out[1], out[2], out[3]
        loss = model.loss(x=data.x,
                          c1=c1,
                          h1=h1,
                          c2=c2,
                          h2=h2,
                          edge_index1=data.edge_index1,
                          edge_index2=data.edge_index2,
                          batch=data.batch,
                          beta=beta,
                          gamma=gamma)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        total_loss.append(loss.item())
        optimizer.step()
    train_loss = np.mean(total_loss)
    return train_loss


def test(model, loader, device):
    model.eval()
    z, y = [], []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        z.append(out.cpu().numpy())
        y.append(data.y.cpu().numpy())
    z = np.concatenate(z, 0)
    y = np.concatenate(y, 0)
    test_acc, test_std = evaluate_embedding(z, y)
    return test_acc, test_std


def cross_validation(dataset, model, epochs, batch_size, lr, lr_decay_factor,
                     lr_decay_step_size, weight_decay, device, beta, gamma):
    model.to(device)
    loader = DataLoader(dataset, batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc, best_std = 0, 0
    for epoch in range(1, epochs + 1):
        train_loss = train(model=model,
                           optimizer=optimizer,
                           loader=loader,
                           beta=beta,
                           gamma=gamma,
                           device=device)

        test_acc, test_std = test(model=model, loader=loader, device=device)

        if test_acc >= best_acc:
            best_acc = test_acc
            best_std = test_std

        eval_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_acc': best_acc,
            'test_std': best_std
        }

        logger(eval_info)

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']
    # logger(eval_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='MUTAG')
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=50)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--view1', type=str, default='adj')
    parser.add_argument('--view2', type=str, default='KNN')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}')
    seed_everything(42)

    data = load_data(args.dataset, cleaned=False)
    # data_attack = attack_edge(data, edge_fraction=0.5, mode='remove')
    dataset = view_generator(data, view1=args.view1, view2=args.view2)

    model = MVGIB(
        encoder_c1=GINEncoder(in_dim=data.num_features,
                              hidden_dim=args.hidden,
                              num_layers=args.num_layers),
        encoder_h1=GINEncoder(in_dim=data.num_features,
                              hidden_dim=args.hidden,
                              num_layers=args.num_layers),
        encoder_c2=GINEncoder(in_dim=data.num_features,
                              hidden_dim=args.hidden,
                              num_layers=args.num_layers),
        encoder_h2=GINEncoder(in_dim=data.num_features,
                              hidden_dim=args.hidden,
                              num_layers=args.num_layers),
        encoder_f=FNN(in_dim=4 * args.hidden,
                      hidden_dim=args.hidden,
                      out_dim=data.num_features,
                      num_layers=4),
        club1=CLUB(encoder=VGINEncoder(
            in_dim=data.num_features, hidden_dim=args.hidden, num_layers=4)),
        club2=CLUB(encoder=VGINEncoder(in_dim=data.num_features,
                                       hidden_dim=args.hidden,
                                       num_layers=4))).to(device)

    cross_validation(dataset=dataset,
                     model=model,
                     epochs=args.epochs,
                     batch_size=args.batch_size,
                     lr=args.lr,
                     lr_decay_factor=args.lr_decay_factor,
                     lr_decay_step_size=args.lr_decay_step_size,
                     weight_decay=0,
                     device=device,
                     beta=args.beta,
                     gamma=args.gamma)
