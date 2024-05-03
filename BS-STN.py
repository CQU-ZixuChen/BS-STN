import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import ChebConv, global_mean_pool as gap, global_max_pool as gmp
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])
args.device = 'cuda'
if torch.cuda.is_available():
    args.device = 'cuda:0'


class STGCN(torch.nn.Module):
    def __init__(self, args):
        super(STGCN, self).__init__()
        # Function graph view #
        self.convF1 = ChebConv(1025, 256, 2)
        self.convF2 = ChebConv(256, 64, 2)
        self.bnF1 = torch.nn.BatchNorm1d(256)
        self.bnF2 = torch.nn.BatchNorm1d(64)

        self.linF1 = torch.nn.Linear(128, 5)
        # Distance graph view #
        self.convD1 = ChebConv(1025, 256, 2)
        self.convD2 = ChebConv(256, 64, 2)
        self.bnD1 = torch.nn.BatchNorm1d(256)
        self.bnD2 = torch.nn.BatchNorm1d(64)

        self.linD1 = torch.nn.Linear(128, 5)
        # Fusion classifier
        self.lin1 = torch.nn.Linear(256, 5)
        # Domain classifier
        # self.discriminator = torch.nn.Linear(256, 1)
        # Learnable temporal and spatial position embeddings
        self.TembF = torch.nn.Parameter(torch.randn(3, 1025), requires_grad=True)
        self.SembF = torch.nn.Parameter(torch.randn(12, 1025), requires_grad=True)
        self.TembD = torch.nn.Parameter(torch.randn(3, 1025), requires_grad=True)
        self.SembD = torch.nn.Parameter(torch.randn(12, 1025), requires_grad=True)

        # Sharing parameter
        self.linF1.weight = torch.nn.Parameter(self.lin1.weight[:, :128].clone())
        self.linD1.weight = torch.nn.Parameter(self.lin1.weight[:, 128:].clone())
        self.linF1.bias = torch.nn.Parameter(self.lin1.bias.clone())
        self.linD1.bias = torch.nn.Parameter(self.lin1.bias.clone())

    def forward(self, dataF, dataD):
        xF, edge_index_F, batch_F, A_F = dataF.x, dataF.edge_index, dataF.batch, dataF.A
        xD, edge_index_D, batch_D, A_D = dataD.x, dataD.edge_index, dataD.batch, dataD.A

        idx = []
        stride = 36
        for start_row in range(12, xD.size(0), stride):
            idx.append(list(range(start_row, start_row + 12)))
        idx = torch.flatten(torch.tensor(idx))

        # Function graph forward propagation
        # Transform the matrix to [B, N, T, C]
        xF = xF.view(max(batch_F)+1, 3, 12, 1025).permute(0, 2, 1, 3)
        # Add temporal and spatial position embeddings
        xF = xF + self.TembF.unsqueeze(0).unsqueeze(1) + self.SembF.unsqueeze(0).unsqueeze(2)
        # Transform the matrix to [BNT, C]
        xF = xF.permute(0, 2, 1, 3).reshape((max(batch_F)+1)*36, 1025)

        xF = F.relu(self.bnF1(self.convF1(xF, edge_index_F)))
        xF = F.relu(self.bnF2(self.convF2(xF, edge_index_F)))
        xF_feature = torch.cat([gmp(xF[idx, :], batch_F[idx]), gap(xF[idx, :], batch_F[idx])], dim=1)
        # xF_feature = torch.cat([gmp(xF, batch_F), gap(xF, batch_F)], dim=1)

        outputF = F.softmax(self.linF1(xF_feature), dim=-1)

        # Distance graph forward propagation
        # Transform the matrix to [B, N, T, C]
        xD = xD.view(max(batch_D) + 1, 3, 12, 1025).permute(0, 2, 1, 3)
        # Add temporal and spatial position embeddings
        xD = xD + self.TembD.unsqueeze(0).unsqueeze(1) + self.SembD.unsqueeze(0).unsqueeze(2)
        # Transform the matrix to [BNT, C]
        xD = xD.permute(0, 2, 1, 3).reshape((max(batch_D) + 1) * 36, 1025)

        xD = F.relu(self.bnD1(self.convD1(xD, edge_index_D)))
        xD = F.relu(self.bnD2(self.convD2(xD, edge_index_D)))
        xD_feature = torch.cat([gmp(xD[idx, :], batch_D[idx]), gap(xD[idx, :], batch_D[idx])], dim=1)
        # xD_feature = torch.cat([gmp(xD, batch_D), gap(xD, batch_D)], dim=1)

        outputD = F.softmax(self.linD1(xD_feature), dim=-1)

        # Fusion feature forward propagation
        fusion = torch.cat([xF_feature, xD_feature], dim=1)
        print(np.shape(fusion))
        output = F.softmax(self.lin1(fusion), dim=-1)

        # Domain classify
        # fusion_reverse = GradientReversalLayer.apply(fusion, 1)
        # domain = F.sigmoid(self.discriminator(fusion_reverse))

        return outputF, outputD, fusion, output


if __name__ == '__main__':
    main()
