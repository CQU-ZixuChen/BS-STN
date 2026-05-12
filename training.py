import torch
import scipy.io as scio
import time
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
from model import STGCN
import argparse
from Dataset import MyGraphDataset
from function import loss_func, test_func, StratifiedSampler
parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=100,
                    help='maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=5,
                    help='batch size')
parser.add_argument('--patience', type=int, default=100,
                    help='patience for earlystopping')

args = parser.parse_args(args=[])
args.device = 'cuda'
if torch.cuda.is_available():
    args.device = 'cuda:0'


y = scio.loadmat("y.mat") # y.mat is the label file, used to sample indices from it each time to construct training and test sets.
y = torch.tensor(y['y'], dtype=torch.long)
char1 = ['Graph_Function'] # char1 stores the filename of the function graph
char2 = ['Graph_Distance'] # char2 stores the filename of the distance graph

count = [1, 2, 3, 4, 5] # Represent 1/2/3/4/5 shot settings

for c1, c2, nn in zip(char1, char2, count):
    dataset1 = MyGraphDataset(c1)
    dataset2 = MyGraphDataset(c2)
    for num in range(5):
        sampler_train = StratifiedSampler(y, nn)
        indices_train = sampler_train.stratified_sample()
        train_dataset_F = Subset(dataset1, indices_train)
        train_dataset_D = Subset(dataset2, indices_train)

        sampler_test = StratifiedSampler(y, 150)
        indices_test = sampler_test.stratified_sample()
        test_dataset_F = Subset(dataset1, indices_test)
        test_dataset_D = Subset(dataset2, indices_test)

        indices = torch.randperm(len(train_dataset_F))
        train_loader_F = DataLoader(Subset(train_dataset_F, indices), batch_size=args.batch_size, shuffle=False)
        train_loader_D = DataLoader(Subset(train_dataset_D, indices), batch_size=args.batch_size, shuffle=False)
        test_loader_F = DataLoader(test_dataset_F, batch_size=int(len(test_dataset_F)), shuffle=False)
        test_loader_D = DataLoader(test_dataset_D, batch_size=int(len(test_dataset_D)), shuffle=False)

        model = STGCN(args).to(args.device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)
        min_loss = 1e10
        patience = 0
               for epoch in range(args.epochs):
            model.train()
            for i, data in enumerate(zip(train_loader_F, train_loader_D)):
                start_train = time.time()
                correct = 0.
                data1 = data[0].to(args.device)
                data2 = data[1].to(args.device)
                loss = loss_func(data1, data2, model)
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            if epoch == args.epochs - 1:
                torch.save(model.state_dict(), 'latest.pth')

            test_acc, pred_p, feature, label = test_func(model, test_loader_F, test_loader_D)

            print("Test accuarcy:{}".format(test_acc))


