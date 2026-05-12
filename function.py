import torch
import torch.nn.functional as F
import argparse
import numpy as np
from torch.utils.data import Dataset
parser = argparse.ArgumentParser()

args = parser.parse_args(args=[])
args.device = 'cuda'
if torch.cuda.is_available():
    args.device = 'cuda:0'

# StratifiedSampler is used to randomly sample from the full dataset to construct the training set and testing set.
# loss_func defined the classification loss function.
# test_func is used to perform inference on the testing set and calculate the test accuracy.

class StratifiedSampler:
    def __init__(self, labels, num_samples_per_class, train_size_ratio=0.8):
        self.labels = labels
        self.num_samples_per_class = num_samples_per_class
        self.train_size_ratio = train_size_ratio

        # Get all classes and convert them to integers in a Python list.
        self.unique_labels = list(torch.unique(torch.tensor(labels), sorted=True).numpy())

        # Used to store the indices of each class.
        self.class_indices = {label: [] for label in self.unique_labels}

        # Assign the index of each sample to its corresponding class.
        for i, label in enumerate(labels):
            self.class_indices[label.item()].append(i)

    def stratified_sample(self):
        indices = []

        # Loop through each class and randomly draw a specified number of samples from it.
        for label in self.unique_labels:
            indices.extend(label * len(self.class_indices[label]) + torch.randperm(len(self.class_indices[label]))[:self.num_samples_per_class])
        return indices


def loss_func(data1, data2, model):
    outputF1, outputD1, x1, output1 = model(data1, data2)
    weight = 2 * torch.sigmoid((1 - F.cosine_similarity(outputF1, outputD1, dim=1)))
    loss_cls = -torch.sum(weight * output1.gather(1, data1.y.view(-1, 1)))
    loss = loss_cls / len(data1.y)
    return loss


def test_func(model, loader1, loader2):
    model.eval()
    correct = 0.
    with torch.no_grad():
        for i, data in enumerate(zip(loader1, loader2)):
            data1 = data[0].to(args.device)
            data2 = data[1].to(args.device)
            outputF, outputD, x, output = model(data1, data2)
            pred_p = output.max(dim=1)[1]
            correct += pred_p.eq(data1.y).sum().item()
            x = x.cpu().detach()
    return correct / len(loader1.dataset), pred_p, x, data1.y
