import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
import scipy.io as scio
import numpy as np

# This code is used to construct your graph dataset. 
# You need to prepare the following files:
# 1. Edge index matrix of your graph dataset --- edge_index.mat
# 2. Node embedding matrix of your graph dataset --- x.mat
# 3. Adjacency matrix of your graph dataset --- A.mat

class MyGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=True):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    #返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        for i in range(The number of graph samples):
            edge_index = scio.loadmat("The file path of edge_index")
            edge_index = torch.tensor(edge_index['edge_index'], dtype=torch.long)

            x = scio.loadmat("The file path of node embedding matrix")
            x = torch.tensor(x['x'], dtype=torch.float)

            y = scio.loadmat("The file path of graph labels")
            y = torch.tensor(y['y'], dtype=torch.long)

            A = scio.loadmat("The file path of adjacency matrix")
            A = torch.tensor(A['A'], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y[i], A=A)

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# Run the main function, and it returns the constructed graph dataset
def main():
    b = MyGraphDataset("Name of your graph dataset")

    print()
    print(f'Dataset: {b}:')
    print('====================')
    print(f'Number of graphs: {len(b)}')
    print(f'Number of features: {b.num_features}')
    print(f'Number of classes: {b.num_classes}')

    data = b[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


if __name__ == '__main__':
    main()
