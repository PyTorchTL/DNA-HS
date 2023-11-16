import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm, global_add_pool
from torcheeg.models import DGCNN


# https://github.com/neerajwagh/eeg-gcnn
# SEED
# class EEGGraphConvNet_SEED(nn.Module):
#     def __init__(self, sfreq=None):
#         super(EEGGraphConvNet_SEED, self).__init__()
#
#         # need these for train_model_and_visualize() function
#         self.sfreq = sfreq
#         self.input_size = 62
#
#         self.conv1 = GCNConv(42, 32, improved=True, cached=True, normalize=False)
#         self.conv2 = GCNConv(32, 20, improved=True, cached=True, normalize=False)
#         self.conv4_bn = BatchNorm(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#
#         self.fc_block1 = nn.Linear(20, 10)
#         self.fc_block2 = nn.Linear(10, 3)
#
#         # Xavier initializations  #init gcn layers
#         self.fc_block1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
#         self.fc_block2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
#
#     def forward(self, X_batch, return_features=False):
#         x, edge_index, edge_weight, batch = X_batch.x.float(), X_batch.edge_index, \
#                                             X_batch.edge_attr.float(), X_batch.batch
#
#         x = F.leaky_relu(self.conv1(x, edge_index, edge_weight))
#
#         x = F.leaky_relu(self.conv2(x, edge_index, edge_weight))
#
#         x = F.leaky_relu(self.conv4_bn(x))
#         out = global_add_pool(x, batch=batch)
#
#         out = F.dropout(out, p=0.2, training=self.training)
#         out = F.leaky_relu(self.fc_block1(out))
#         out = self.fc_block2(out)
#
#         features = torch.clone(x)
#
#         if return_features:
#             return out, features
#         else:
#             return out
#
# # Fatigue
# class EEGGraphConvNet(nn.Module):
#     def __init__(self, sfreq=None):
#         super(EEGGraphConvNet, self).__init__()
#
#         # need these for train_model_and_visualize() function
#         self.sfreq = sfreq
#         self.input_size = 61
#
#         self.conv1 = GCNConv(27, 32, improved=True, cached=True, normalize=False)
#         self.conv2 = GCNConv(32, 20, improved=True, cached=True, normalize=False)
#         self.conv4_bn = BatchNorm(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#
#         self.fc_block1 = nn.Linear(20, 10)
#         self.fc_block2 = nn.Linear(10, 2)
#
#         # Xavier initializations  #init gcn layers
#         self.fc_block1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
#         self.fc_block2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
#
#     def forward(self, X_batch, return_features=False):
#         x, edge_index, edge_weight, batch = X_batch.x.float(), X_batch.edge_index, \
#                                             X_batch.edge_attr.float(), X_batch.batch
#
#         x = F.leaky_relu(self.conv1(x, edge_index, edge_weight))
#
#         x = F.leaky_relu(self.conv2(x, edge_index, edge_weight))
#
#         x = F.leaky_relu(self.conv4_bn(x))
#         out = global_add_pool(x, batch=batch)
#         out = F.dropout(out, p=0.2, training=self.training)
#         out = F.leaky_relu(self.fc_block1(out))
#         out = self.fc_block2(out)
#
#         features = torch.clone(x)
#
#         if return_features:
#             return out, features
#         else:
#             return out
#
#
# def GCN(num_classes=10, dropout=0, in_channels=5, args=None):
#     return EEGGraphConvNet(in_channels=args.num_freq, num_electrodes=args.num_nodes, num_layers=2, hid_channels=32,
#                       num_classes=args.num_classes)

class GraphConvNet(nn.Module):

    def __init__(self, in_channels, num_classes, args):
        super(GraphConvNet, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.gcn = DGCNN(in_channels=in_channels, num_electrodes=args.channel_2Dto1D, num_layers=2, hid_channels=32, num_classes=num_classes)

        self.timesteps = args.timesteps

    def forward(self, x):

        x = x.permute(1, 0, 2, 3, 4)
        x_c = torch.zeros((self.timesteps,) + self.cnn(x[0, ...]).shape, device=x.device)
        for step in range(self.timesteps):
            x_c[step, ...] = self.cnn(x[step, ...])
        x_c = x_c.view(x_c.shape[0], x_c.shape[1], -1, x_c.shape[2])
        x_ = torch.zeros((self.timesteps,) + self.gcn(x_c[0, ...]).shape, device=x.device)
        for step in range(self.timesteps):
            x_[step, ...] = self.gcn(x_c[step, ...])

        return x_.mean(0)


def DGCN(num_classes=10, dropout=0, in_channels=5, args=None):
    return GraphConvNet(in_channels=in_channels, num_classes=num_classes, args=args)