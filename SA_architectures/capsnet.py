"""
Some key layers used for constructing a Capsule Network. These layers can used to construct CapsNet on other dataset,
not just on MNIST.

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Pytorch`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    # torch.norm()求指定维度上的范数，p=2就是L2范数，dim（x）为缩减的维度
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs


class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.

    :param in_num_caps: 输入胶囊的数量
    :param in_dim_caps: 输入胶囊的长度（维数）
    :param out_num_caps: 输出胶囊的数量
    :param out_dim_caps: 输出胶囊的长度（维数）
    :param routings: 动态路由的迭代次数
    """
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        # 权值尺寸为[out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]，即每个输入和输出胶囊都有连接
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))

    def forward(self, x):
        # x[:, None, :, :, None]-- 扩展 x [batch, in_num_caps, in_dim_caps] → [batch, 1, in_num_caps, in_dim_caps, 1]
        # weight.size   = [out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: torch.matmul()将weight和扩展后的输入相乘
        # [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # 相乘结果： x_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps，1]
        # torch.squeeze 去除多余的维度，变成[batch, out_num_caps, in_num_caps, out_dim_caps]
        temp = torch.matmul(self.weight, x[:, None, :, :, None])
        x_hat = torch.squeeze(temp, dim=-1)

        # In forward pass, `x_hat_detached` = `x_hat`;
        # In backward, no gradient can flow from `x_hat_detached` back to `x_hat`.
        # 截断梯度反向传播
        x_hat_detached = x_hat.detach()
        # 这一部分结束后，每个输入胶囊都产生了out_num_caps个输出胶囊，
        # 目前共有in_num_caps * out_num_caps个胶囊

        # 动态路由过程！！！
        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)).cuda()

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            c = F.softmax(b, dim=1)
            # print('c:', c)
            # 在最后一次迭代，使用' x_hat '来计算'输出'，以便反向传播梯度
            if i == self.routings - 1:
                # 扩展 c.size：[batch, out_num_caps, in_num_caps, 1 ]
                # x_hat.size：[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1, out_dim_caps]
                # 在倒数第二维上求和
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat))  # 替代方法

            else:  # 否则，使用'x_hat_detached'来更新'b'。这条路上没有梯度流动。
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # 替代方法

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)

        return torch.squeeze(outputs, dim=-2)


class PrimaryCapsule(nn.Module):
    """
    Apply Conv2D with `out_channels` and then reshape to get capsules
    :param in_channels: input channels
    :param out_channels: output channels
    :param dim_caps: dimension of capsule
    :param kernel_size: kernel size
    :return: output tensor, size=[batch, num_caps, dim_caps]
    """
    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # outputs的形状是[batch，channels，w，h]
        outputs = self.conv2d(x)
        # Reshape函数里面-1指的是某个维度大小，使得变换维度后的变量和变换前的变量的总元素个数不变
        # 将4D的卷积输出变为3D的胶囊输出形式，output的形状为[batch, caps_num, dim_caps]，
        # 其中caps_num为胶囊数量，可自动计算；dim_caps为胶囊长度，需要预先指定。
        outputs = outputs.view(x.size(0), -1, self.dim_caps)
        return squash(outputs)


class CapsuleNet(nn.Module):

    def __init__(self, in_channels, channels, classes, routings, args):
        super(CapsuleNet, self).__init__()
        self.classes = classes
        self.routings = routings

        # Layer 1: Just a conventional Conv2D layer
        # input_size = [1, 27, 61]
        self.caps = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            PrimaryCapsule(channels, channels, 4, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            DenseCapsule(in_num_caps=args.num_capsnet, in_dim_caps=4,
                     out_num_caps=classes, out_dim_caps=8, routings=routings)
        )

        self.timesteps = args.timesteps

    def forward(self, x):

        x = x.permute(1, 0, 2, 3, 4)
        x_ = torch.zeros((self.timesteps,) + self.caps(x[0, ...]).shape, device=x.device)
        for step in range(self.timesteps):
            x_[step, ...] = self.caps(x[step, ...])

        return x_.norm(dim=-1).mean(0)



def capsnet(num_classes=10, dropout=0, in_channels=5, args=None):
    channels = args.channels
    routings = 3
    return CapsuleNet(in_channels, channels, num_classes, routings, args)