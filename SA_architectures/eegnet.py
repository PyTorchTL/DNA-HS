import torch.nn as nn
import torch


class EEGNets(nn.Module):
    def __init__(self, input_channels, channels: int, classes, arg):
        super(EEGNets, self).__init__()
        self.drop_out = 0.5

        self.block = nn.Sequential(  # (32, 30(5*6), 9, 9)
            # left, right, up, bottom
            nn.ZeroPad2d((8, 9, 0, 0)),
            nn.Conv2d(
                in_channels=input_channels,  # input shape (1, C=32, T=42)
                out_channels=channels,  # num_filters
                # out_channels=32*4,  # num_filters
                kernel_size=(1, 18),  # filter size
                bias=False
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(channels),  # output shape (8, C=32, T=42)

            # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
            nn.Conv2d(
                in_channels=channels,  # input shape (8, C, T)
                out_channels=channels,  # num_filters
                kernel_size=(arg.input_shape1, 1),  # filter size
                groups=channels,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(channels),  # output shape (16, 1, T)
            nn.ELU(),

            # block3
            nn.ZeroPad2d((4, 4, 0, 0)),
            nn.Conv2d(
                in_channels=channels,  # input shape (16, 1, T//4)
                out_channels=channels,  # num_filters
                kernel_size=(1, 8),  # filter size
                groups=channels,
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=channels,  # input shape (16, 1, T//4)
                out_channels=channels,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(channels),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 2)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 1 * arg.channels_eegnet, classes, bias=False)
        )
        self.channels = channels
        self.timesteps = arg.timesteps

    def forward(self, x):

        x = x.permute(1, 0, 2, 3, 4)
        x_ = torch.zeros((self.timesteps,) + self.block(x[0, ...]).shape, device=x.device)
        for step in range(self.timesteps):
            x_[step, ...] = self.block(x[step, ...])
        out = torch.zeros((self.timesteps,) + self.out(x_[0, ...]).shape, device=x.device)
        for step in range(self.timesteps):
            out[step, ...] = self.out(x_[step, ...])

        return out.mean(0)


def EEGNet(num_classes=10, dropout=0, in_channels=5, args=None):
    channels = args.channels
    return EEGNets(in_channels, channels, num_classes, args)
