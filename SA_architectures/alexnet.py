import torch.nn as nn
import torch

class ANN(nn.Module):
    def __init__(self, num_classes, dropout, in_channels, args):
        super().__init__()
        self.init_channels = in_channels
        self.timesteps = args.timesteps
        self.layer = nn.Sequential(
            nn.Conv2d(self.init_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(1),

            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(1),

            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.ReLU(),
            nn.AvgPool2d(2))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * args.channels_ANN1 * args.channels_ANN2, num_classes)
        )

    def forward(self,x):
        x = x.permute(1, 0, 2, 3, 4)
        x_ = torch.zeros((self.timesteps,) + self.layer(x[0, ...]).shape, device=x.device)
        for step in range(self.timesteps):
            x_[step, ...] = self.layer(x[step, ...])
        out = torch.zeros((self.timesteps,) + self.classifier(x_[0, ...]).shape, device=x.device)
        for step in range(self.timesteps):
            out[step, ...] = self.classifier(x_[step, ...])
        # x = self.network(x)
        return out.mean(0)


class AlexNet(nn.Module):
    def __init__(self, num_classes, dropout, in_channels, args):
        super().__init__()
        self.init_channels = in_channels
        self.timesteps = args.timesteps
        self.layer = nn.Sequential(
            nn.Conv2d(self.init_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1),
            nn.Dropout(dropout),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1),
            nn.Dropout(dropout),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * args.channels_alexnet1 * args.channels_alexnet2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self,x):
        x = x.permute(1, 0, 2, 3, 4)
        x_ = torch.zeros((self.timesteps,) + self.layer(x[0, ...]).shape, device=x.device)
        for step in range(self.timesteps):
            x_[step, ...] = self.layer(x[step, ...])
        out = torch.zeros((self.timesteps,) + self.classifier(x_[0, ...]).shape, device=x.device)
        for step in range(self.timesteps):
            out[step, ...] = self.classifier(x_[step, ...])
        return out.mean(0)

        # out = self.layer1(x)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = self.classifier(out)
        # return out

def alexnet(num_classes=10, dropout=0, in_channels=5, args=None):
    return AlexNet(num_classes, dropout, in_channels, args)

def cnn(num_classes=10, dropout=0, in_channels=5, args=None):
    return ANN(num_classes, dropout, in_channels, args)