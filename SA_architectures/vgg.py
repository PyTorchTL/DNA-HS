import torch.nn as nn
import torch

cfg = {
    'VGG5': [64, 64, 64],
    'VGG9':  [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512]
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout, in_channels, args):
        super(VGG, self).__init__()
        self.init_channels = in_channels
        self.timesteps = args.timesteps
        self.layer = self._make_layers(cfg[vgg_name], dropout)
        if vgg_name == 'VGG5':
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.init_channels * args.input_shape1 * args.input_shape2, self.init_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.init_channels, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=1))
            elif x == 'A':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=1))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)
        x_ = torch.zeros((self.timesteps,) + self.layer(x[0, ...]).shape, device=x.device)
        for step in range(self.timesteps):
            x_[step, ...] = self.layer(x[step, ...])
        out = torch.zeros((self.timesteps,) + self.classifier(x_[0, ...]).shape, device=x.device)
        for step in range(self.timesteps):
            out[step, ...] = self.classifier(x_[step, ...])

        # out = self.layer(x)
        # out = self.classifier(out)
        return out.mean(0)


class VGG_normed(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout, in_channels):
        super(VGG_normed, self).__init__()
        self.init_channels = in_channels
        self.num_classes = num_classes
        self.module_list = self._make_layers(cfg[vgg_name], dropout)


    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif x == 'A':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
                self.init_channels = x
        layers.append(nn.Flatten())
        layers.append(nn.Linear(512, 4096))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(4096, 4096))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(4096, self.num_classes))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.module_list(x)



def vgg5(num_classes=10, dropout=0, in_channels=5, args=None, **kargs):
    return VGG('VGG5', num_classes, dropout, in_channels, args)


def vgg9(num_classes=10, dropout=0, in_channels=5, **kargs):
    return VGG('VGG9', num_classes, dropout, in_channels)


def vgg11(num_classes=10, dropout=0, in_channels=5, **kargs):
    return VGG('VGG11', num_classes, dropout, in_channels)


def vgg13(num_classes=10, dropout=0, in_channels=5, **kargs):
    return VGG('VGG13', num_classes, dropout, in_channels)


def vgg16(num_classes=10, dropout=0, in_channels=5, **kargs):
    return VGG('VGG16', num_classes, dropout, in_channels)


def vgg19(num_classes=10, dropout=0, in_channels=5, **kargs):
    return VGG('VGG19', num_classes, dropout, in_channels)


def vgg16_normed(num_classes=10, dropout=0, in_channels=5, **kargs):
    return VGG_normed('VGG16', num_classes, dropout, in_channels)