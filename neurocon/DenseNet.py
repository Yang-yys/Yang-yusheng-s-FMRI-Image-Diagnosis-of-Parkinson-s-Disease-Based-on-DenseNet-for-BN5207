import torch
import torch.nn as nn

# 定义 DenseBlock
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([nn.Sequential(
            nn.BatchNorm3d(in_channels + i * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels + i * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        ) for i in range(num_layers)])

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            x = layer(torch.cat(features, dim=1))
            features.append(x)
        return torch.cat(features, dim=1)

# 定义 Transition Layer
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

# 定义 DenseNet
class DenseNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, growth_rate=12, block_config=(6, 12, 24, 16)):
        super(DenseNet3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(2 * growth_rate),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        num_features = 2 * growth_rate
        self.dense_blocks = nn.ModuleList()
        self.trans_layers = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_features, growth_rate, num_layers)
            self.dense_blocks.append(block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, num_features // 2)
                self.trans_layers.append(trans)
                num_features //= 2
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        for block, trans in zip(self.dense_blocks, self.trans_layers):
            x = block(x)
            if trans is not None:
                x = trans(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
