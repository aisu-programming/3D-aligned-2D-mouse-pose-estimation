import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    卷积块：Conv2d -> GroupNorm -> ReLU -> Dropout
    """
    def __init__(self, in_channels, out_channels, num_groups=8, dropout_prob=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class UpBlock(nn.Module):
    """
    上采样块：Upsample -> ConvBlock
    """
    def __init__(self, in_channels, out_channels, num_groups=8, dropout_prob=0.1):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels, num_groups, dropout_prob),
            ConvBlock(out_channels, out_channels, num_groups, dropout_prob)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 确保尺寸匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_filters, num_layers, expand_factor, num_groups, dropout_prob):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        filters = [ int(base_filters * expand_factor**nl) // num_groups * num_groups
                    for nl in range(num_layers) ]

        self.encoder = nn.ModuleList()
        for i in range(len(filters)):
            if i == 0:
                self.encoder.append(nn.Sequential(
                    ConvBlock(n_channels, filters[i], num_groups, dropout_prob),
                    ConvBlock(filters[i], filters[i], num_groups, dropout_prob)
                ))
            else:
                self.encoder.append(nn.Sequential(
                    ConvBlock(filters[i-1], filters[i], num_groups, dropout_prob),
                    ConvBlock(filters[i], filters[i], num_groups, dropout_prob)
                ))

        self.pool = nn.MaxPool2d(2)

        self.up_blocks = nn.ModuleList()
        for i in range(len(filters)-1, 0, -1):
            in_channels = filters[i] + filters[i-1]
            out_channels = filters[i-1]
            self.up_blocks.append(UpBlock(in_channels, out_channels, num_groups, dropout_prob))

        self.out_conv = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def forward(self, x):
        # 编码器路径
        enc_features = []
        for enc in self.encoder:
            x = enc(x)
            enc_features.append(x)
            x = self.pool(x)

        # 最后一层编码器的输出
        x = enc_features[-1]

        # 解码器路径
        for idx, up in enumerate(self.up_blocks):
            x = up(x, enc_features[-(idx+2)])

        # 输出层
        logits = self.out_conv(x)
        return logits
