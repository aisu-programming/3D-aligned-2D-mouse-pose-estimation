import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import ConvBlock, UpBlock


# Updated UNet Model with Embedding Head
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_filters, num_layers, expand_factor, num_groups, dropout_prob,
                 embed_dim=128):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        filters = [int(base_filters * expand_factor ** nl) // num_groups * num_groups for nl in range(num_layers)]

        self.encoder = nn.ModuleList()
        for i in range(len(filters)):
            if i == 0:
                self.encoder.append(nn.Sequential(
                    ConvBlock(n_channels, filters[i], num_groups, dropout_prob),
                    ConvBlock(filters[i], filters[i], num_groups, dropout_prob)
                ))
            else:
                self.encoder.append(nn.Sequential(
                    ConvBlock(filters[i - 1], filters[i], num_groups, dropout_prob),
                    ConvBlock(filters[i], filters[i], num_groups, dropout_prob)
                ))

        self.pool = nn.MaxPool2d(2)

        self.up_blocks = nn.ModuleList()
        for i in range(len(filters) - 1, 0, -1):
            in_channels = filters[i] + filters[i - 1]
            out_channels = filters[i - 1]
            self.up_blocks.append(UpBlock(in_channels, out_channels, num_groups, dropout_prob))

        self.out_conv = nn.Conv2d(filters[0], n_classes, kernel_size=1)

        # Embedding Head for Contrastive Learning
        self.embed_head = EnhancedEmbeddingHead()

        nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(filters[-1], embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, ret_rep=False):
        enc_features = []
        for enc in self.encoder:
            x = enc(x)
            enc_features.append(x)
            x = self.pool(x)

        x = enc_features[-1]  # Bottleneck

        for idx, up in enumerate(self.up_blocks):
            x = up(x, enc_features[-(idx + 2)])

        logits = self.out_conv(x)
        embeddings = F.normalize(self.embed_head(enc_features[-1]), dim=1)

        if not ret_rep:
            return logits
        else:
            return logits, embeddings


class EnhancedEmbeddingHead(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4)

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(0)  # Add batch dimension for attention
        x, _ = self.attention(x, x, x)
        return F.normalize(x.squeeze(0), dim=1)