import torch.nn as nn
from torchvision import models


class SpatialEncoderCNN(nn.Module):
    def __init__(self, trainable=False):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        if not trainable:
            for parameter in self.resnet.parameters():
                parameter.requires_grad = False

    def forward(self, images):
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        features = features.permute(0, 2, 3, 1)
        return features.reshape(features.size(0), -1, features.size(-1))
