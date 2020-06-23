import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet(nn.Module):
    """ ResNet50
    """
    def __init__(self, num_classes=2, pretrained=False):
        super(ResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        modules = list(self.resnet.children())[:-1] # right after flatten
        self.extractor = nn.Sequential(
                            *modules
                        )
        self.classifier = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.resnet = None
        
    def forward(self, x):
        feature = self.extractor(x)
        feature = feature.view(feature.size(0), -1)
        classification = self.classifier(feature)
        return classification

