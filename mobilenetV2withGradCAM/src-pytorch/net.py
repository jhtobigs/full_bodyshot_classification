import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class MobileNetV2(nn.Module):
    """ MobileNetV2
    """
    def __init__(self, num_classes=2, pretrained=False):
        super(MobileNetV2, self).__init__()
        self.moblienet = models.mobilenet_v2(pretrained=pretrained)
        self.features_conv = self.moblienet.features
        
        self.classifier = nn.Sequential(
            nn.Linear( 8960 * 7, self.moblienet.last_channel)
            , nn.Linear(self.moblienet.last_channel, num_classes)
        )
   
        self.moblienet = None
        self.gradients = None

        
    def forward(self, x):
        feature = self.features_conv(x)
        h = feature.register_hook(self.activations_hook)
        feature = feature.view(feature.size(0), -1)
        classification = self.classifier(feature)
        return classification

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)