import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class CNN_Feature_Extractor_pretrained(nn.Module):
    def __init__(self):
        super(CNN_Feature_Extractor_pretrained, self).__init__()

        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.fc = nn.Linear(2048, 512)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
