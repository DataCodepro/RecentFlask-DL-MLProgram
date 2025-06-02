import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
class DualIrisNet(nn.Module):
    def __init__(self):
        super(DualIrisNet, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # Remove final classification layer

        # Feature size from ResNet18
        self.shared_dim = 512

        # Biometric classification head
        self.biometric_head = nn.Sequential(
            nn.Linear(self.shared_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        # Diabetes risk head
        self.diabetes_head = nn.Sequential(
            nn.Linear(self.shared_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        bio_out = self.biometric_head(features)
        diab_out = self.diabetes_head(features)
        return bio_out, diab_out
