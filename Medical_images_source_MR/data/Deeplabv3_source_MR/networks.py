import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class classifier(nn.Module):
    def __init__(self, num_classes):
        super(classifier, self).__init__()
        
        self.conv = nn.Conv3d(num_classes, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        logits = self.conv(x)
        
        return logits