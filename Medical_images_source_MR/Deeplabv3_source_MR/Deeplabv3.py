# num_filter 提到 16；并且 减少一层深度

import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16_3D(nn.Module):
    def __init__(self):
        super(VGG16_3D, self).__init__()

        self.features = nn.Sequential(
            # block1
            nn.Conv3d(1, 64, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            # block2
            nn.Conv3d(64, 128, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            # block3
            nn.Conv3d(128, 256, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            # block4
            nn.Conv3d(256, 512, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            
            # block5
            nn.Conv3d(512, 512, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )


    def forward(self, x):
        x = self.features(x)
        
        
class VGG16_Backbone(nn.Module):
    def __init__(self, dropout=None):
        super(VGG16_Backbone, self).__init__()

        original = VGG16_3D()
        layers = []
        block_counter = 1
        for idx, layer in enumerate(original.features.children()):
            if isinstance(layer, nn.Conv3d):
                if block_counter == 4:
                    layer = nn.Conv3d(layer.in_channels, layer.out_channels, layer.kernel_size,
                                      padding=2, dilation=2, bias=layer.bias is not None)
                elif block_counter == 5:
                    layer = nn.Conv3d(layer.in_channels, layer.out_channels, layer.kernel_size,
                                      padding=4, dilation=4, bias=layer.bias is not None)

            if not (isinstance(layer, nn.MaxPool3d) and (block_counter == 4 or block_counter == 5)):
                    layers.append(layer)

            if isinstance(layer, nn.MaxPool3d):
                block_counter += 1

        self.backbone = nn.Sequential(*layers)

        if dropout is not None:
            for idx, layer in enumerate(self.backbone.children()):
                if isinstance(layer, nn.Conv3d):
                    self.backbone[idx] = nn.Sequential(layer, nn.Dropout(dropout))

    def forward(self, x):
        return self.backbone(x)
    
    
    
class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPModule, self).__init__()
        self.padding = dilation
        self.kernel_size = 3
        self.dilation = dilation
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=self.kernel_size, padding=0, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        padding = ((self.kernel_size - 1) * self.dilation) // 2
        x = F.pad(x, (padding, padding, padding, padding, padding, padding))
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x
    
    
    
class DeepLabV3(nn.Module):
    def __init__(self, num_classes=5, backbone='vgg16', activation=None):
        super(DeepLabV3, self).__init__()
        assert backbone in ['vgg16']

        if backbone == 'vgg16':
            self.backbone = VGG16_Backbone(dropout=None)  # You need to ensure VGG16_Backbone is the 3D version

        self.aspp1 = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        self.aspp_modules = nn.ModuleList([
            ASPPModule(512, 256, dilation=12),
            ASPPModule(512, 256, dilation=24),
            ASPPModule(512, 256, dilation=36)
        ])

        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        self.concat = nn.Sequential(
            nn.Conv3d(1280, 256, kernel_size=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.project = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.post_project = nn.Sequential(
            nn.ConstantPad3d(1, 0),
            nn.Conv3d(256, 256, kernel_size=3, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, num_classes, kernel_size=1)
        )

        self.activation = activation

    def forward(self, x):
        img_size = (x.size(2), x.size(3), x.size(4))

        x = self.backbone(x)
        x1 = self.aspp1(x)

        aspp_outputs = [x1]
        for aspp_module in self.aspp_modules:
            aspp_outputs.append(aspp_module(x))

        x5 = self.global_pooling(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='trilinear', align_corners=False)
        aspp_outputs.append(x5)

        x = torch.cat(aspp_outputs, dim=1)
        x = self.concat(x)
        x = self.project(x)
        x = self.post_project(x)

        x = F.interpolate(x, size=img_size, mode='trilinear', align_corners=False)

        if self.activation is not None:
            x = self.activation(x)

        return x