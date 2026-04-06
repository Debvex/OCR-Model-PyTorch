"""
ASTER: Attentional Scene Text Recognition
Feature Extractor using ResNet architecture
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BasicBlock(nn.Module):
    """Basic ResNet block"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-based feature extractor for ASTER
    Extracts visual features from rectified text images
    """

    def __init__(self, num_channels=3, output_channels=512):
        super(ResNetFeatureExtractor, self).__init__()
        self.num_channels = num_channels
        self.output_channels = output_channels

        # Initial conv layer
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)   # Output: 64 channels
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # Output: 128 channels
        self.layer3 = self._make_layer(128, 256, 2, stride=2) # Output: 256 channels
        self.layer4 = self._make_layer(256, output_channels, 2, stride=2) # Output: output_channels

        # Adaptive pooling to get consistent feature map size
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a ResNet layer with multiple blocks"""
        downsample = None
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input image (B, C, H, W) - typically rectified text
        Returns:
            features: Feature maps (B, output_channels, H', W')
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # For text recognition, we want 2D feature maps
        # H' = H / 16, W' = W / 16 approximately
        # We don't apply avgpool here to preserve spatial information
        return x


class CNNFeatureExtractor(nn.Module):
    """
    Alternative CNN feature extractor (similar to ASTER paper)
    Using VGG-style architecture
    """

    def __init__(self, num_channels=3, num_filters=512):
        super(CNNFeatureExtractor, self).__init__()

        self.features = nn.Sequential(
            # Block 1: (B, 3, 32, 100) -> (B, 64, 16, 50)
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: (B, 64, 16, 50) -> (B, 128, 8, 25)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: (B, 128, 8, 25) -> (B, 256, 4, 13)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 1)),

            # Block 4: (B, 256, 4, 13) -> (B, 512, 2, 7)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: (B, 512, 2, 7) -> (B, 512, 1, 7)
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.output_channels = num_filters

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input image (B, C, H, W)
        Returns:
            features: Feature maps (B, 512, H', W')
        """
        return self.features(x)


if __name__ == "__main__":
    # Test feature extractors
    print("Testing Feature Extractors...")

    # Test ResNet feature extractor
    print("\n1. Testing ResNet Feature Extractor:")
    B, C, H, W = 2, 3, 32, 100
    x = torch.randn(B, C, H, W)

    resnet_extractor = ResNetFeatureExtractor(num_channels=C, output_channels=512)
    resnet_features = resnet_extractor(x)
    print(f"Input shape: {x.shape}")
    print(f"ResNet features shape: {resnet_features.shape}")

    # Test CNN feature extractor
    print("\n2. Testing VGG-style CNN Feature Extractor:")
    cnn_extractor = CNNFeatureExtractor(num_channels=C, num_filters=512)
    cnn_features = cnn_extractor(x)
    print(f"Input shape: {x.shape}")
    print(f"CNN features shape: {cnn_features.shape}")

    print("\nFeature Extractors test passed!")
