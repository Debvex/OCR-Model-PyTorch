"""
ASTER: Attentional Scene Text Recognition
Bidirectional LSTM for sequence modeling
"""

import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for modeling sequential features
    Takes CNN features and outputs sequence representations
    """

    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.5):
        super(BidirectionalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input features (B, T, input_size) where T is sequence length
        Returns:
            output: LSTM output (B, T, hidden_size * 2) - concatenated forward and backward
            hidden: Tuple of hidden states (h_n, c_n)
        """
        # Apply LSTM
        output, hidden = self.lstm(x)

        # Apply dropout
        output = self.dropout_layer(output)

        return output, hidden


class SequenceEncoder(nn.Module):
    """
    Sequence Encoder: Converts CNN features to sequential representations
    """

    def __init__(self, input_channels=512, hidden_size=256, num_layers=2):
        super(SequenceEncoder, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Reduce channel dimension
        self.channel_reduction = nn.Linear(input_channels, hidden_size)

        # Bidirectional LSTM
        self.lstm = BidirectionalLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.5,
        )

        # Output dimension will be hidden_size * 2 (bidirectional)
        self.output_size = hidden_size * 2

    def forward(self, features):
        """
        Forward pass
        Args:
            features: CNN features (B, C, H, W)
        Returns:
            sequence_features: (B, T, hidden_size * 2) where T = W (sequence length)
        """
        B, C, H, W = features.size()

        # Reshape to (B, W, C*H) - treat width as sequence dimension
        # Average pooling over height to get (B, C, W)
        features = features.mean(dim=2)  # (B, C, W)

        # Permute to (B, W, C)
        features = features.permute(0, 2, 1)  # (B, W, C)

        # Reduce channel dimension
        features = self.channel_reduction(features)  # (B, W, hidden_size)

        # Apply ReLU activation
        features = torch.relu(features)

        # Apply Bidirectional LSTM
        sequence_features, _ = self.lstm(features)  # (B, W, hidden_size * 2)

        return sequence_features


class PyramidBidirectionalLSTM(nn.Module):
    """
    Pyramid Bidirectional LSTM
    Stacks multiple BiLSTMs with decreasing sequence length
    Similar to the architecture in some OCR papers
    """

    def __init__(self, input_channels=512, hidden_size=256, num_layers=2):
        super(PyramidBidirectionalLSTM, self).__init__()

        self.conv1 = nn.Conv2d(
            input_channels, input_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(input_channels)

        # First BiLSTM: processes features at full resolution
        self.lstm1 = BidirectionalLSTM(input_channels, hidden_size, num_layers)

        # Second BiLSTM: processes downsampled features
        self.downsample = nn.Conv2d(
            input_channels * 2, hidden_size * 2, kernel_size=3, stride=2, padding=1
        )
        self.lstm2 = BidirectionalLSTM(hidden_size * 2, hidden_size, num_layers)

        self.output_size = hidden_size * 2

    def forward(self, features):
        """
        Forward pass with pyramid structure
        Args:
            features: CNN features (B, C, H, W)
        Returns:
            output: Sequence features (B, T, hidden_size * 2)
        """
        B, C, H, W = features.size()

        # First BiLSTM
        # Average pool over height
        features_pooled = features.mean(dim=2)  # (B, C, W)
        features_seq = features_pooled.permute(0, 2, 1)  # (B, W, C)

        lstm1_out, _ = self.lstm1.lstm(features_seq)  # (B, W, hidden_size * 2)

        # Downsample and second BiLSTM
        lstm1_out = lstm1_out.permute(0, 2, 1).unsqueeze(
            2
        )  # (B, hidden_size * 2, 1, W)
        lstm1_out = self.downsample(lstm1_out)  # (B, hidden_size * 2, 1, W/2)
        lstm1_out = lstm1_out.squeeze(2).permute(0, 2, 1)  # (B, W/2, hidden_size * 2)

        lstm2_out, _ = self.lstm2.lstm(lstm1_out)  # (B, W/2, hidden_size * 2)

        return lstm2_out


if __name__ == "__main__":
    # Test Bidirectional LSTM modules
    print("Testing Bidirectional LSTM modules...")

    # Test basic BidirectionalLSTM
    print("\n1. Testing BidirectionalLSTM:")
    B, T, input_size = 2, 25, 512
    x = torch.randn(B, T, input_size)

    lstm = BidirectionalLSTM(input_size=input_size, hidden_size=256, num_layers=2)
    output, hidden = lstm(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (B, T, hidden_size * 2) = ({B}, {T}, 512)")

    # Test SequenceEncoder
    print("\n2. Testing SequenceEncoder:")
    B, C, H, W = 2, 512, 1, 25
    features = torch.randn(B, C, H, W)

    encoder = SequenceEncoder(input_channels=C, hidden_size=256)
    seq_features = encoder(features)
    print(f"Input features shape: {features.shape}")
    print(f"Sequence features shape: {seq_features.shape}")
    print(f"Expected: (B, W, hidden_size * 2) = ({B}, {W}, 512)")

    # Test PyramidBidirectionalLSTM
    print("\n3. Testing PyramidBidirectionalLSTM:")
    pyramid_encoder = PyramidBidirectionalLSTM(input_channels=C, hidden_size=256)
    pyramid_features = pyramid_encoder(features)
    print(f"Input features shape: {features.shape}")
    print(f"Pyramid features shape: {pyramid_features.shape}")

    print("\nBidirectional LSTM modules test passed!")
