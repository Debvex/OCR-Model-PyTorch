"""
ASTER: Attentional Scene Text Recognition
Complete ASTER Model combining all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rectification import TPSRectification
from feature_extractor import CNNFeatureExtractor, ResNetFeatureExtractor
from bidirectional_lstm import SequenceEncoder
from attention_decoder import AttentionDecoderV2


class ASTER(nn.Module):
    """
    Complete ASTER model for scene text recognition

    Architecture:
    1. TPS Rectification Network - Rectifies curved/distorted text
    2. CNN Feature Extractor - Extracts visual features
    3. BiLSTM Sequence Encoder - Models sequential dependencies
    4. Attention Decoder - Decodes to text with attention mechanism
    """

    def __init__(self, config):
        super(ASTER, self).__init__()

        self.config = config
        self.num_classes = config.NUM_CLASSES
        self.num_fiducial = config.NUM_FIDUCIAL
        self.img_height = config.IMG_HEIGHT
        self.img_width = config.IMG_WIDTH
        self.num_channels = config.NUM_CHANNELS
        self.hidden_size = config.HIDDEN_SIZE
        self.attention_dim = config.ATTENTION_DIM
        self.embedding_dim = config.EMBEDDING_DIM

        # 1. TPS Rectification Network
        self.rectification = TPSRectification(
            num_fiducial=self.num_fiducial,
            img_height=self.img_height,
            img_width=self.img_width,
            num_channels=self.num_channels,
        )

        # 2. CNN Feature Extractor
        if config.CNN == "ResNet":
            self.feature_extractor = ResNetFeatureExtractor(
                num_channels=self.num_channels, output_channels=512
            )
            cnn_output_channels = 512
        else:
            self.feature_extractor = CNNFeatureExtractor(
                num_channels=self.num_channels, num_filters=512
            )
            cnn_output_channels = 512

        # 3. Sequence Encoder (BiLSTM)
        self.sequence_encoder = SequenceEncoder(
            input_channels=cnn_output_channels,
            hidden_size=self.hidden_size,
            num_layers=2,
        )

        # 4. Attention Decoder
        encoder_dim = self.sequence_encoder.output_size
        self.decoder = AttentionDecoderV2(
            num_classes=self.num_classes,
            encoder_dim=encoder_dim,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_size,
            attention_dim=self.attention_dim,
            dropout=0.5,
        )

    def forward(self, images, targets=None, teacher_forcing_ratio=0.5):
        """
        Forward pass through complete ASTER model

        Args:
            images: Input images (B, C, H, W) - can be distorted/curved text
            targets: Ground truth labels (B, max_length) - for training with teacher forcing
            teacher_forcing_ratio: Probability of using teacher forcing during training

        Returns:
            outputs: (B, max_length, num_classes) - prediction logits
            attention_weights: list of attention weights
            rectified: Rectified images
            ctrl_points: Predicted control points
        """
        # 1. Rectification
        rectified, ctrl_points = self.rectification(images)

        # 2. Feature Extraction
        features = self.feature_extractor(rectified)  # (B, C, H', W')

        # 3. Sequence Encoding
        sequence_features = self.sequence_encoder(features)  # (B, T, encoder_dim)

        # 4. Decoding
        outputs, attention_weights = self.decoder(
            sequence_features,
            targets=targets,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        return outputs, attention_weights, rectified, ctrl_points

    def predict(self, images, max_length=25):
        """
        Prediction mode (inference)

        Args:
            images: Input images (B, C, H, W)
            max_length: Maximum sequence length

        Returns:
            predictions: (B, max_length) - predicted token indices
            attention_weights: list of attention weights
            rectified: Rectified images
            ctrl_points: Predicted control points
        """
        with torch.no_grad():
            # Rectification
            rectified, ctrl_points = self.rectification(images)

            # Feature Extraction
            features = self.feature_extractor(rectified)

            # Sequence Encoding
            sequence_features = self.sequence_encoder(features)

            # Greedy Decoding
            predictions, attention_weights = self.decoder.greedy_decode(
                sequence_features, max_length=max_length
            )

        return predictions, attention_weights, rectified, ctrl_points

    def decode_predictions(self, predictions, charset):
        """
        Decode predicted indices to text strings

        Args:
            predictions: (B, T) - predicted token indices
            charset: Character set string

        Returns:
            texts: list of decoded strings
        """
        texts = []
        for pred in predictions:
            text = []
            for idx in pred:
                idx_item = idx.item()
                if idx_item < len(charset):
                    text.append(charset[idx_item])
                else:
                    break
            texts.append("".join(text))
        return texts


class ASTERLoss(nn.Module):
    """
    Combined loss for ASTER training
    - Cross-entropy loss for recognition
    - Optional: smoothness loss for rectification
    """

    def __init__(self, pad_token_idx=2):
        super(ASTERLoss, self).__init__()
        self.pad_token_idx = pad_token_idx
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_token_idx, reduction="mean")

    def forward(self, predictions, targets, ctrl_points=None, lambda_smooth=0.0):
        """
        Calculate loss

        Args:
            predictions: (B, T, num_classes) - model predictions
            targets: (B, T) - ground truth labels
            ctrl_points: (B, num_fiducial, 2) - control points for smoothness loss
            lambda_smooth: Weight for smoothness loss

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        # Reshape for cross-entropy
        B, T, C = predictions.size()
        predictions_flat = predictions.view(-1, C)  # (B*T, C)
        targets_flat = targets.view(-1)  # (B*T,)

        # Recognition loss
        recognition_loss = self.ce_loss(predictions_flat, targets_flat)

        # Smoothness loss for rectification (encourage smooth transformations)
        smooth_loss = torch.tensor(0.0, device=predictions.device)
        if ctrl_points is not None and lambda_smooth > 0:
            # Calculate difference between consecutive control points
            diff = ctrl_points[:, 1:, :] - ctrl_points[:, :-1, :]
            smooth_loss = torch.mean(diff**2)

        # Total loss
        total_loss = recognition_loss + lambda_smooth * smooth_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "recognition_loss": recognition_loss.item(),
            "smooth_loss": smooth_loss.item()
            if isinstance(smooth_loss, torch.Tensor)
            else smooth_loss,
        }

        return total_loss, loss_dict


if __name__ == "__main__":
    print("Testing Complete ASTER Model...")

    from config import Config

    # Create config
    config = Config()

    # Create model
    model = ASTER(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print("\n1. Testing forward pass (training mode):")
    B, C, H, W = 2, 3, 32, 100
    images = torch.randn(B, C, H, W)
    targets = torch.randint(0, config.NUM_CLASSES, (B, 25))

    outputs, attention_weights, rectified, ctrl_points = model(
        images, targets, teacher_forcing_ratio=0.5
    )

    print(f"Input images shape: {images.shape}")
    print(f"Rectified shape: {rectified.shape}")
    print(f"Control points shape: {ctrl_points.shape}")
    print(f"Outputs shape: {outputs.shape}")
    print(f"Expected: (B, T, num_classes) = ({B}, 25, {config.NUM_CLASSES})")

    # Test prediction mode
    print("\n2. Testing prediction mode:")
    predictions, attention_weights, rectified, ctrl_points = model.predict(
        images, max_length=25
    )
    print(f"Predictions shape: {predictions.shape}")
    print(f"Expected: (B, max_length) = ({B}, 25)")

    # Test loss
    print("\n3. Testing loss function:")
    loss_fn = ASTERLoss()
    total_loss, loss_dict = loss_fn(outputs, targets, ctrl_points, lambda_smooth=0.01)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Recognition loss: {loss_dict['recognition_loss']:.4f}")
    print(f"Smooth loss: {loss_dict['smooth_loss']:.4f}")

    # Test decode predictions
    print("\n4. Testing decode predictions:")
    decoded_texts = model.decode_predictions(predictions, config.CHARACTERS)
    print(f"Sample decoded text: {decoded_texts[0][:20]}...")

    print("\nASTER Model test passed!")
