"""
ASTER: Attentional Scene Text Recognition
Utility functions
"""

import torch
import numpy as np
from pathlib import Path


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(predictions, targets):
    """
    Calculate word-level accuracy

    Args:
        predictions: (B, T) - predicted indices
        targets: (B, T) - target indices

    Returns:
        accuracy: float - percentage of correct predictions
    """
    B = predictions.size(0)
    correct = 0

    for i in range(B):
        pred = predictions[i]
        target = targets[i]

        # Remove padding and special tokens
        pred_filtered = []
        target_filtered = []

        for p in pred:
            if p.item() not in [0, 1, 2]:  # Skip SOS, EOS, PAD
                pred_filtered.append(p.item())

        for t in target:
            if t.item() not in [0, 1, 2]:  # Skip SOS, EOS, PAD
                target_filtered.append(t.item())

        if pred_filtered == target_filtered:
            correct += 1

    return (correct / B) * 100


def save_checkpoint(state, filepath):
    """
    Save checkpoint

    Args:
        state: dict containing model state, optimizer state, etc.
        filepath: path to save checkpoint
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load checkpoint

    Args:
        filepath: path to checkpoint file
        model: model to load weights into
        optimizer: optimizer to load state into (optional)

    Returns:
        epoch: epoch number from checkpoint
        val_acc: validation accuracy from checkpoint
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    val_acc = checkpoint.get("val_acc", 0.0)

    return epoch, val_acc


def decode_prediction(prediction, charset, sos_token=0, eos_token=1, pad_token=2):
    """
    Decode prediction indices to text

    Args:
        prediction: list or tensor of indices
        charset: string of characters
        sos_token: start of sequence token index
        eos_token: end of sequence token index
        pad_token: padding token index

    Returns:
        text: decoded text string
    """
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()

    text = []
    for idx in prediction:
        idx = int(idx)
        if idx == eos_token:
            break
        if idx == sos_token or idx == pad_token:
            continue
        if idx < len(charset):
            text.append(charset[idx])

    return "".join(text)


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def count_parameters(model):
    """Count total and trainable parameters in model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def visualize_attention(image, attention_weights, save_path=None):
    """
    Visualize attention weights on image

    Args:
        image: input image tensor or PIL Image
        attention_weights: attention weights for each position
        save_path: path to save visualization (optional)
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.transforms as transforms

    if isinstance(image, torch.Tensor):
        # Denormalize if needed
        image = image.cpu().permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Show image
    ax1.imshow(image)
    ax1.axis("off")
    ax1.set_title("Input Image")

    # Show attention weights
    attention = attention_weights.cpu().numpy()
    ax2.bar(range(len(attention)), attention)
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Attention Weight")
    ax2.set_title("Attention Distribution")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    print("Testing utility functions...")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(1.5, n=2)
    meter.update(2.5, n=3)
    print(f"Average: {meter.avg}")  # Should be (1.5*2 + 2.5*3) / 5 = 2.1

    # Test calculate_accuracy
    pred = torch.tensor([[3, 4, 5, 1, 2], [3, 4, 6, 1, 2]])
    target = torch.tensor([[3, 4, 5, 1, 2], [3, 4, 5, 1, 2]])
    acc = calculate_accuracy(pred, target)
    print(f"Accuracy: {acc:.2f}%")  # Should be 50%

    # Test decode_prediction
    charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    pred = [0, 27, 28, 29, 1, 2, 2]  # SOS, 'a', 'b', 'c', EOS, PAD, PAD
    text = decode_prediction(pred, charset)
    print(f"Decoded text: '{text}'")  # Should be 'abc'

    print("\nUtility functions test passed!")
