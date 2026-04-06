"""
ASTER: Attentional Scene Text Recognition
Inference script for testing trained models
"""

import os
import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from config import get_config
from model import ASTER
from utils import load_checkpoint, decode_prediction


def preprocess_image(image_path, img_height=32, img_width=100):
    """
    Preprocess image for inference

    Args:
        image_path: path to image file
        img_height: target height
        img_width: target width

    Returns:
        image_tensor: preprocessed image tensor (1, C, H, W)
        original_image: original PIL Image
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()

    # Resize
    image = image.resize((img_width, img_height), Image.Resampling.BILINEAR)

    # Transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor, original_image


def inference(model, image_tensor, charset, max_length=25, device="cpu"):
    """
    Run inference on a single image

    Args:
        model: ASTER model
        image_tensor: preprocessed image tensor (1, C, H, W)
        charset: character set
        max_length: maximum sequence length
        device: device to run on

    Returns:
        text: predicted text string
        predictions: predicted indices
        rectified: rectified image
        attention_weights: attention weights
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        predictions, attention_weights, rectified, ctrl_points = model.predict(
            image_tensor, max_length=max_length
        )

    # Decode prediction
    text = decode_prediction(predictions[0], charset)

    return text, predictions[0], rectified, attention_weights


def visualize_results(image_path, text, rectified, attention_weights, save_path=None):
    """
    Visualize inference results

    Args:
        image_path: path to input image
        text: predicted text
        rectified: rectified image tensor
        attention_weights: attention weights
        save_path: path to save visualization
    """
    # Load original image
    original = Image.open(image_path).convert("RGB")

    # Create figure
    fig = plt.figure(figsize=(15, 5))

    # Original image
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(original)
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Rectified image
    ax2 = plt.subplot(1, 3, 2)
    rectified_np = rectified[0].cpu().permute(1, 2, 0).numpy()
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
    rectified_np = rectified_np * std.numpy() + mean.numpy()
    rectified_np = (rectified_np * 255).clip(0, 255).astype("uint8")
    ax2.imshow(rectified_np)
    ax2.set_title("Rectified Image")
    ax2.axis("off")

    # Attention visualization
    ax3 = plt.subplot(1, 3, 3)
    # Average attention weights over all time steps
    avg_attention = (
        torch.stack([aw for aw in attention_weights]).mean(dim=0)[0].cpu().numpy()
    )
    ax3.bar(range(len(avg_attention)), avg_attention)
    ax3.set_xlabel("Position")
    ax3.set_ylabel("Attention Weight")
    ax3.set_title("Attention Distribution")

    # Add text as figure title
    fig.suptitle(f'Predicted Text: "{text}"', fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def main(args):
    """Main inference function"""

    # Load configuration
    config = get_config(args.dataset)

    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = ASTER(config)
    model = model.to(device)

    # Load checkpoint
    load_checkpoint(args.checkpoint, model)
    print("Model loaded successfully!")

    # Process single image or directory
    if args.image:
        # Single image inference
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"\nProcessing image: {image_path}")

        # Preprocess
        image_tensor, original_image = preprocess_image(
            image_path, config.IMG_HEIGHT, config.IMG_WIDTH
        )

        # Inference
        text, predictions, rectified, attention_weights = inference(
            model,
            image_tensor,
            config.CHARACTERS,
            max_length=config.MAX_SEQ_LENGTH,
            device=device,
        )

        print(f"Predicted text: {text}")

        # Visualize if requested
        if args.visualize:
            save_path = (
                args.output / f"{image_path.stem}_result.png" if args.output else None
            )
            visualize_results(image_path, text, rectified, attention_weights, save_path)

    elif args.image_dir:
        # Directory inference
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")

        # Supported image extensions
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
        image_paths = [p for p in image_dir.iterdir() if p.suffix.lower() in extensions]

        print(f"\nProcessing {len(image_paths)} images from {image_dir}")

        results = []
        for image_path in image_paths:
            print(f"Processing: {image_path.name}", end=" -> ")

            # Preprocess
            image_tensor, _ = preprocess_image(
                image_path, config.IMG_HEIGHT, config.IMG_WIDTH
            )

            # Inference
            text, _, rectified, attention_weights = inference(
                model,
                image_tensor,
                config.CHARACTERS,
                max_length=config.MAX_SEQ_LENGTH,
                device=device,
            )

            print(f"'{text}'")
            results.append((image_path.name, text))

            # Visualize if requested
            if args.visualize and args.output:
                save_path = args.output / f"{image_path.stem}_result.png"
                visualize_results(
                    image_path, text, rectified, attention_weights, save_path
                )

        # Save results to file
        if args.output:
            results_file = args.output / "results.txt"
            with open(results_file, "w") as f:
                for name, text in results:
                    f.write(f"{name}\t{text}\n")
            print(f"\nSaved results to {results_file}")

    else:
        raise ValueError("Please provide either --image or --image_dir")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASTER OCR Inference")

    # Input arguments
    parser.add_argument(
        "--image", type=str, default=None, help="Path to single image file"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to directory containing images",
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Synth90K",
        choices=["Synth90K", "SynthText", "IIIT5K", "SVT", "IC13", "IC15", "Synthetic"],
        help="Dataset configuration to use",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save results and visualizations",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize attention and rectification"
    )

    # Other arguments
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")

    args = parser.parse_args()

    # Create output directory
    if args.output:
        args.output = Path(args.output)
        args.output.mkdir(parents=True, exist_ok=True)

    main(args)
