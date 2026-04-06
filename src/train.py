"""
ASTER: Attentional Scene Text Recognition
Training script
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from pathlib import Path

from config import get_config
from model import ASTER, ASTERLoss
from datasets import get_dataset, get_transforms
from utils import save_checkpoint, load_checkpoint, AverageMeter, calculate_accuracy


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """Train for one epoch"""
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        outputs, attention_weights, rectified, ctrl_points = model(
            images, targets=labels, teacher_forcing_ratio=0.5
        )

        # Calculate loss
        loss, loss_dict = criterion(outputs, labels, ctrl_points, lambda_smooth=0.01)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()

        # Calculate accuracy
        predictions = outputs.argmax(dim=2)
        accuracy = calculate_accuracy(predictions, labels)

        # Update metrics
        losses.update(loss.item(), images.size(0))
        accs.update(accuracy, images.size(0))

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{losses.avg:.4f}",
                "Acc": f"{accs.avg:.2f}%",
                "Rec_Loss": f"{loss_dict['recognition_loss']:.4f}",
            }
        )

        # Log to tensorboard
        if writer is not None and batch_idx % 100 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("train/loss", losses.avg, global_step)
            writer.add_scalar("train/accuracy", accs.avg, global_step)
            writer.add_scalar(
                "train/recognition_loss", loss_dict["recognition_loss"], global_step
            )
            writer.add_scalar(
                "train/smooth_loss", loss_dict["smooth_loss"], global_step
            )

    return losses.avg, accs.avg


def validate(model, dataloader, criterion, device, epoch, writer=None):
    """Validate the model"""
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")

        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            predictions, attention_weights, rectified, ctrl_points = model.predict(
                images
            )

            # For loss calculation, need outputs with logits
            outputs, _, _, _ = model(images, targets=labels, teacher_forcing_ratio=0.0)
            loss, loss_dict = criterion(
                outputs, labels, ctrl_points, lambda_smooth=0.01
            )

            # Calculate accuracy
            accuracy = calculate_accuracy(predictions, labels)

            losses.update(loss.item(), images.size(0))
            accs.update(accuracy, images.size(0))

            # Store for detailed analysis
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            pbar.set_postfix(
                {"Val_Loss": f"{losses.avg:.4f}", "Val_Acc": f"{accs.avg:.2f}%"}
            )

    # Log to tensorboard
    if writer is not None:
        writer.add_scalar("val/loss", losses.avg, epoch)
        writer.add_scalar("val/accuracy", accs.avg, epoch)

    return losses.avg, accs.avg


def main(args):
    """Main training function"""

    # Set random seed
    set_seed(args.seed)

    # Get configuration
    config = get_config(args.dataset)

    # Override config with command line arguments
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.epochs:
        config.NUM_EPOCHS = args.epochs

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Using device: {device}")

    # Create model
    print("\nCreating model...")
    model = ASTER(config)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create datasets
    print(f"\nLoading {args.dataset} dataset...")
    train_transform = get_transforms(
        is_training=True, img_height=config.IMG_HEIGHT, img_width=config.IMG_WIDTH
    )
    val_transform = get_transforms(
        is_training=False, img_height=config.IMG_HEIGHT, img_width=config.IMG_WIDTH
    )

    # Use synthetic dataset if data path doesn't exist
    data_path = Path(args.data_path) / args.dataset
    if not data_path.exists():
        print(f"Warning: {data_path} not found. Using synthetic dataset for testing.")
        from datasets import SyntheticTextGenerator

        train_dataset = SyntheticTextGenerator(
            num_samples=10000,
            charset=config.CHARACTERS,
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            max_length=config.MAX_SEQ_LENGTH,
            transform=train_transform,
        )
        val_dataset = SyntheticTextGenerator(
            num_samples=1000,
            charset=config.CHARACTERS,
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            max_length=config.MAX_SEQ_LENGTH,
            transform=val_transform,
        )
    else:
        train_dataset = get_dataset(
            args.dataset,
            data_path,
            config.CHARACTERS,
            split="train",
            transform=train_transform,
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            max_length=config.MAX_SEQ_LENGTH,
        )

        val_dataset = get_dataset(
            args.dataset,
            data_path,
            config.CHARACTERS,
            split="test",
            transform=val_transform,
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            max_length=config.MAX_SEQ_LENGTH,
        )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create loss function
    criterion = ASTERLoss(pad_token_idx=2)

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Create tensorboard writer
    writer = SummaryWriter(log_dir / args.exp_name)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = checkpoint_dir / f"{args.exp_name}_epoch_{epoch}.pth"
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": vars(config),
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_checkpoint_path = checkpoint_dir / f"{args.exp_name}_best.pth"
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": vars(config),
                },
                best_checkpoint_path,
            )
            print(f"Saved best model with val acc: {val_acc:.2f}%")

    writer.close()
    print("\nTraining completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASTER OCR Model")

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="Synthetic",
        choices=["Synth90K", "SynthText", "IIIT5K", "SVT", "IC13", "IC15", "Synthetic"],
        help="Dataset name",
    )
    parser.add_argument(
        "--data_path", type=str, default="./data", help="Path to dataset"
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save_freq", type=int, default=1, help="Save checkpoint every N epochs"
    )

    # Logging arguments
    parser.add_argument(
        "--exp_name", type=str, default="aster_ocr", help="Experiment name"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="Directory for tensorboard logs"
    )

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")

    args = parser.parse_args()

    main(args)
