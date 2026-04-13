# ASTER: Attentional Scene Text Recognition

A complete PyTorch implementation of ASTER (Attentional Scene Text Recognition) with step-by-step architecture documentation and support for multiple training datasets.

## Overview

ASTER is an end-to-end scene text recognition system that rectifies curved or distorted text and recognizes it using an attention-based encoder-decoder architecture.

### Architecture Flow

```
Input Image (Curved Text)
    ↓
TPS Rectification Network (20 fiducial points)
    ↓
Rectified Image (Straight Text)
    ↓
CNN Feature Extractor (ResNet-based)
    ↓
Feature Maps (512 channels)
    ↓
Bidirectional LSTM (2 layers, 256 hidden units)
    ↓
Sequential Features (512 dimensions)
    ↓
Attention Decoder (Bahdanau attention)
    ↓
Predicted Text
```

## Project Structure

```
.
├── config.py                      # Configuration classes
├── rectification.py               # TPS Rectification Network
├── feature_extractor.py           # CNN Feature Extractor (ResNet/VGG)
├── bidirectional_lstm.py          # BiLSTM Sequence Encoder
├── attention_decoder.py           # Attention-based Decoder
├── model.py                       # Complete ASTER Model
├── datasets.py                    # Dataset classes
├── train.py                       # Training script
├── inference.py                   # Inference script
├── utils.py                       # Utility functions
└── README.md                      # This file
```

## Features

### 1. TPS Rectification Network
- **Localization Network**: Predicts 20 fiducial points using CNN
- **Grid Generator**: Creates sampling grid using TPS transformation
- **Differentiable**: Fully differentiable rectification for end-to-end training
- **Input**: Distorted/curved text images
- **Output**: Horizontally rectified text

### 2. CNN Feature Extractor
- **ResNet-based**: Uses ResNet-18 style architecture
- **Alternative**: VGG-style architecture also available
- **Output**: Feature maps maintaining spatial information

### 3. Bidirectional LSTM
- **Sequence modeling**: Processes features left-to-right and right-to-left
- **Context-aware**: Provides context-rich representations
- **Pyramid variant**: Optional pyramid architecture for multi-scale features

### 4. Attention Decoder
- **Bahdanau attention**: Computes attention weights dynamically
- **Dual LSTM**: Separate attention and character LSTMs
- **Teacher forcing**: Supports teacher forcing during training
- **Greedy decoding**: For inference

## Supported Datasets

1. **Synth90K** (~7M images) - Synthetic word images
2. **SynthText** (~8M images) - Synthetic text images
3. **IIIT5K** (5K images) - Real scene text
4. **SVT** (349 images) - Street View Text
5. **IC13** (1,095 images) - ICDAR 2013
6. **IC15** (2,077 images) - ICDAR 2015
7. **Synthetic** - On-the-fly generated for testing

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd aster-ocr

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (if available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.62.0
tensorboard>=2.10.0
scipy>=1.7.0
```

## Usage

### Quick Start with Synthetic Data

```bash
# Train with synthetic data (for testing)
python train.py --dataset Synthetic --epochs 10 --batch_size 32

# Run inference
python inference.py --image path/to/image.jpg --checkpoint checkpoints/aster_best.pth
```

### Training on Real Datasets

```bash
# Train on Synth90K
python train.py --dataset Synth90K --data_path ./data/Synth90K --epochs 100

# Train on multiple datasets (sequential training)
# First, pre-train on synthetic data
python train.py --dataset Synth90K --epochs 50 --exp_name aster_pretrain

# Then fine-tune on real data
python train.py --dataset IIIT5K --data_path ./data/IIIT5K \
    --epochs 20 --exp_name aster_finetune \
    --checkpoint checkpoints/aster_pretrain_best.pth
```

### Inference

```bash
# Single image
python inference.py \
    --image path/to/image.jpg \
    --checkpoint checkpoints/aster_best.pth \
    --dataset Synth90K \
    --visualize

# Directory of images
python inference.py \
    --image_dir path/to/images/ \
    --checkpoint checkpoints/aster_best.pth \
    --dataset Synth90K \
    --visualize \
    --output ./results/
```

### Jupyter Notebook Tutorial

Open `ASTER_Tutorial.ipynb` for an interactive walkthrough of the architecture:

```bash
jupyter notebook ASTER_Tutorial.ipynb
```

The notebook covers:
- Step-by-step architecture exploration
- Visualization of each component
- Training process demonstration
- Inference examples
- Customization guide

## Training Pipeline

### 1. Pre-training on Synthetic Data

Synthetic datasets provide large-scale training data:

```bash
# Synth90K (recommended for initial training)
python train.py \
    --dataset Synth90K \
    --data_path ./data/Synth90K \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --exp_name aster_synth90k

# SynthText (alternative, larger dataset)
python train.py \
    --dataset SynthText \
    --data_path ./data/SynthText \
    --epochs 50 \
    --batch_size 32 \
    --exp_name aster_synthtext
```

### 2. Fine-tuning on Real Data

Fine-tune on real-world datasets for better accuracy:

```bash
# IIIT5K (good starting point)
python train.py \
    --dataset IIIT5K \
    --data_path ./data/IIIT5K \
    --epochs 20 \
    --lr 0.0001 \
    --checkpoint checkpoints/aster_synth90k_best.pth \
    --exp_name aster_iiit5k_finetune

# SVT (street view text)
python train.py \
    --dataset SVT \
    --data_path ./data/SVT \
    --epochs 30 \
    --checkpoint checkpoints/aster_iiit5k_best.pth \
    --exp_name aster_svt_finetune

# ICDAR datasets
python train.py \
    --dataset IC15 \
    --data_path ./data/IC15 \
    --epochs 30 \
    --checkpoint checkpoints/aster_synth90k_best.pth \
    --exp_name aster_ic15
```

### 3. Sequential Training Strategy

For best results, train sequentially:

1. **Stage 1**: Synth90K (50 epochs) - General text recognition
2. **Stage 2**: IIIT5K (20 epochs) - Real scene text
3. **Stage 3**: SVT/IC15 (20 epochs) - Challenging cases

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name (Synth90K, SynthText, IIIT5K, SVT, IC13, IC15, Synthetic) | Synthetic |
| `--data_path` | Path to dataset | ./data |
| `--batch_size` | Batch size | 32 |
| `--lr` | Learning rate | 0.001 |
| `--epochs` | Number of epochs | 100 |
| `--num_workers` | Data loading workers | 4 |
| `--checkpoint_dir` | Directory to save checkpoints | ./checkpoints |
| `--exp_name` | Experiment name | aster_ocr |
| `--log_dir` | TensorBoard log directory | ./logs |
| `--seed` | Random seed | 42 |
| `--no_cuda` | Disable CUDA | False |

## Configuration

The `config.py` file provides configuration classes for different datasets. Key parameters:

```python
class Config:
    # Model Architecture
    NUM_FIDUCIAL = 20          # TPS fiducial points
    IMG_HEIGHT = 32            # Input image height
    IMG_WIDTH = 100            # Input image width
    HIDDEN_SIZE = 256          # LSTM hidden size
    ATTENTION_DIM = 256        # Attention dimension
    EMBEDDING_DIM = 256        # Character embedding dimension
    MAX_SEQ_LENGTH = 25        # Maximum sequence length
    
    # Character Set
    CHARACTERS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ..."
    NUM_CLASSES = len(CHARACTERS) + 1  # +1 for special tokens
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    NUM_EPOCHS = 100
```

## Model Components

### 1. TPS Rectification

```python
from rectification import TPSRectification

rectifier = TPSRectification(
    num_fiducial=20,
    img_height=32,
    img_width=100,
    num_channels=3
)

rectified, ctrl_points = rectifier(distorted_image)
```

### 2. Feature Extraction

```python
from feature_extractor import ResNetFeatureExtractor

extractor = ResNetFeatureExtractor(
    num_channels=3,
    output_channels=512
)

features = extractor(rectified_image)
```

### 3. Sequence Encoding

```python
from bidirectional_lstm import SequenceEncoder

encoder = SequenceEncoder(
    input_channels=512,
    hidden_size=256,
    num_layers=2
)

sequence_features = encoder(features)
```

### 4. Attention Decoding

```python
from attention_decoder import AttentionDecoderV2

decoder = AttentionDecoderV2(
    num_classes=num_classes,
    encoder_dim=512,
    embedding_dim=256,
    hidden_dim=256,
    attention_dim=256
)

outputs, attention_weights = decoder(sequence_features)
```

### 5. Complete Model

```python
from model import ASTER
from config import get_config

config = get_config('Synth90K')
model = ASTER(config)

# Training
outputs, attention_weights, rectified, ctrl_points = model(
    images, targets=labels, teacher_forcing_ratio=0.5
)

# Inference
predictions, attention_weights, rectified, ctrl_points = model.predict(images)
```

## Evaluation

The model is evaluated using word-level accuracy:

```python
from utils import calculate_accuracy

accuracy = calculate_accuracy(predictions, targets)
print(f"Accuracy: {accuracy:.2f}%")
```

## Visualization

Visualize rectification and attention:

```python
import matplotlib.pyplot as plt

# Visualize rectification
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(rectified_image)
plt.title('Rectified')

# Visualize attention
attention_weights = [aw[0].cpu().numpy() for aw in attention_weights]
plt.bar(range(len(attention_weights[0])), attention_weights[0])
plt.title('Attention Distribution')
```

## Performance

Expected performance on various datasets (pre-trained on Synth90K):

| Dataset | Accuracy | Notes |
|---------|----------|-------|
| IIIT5K | ~95% | Regular text |
| SVT | ~88% | Street view text |
| IC13 | ~92% | ICDAR 2013 |
| IC15 | ~78% | ICDAR 2015 (challenging) |

## Tips for Training

1. **Start with synthetic data**: Synth90K provides good initialization
2. **Use gradient clipping**: Prevents exploding gradients (max_norm=5)
3. **Monitor smoothness loss**: Ensures TPS produces reasonable transformations
4. **Fine-tune learning rate**: Use 0.0001 for fine-tuning on real data
5. **Data augmentation**: Include ColorJitter, random crops
6. **Teacher forcing**: Start with 0.5, reduce to 0.3 after few epochs

## Common Issues

### 1. Out of Memory

```bash
# Reduce batch size
python train.py --batch_size 16
```

### 2. Slow Training

```bash
# Increase num_workers
python train.py --num_workers 8
```

### 3. Poor Rectification

- Check smoothness loss is being applied (lambda_smooth > 0)
- Verify fiducial points are within image bounds
- Visualize rectified images during training

### 4. Low Accuracy

- Train longer on synthetic data first
- Ensure learning rate is appropriate for fine-tuning
- Check data preprocessing (normalization, resizing)

## Citation

If you use this implementation, please cite the original ASTER paper:

```bibtex
@inproceedings{shi2018aster,
  title={ASTER: An attentional scene text recognizer with flexible rectification},
  author={Shi, Baoguang and Yang, Mingkun and Wang, Xinggang and Lyu, Pengyuan and Yao, Cong and Bai, Xiang},
  booktitle={IEEE transactions on pattern analysis and machine intelligence},
  year={2018}
}
```

## License

This implementation is provided for research and educational purposes.

## Acknowledgments

- Original ASTER paper authors
- PyTorch team
- Synthetic dataset creators (Synth90K, SynthText)

## Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Text Recognition! 📝**
