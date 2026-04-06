"""
ASTER: Attentional Scene Text Recognition
Dataset classes for various OCR datasets
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path


class BaseOCRDataset(Dataset):
    """Base class for OCR datasets"""

    def __init__(
        self,
        data_path,
        charset,
        transform=None,
        img_height=32,
        img_width=100,
        max_length=25,
    ):
        self.data_path = Path(data_path)
        self.charset = charset
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width
        self.max_length = max_length

        # Create character to index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(charset)} #(char : index) dictionary
        self.idx_to_char = {idx: char for idx, char in enumerate(charset)} #(index : char) dictionary

        # Special tokens
        self.sos_token = 0
        self.eos_token = 1
        self.pad_token = 2

        self.samples = []

    def encode_text(self, text):
        """Encode text to indices"""
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                # Unknown character - use a special index or skip
                indices.append(len(self.charset))  # Unknown token
        return indices

    def decode_indices(self, indices):
        """Decode indices to text"""
        text = []
        for idx in indices:
            if idx == self.eos_token:
                break
            if idx < len(self.charset):
                text.append(self.idx_to_char[idx])
        return "".join(text)

    def pad_sequence(self, indices):
        """Pad sequence to max_length"""
        if len(indices) > self.max_length - 2:  # -2 for SOS and EOS
            indices = indices[: self.max_length - 2]

        # Add SOS and EOS
        indices = [self.sos_token] + indices + [self.eos_token]

        # Pad to max_length
        while len(indices) < self.max_length:
            indices.append(self.pad_token)

        return indices

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")


class Synth90KDataset(BaseOCRDataset):
    """
    Synth90K (Synthetic Word Dataset)

    Dataset structure:
    - annotation.txt: each line contains "path label"
    - images/ directory with image files
    """

    def __init__(
        self,
        data_path,
        charset,
        split="train",
        transform=None,
        img_height=32,
        img_width=100,
        max_length=25,
    ):
        super().__init__(
            data_path, charset, transform, img_height, img_width, max_length
        )

        self.split = split
        self.annotation_file = self.data_path / f"{split}_annotation.txt"

        # Load annotations
        self._load_annotations()

    def _load_annotations(self):
        """Load annotations from file"""
        if not self.annotation_file.exists():
            print(f"Warning: {self.annotation_file} not found. Creating dummy dataset.")
            self.samples = []
            return

        with open(self.annotation_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    img_path = self.data_path / "images" / parts[0]
                    label = " ".join(parts[1:])

                    if img_path.exists():
                        self.samples.append({"img_path": str(img_path), "label": label})

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["img_path"]).convert("RGB")

        # Resize to target size
        image = image.resize((self.img_width, self.img_height), Image.LANCZOS)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Encode label
        label_indices = self.encode_text(sample["label"])
        label_indices = self.pad_sequence(label_indices)

        return {
            "image": image,
            "label": torch.tensor(label_indices, dtype=torch.long),
            "text": sample["label"],
            "img_path": sample["img_path"],
        }


class SynthTextDataset(BaseOCRDataset):
    """
    SynthText Dataset

    Dataset structure:
    - gt.mat: MATLAB file with annotations
    - images/ directory with image files
    """

    def __init__(
        self,
        data_path,
        charset,
        split="train",
        transform=None,
        img_height=32,
        img_width=100,
        max_length=25,
    ):
        super().__init__(
            data_path, charset, transform, img_height, img_width, max_length
        )

        self.split = split
        self._load_annotations()

    def _load_annotations(self):
        """Load annotations from gt.mat file"""
        import scipy.io as sio

        gt_file = self.data_path / "gt.mat"

        if not gt_file.exists():
            print(f"Warning: {gt_file} not found. Creating dummy dataset.")
            self.samples = []
            return

        try:
            mat = sio.loadmat(gt_file)

            # Extract image paths and labels
            # SynthText format varies, this is a simplified version
            # In practice, you'd need to parse the specific format

            # For now, create dummy samples
            # In real implementation, parse mat['imnames'], mat['txt'], etc.
            self.samples = []

        except Exception as e:
            print(f"Error loading SynthText annotations: {e}")
            self.samples = []


class IIIT5KDataset(BaseOCRDataset):
    """
    IIIT5K Dataset

    Dataset structure:
    - train.json / test.json: annotations
    - images/ directory with image files
    """

    def __init__(
        self,
        data_path,
        charset,
        split="train",
        transform=None,
        img_height=32,
        img_width=100,
        max_length=25,
    ):
        super().__init__(
            data_path, charset, transform, img_height, img_width, max_length
        )

        self.split = split
        self.annotation_file = self.data_path / f"{split}.json"

        self._load_annotations()

    def _load_annotations(self):
        """Load annotations from JSON file"""
        if not self.annotation_file.exists():
            print(f"Warning: {self.annotation_file} not found. Creating dummy dataset.")
            self.samples = []
            return

        with open(self.annotation_file, "r") as f:
            data = json.load(f)

        for item in data:
            img_path = self.data_path / item["img_path"]
            if img_path.exists():
                self.samples.append({"img_path": str(img_path), "label": item["label"]})

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["img_path"]).convert("RGB")

        # Resize
        image = image.resize((self.img_width, self.img_height), Image.LANCZOS)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Encode label
        label_indices = self.encode_text(sample["label"])
        label_indices = self.pad_sequence(label_indices)

        return {
            "image": image,
            "label": torch.tensor(label_indices, dtype=torch.long),
            "text": sample["label"],
            "img_path": sample["img_path"],
        }


class SVTDataset(BaseOCRDataset):
    """Street View Text (SVT) Dataset"""

    def __init__(
        self,
        data_path,
        charset,
        split="train",
        transform=None,
        img_height=32,
        img_width=100,
        max_length=25,
    ):
        super().__init__(
            data_path, charset, transform, img_height, img_width, max_length
        )

        self.split = split
        self._load_annotations()

    def _load_annotations(self):
        """Load annotations from XML file"""
        import xml.etree.ElementTree as ET

        xml_file = self.data_path / f"{self.split}.xml"

        if not xml_file.exists():
            print(f"Warning: {xml_file} not found. Creating dummy dataset.")
            self.samples = []
            return

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for image in root.findall("image"):
                img_name = image.get("file")
                img_path = self.data_path / img_name

                # Get label from tags
                label = image.find("tag").get("label")

                if img_path.exists():
                    self.samples.append({"img_path": str(img_path), "label": label})
        except Exception as e:
            print(f"Error loading SVT annotations: {e}")
            self.samples = []


class ICDARDataset(BaseOCRDataset):
    """
    ICDAR Dataset (IC13, IC15)

    Dataset structure:
    - gt.txt: ground truth annotations
    - images/: image files
    """

    def __init__(
        self,
        data_path,
        charset,
        dataset_name="IC13",
        split="train",
        transform=None,
        img_height=32,
        img_width=100,
        max_length=25,
    ):
        super().__init__(
            data_path, charset, transform, img_height, img_width, max_length
        )

        self.dataset_name = dataset_name
        self.split = split

        self._load_annotations()

    def _load_annotations(self):
        """Load annotations from gt.txt file"""
        gt_file = self.data_path / f"{self.split}_gt.txt"

        if not gt_file.exists():
            print(f"Warning: {gt_file} not found. Creating dummy dataset.")
            self.samples = []
            return

        with open(gt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # ICDAR format: "img_name label"
                parts = line.split(",")
                if len(parts) >= 2:
                    img_name = parts[0]
                    label = parts[1].strip('"')

                    img_path = self.data_path / "images" / img_name

                    if img_path.exists():
                        self.samples.append({"img_path": str(img_path), "label": label})


class SyntheticTextGenerator(Dataset):
    """
    Synthetic Text Generator for quick testing
    Generates random text images with distortions
    """

    def __init__(
        self,
        num_samples=1000,
        charset=None,
        img_height=32,
        img_width=100,
        max_length=10,
        transform=None,
    ):
        self.num_samples = num_samples
        self.charset = (
            charset or "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        )
        self.img_height = img_height
        self.img_width = img_width
        self.max_length = max_length
        self.transform = transform

        # Create character mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random text
        text_length = random.randint(3, self.max_length)
        text = "".join(random.choices(self.charset, k=text_length))

        # Create image with text (simplified - in practice use PIL ImageFont)
        image = Image.new(
            "RGB", (self.img_width, self.img_height), color=(255, 255, 255)
        )

        # Apply random distortions (simplified)
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Encode text
        label_indices = [self.char_to_idx.get(c, len(self.charset)) for c in text]
        label_indices = [0] + label_indices + [1]  # Add SOS and EOS
        while len(label_indices) < self.max_length + 2:
            label_indices.append(2)  # PAD
        label_indices = label_indices[: self.max_length + 2]

        return {
            "image": image,
            "label": torch.tensor(label_indices, dtype=torch.long),
            "text": text,
            "img_path": f"synthetic_{idx}.jpg",
        }


def get_dataset(
    dataset_name,
    data_path,
    charset,
    split="train",
    transform=None,
    img_height=32,
    img_width=100,
    max_length=25,
):
    """
    Factory function to get dataset by name

    Args:
        dataset_name: Name of dataset ('Synth90K', 'SynthText', 'IIIT5K', 'SVT', 'IC13', 'IC15', 'Synthetic')
        data_path: Path to dataset
        charset: Character set
        split: Dataset split ('train', 'test', 'val')
        transform: Image transforms
        img_height: Image height
        img_width: Image width
        max_length: Maximum text length

    Returns:
        dataset: Dataset instance
    """
    if dataset_name == "Synth90K":
        return Synth90KDataset(
            data_path, charset, split, transform, img_height, img_width, max_length
        )
    elif dataset_name == "SynthText":
        return SynthTextDataset(
            data_path, charset, split, transform, img_height, img_width, max_length
        )
    elif dataset_name == "IIIT5K":
        return IIIT5KDataset(
            data_path, charset, split, transform, img_height, img_width, max_length
        )
    elif dataset_name == "SVT":
        return SVTDataset(
            data_path, charset, split, transform, img_height, img_width, max_length
        )
    elif dataset_name == "IC13" or dataset_name == "IC15":
        return ICDARDataset(
            data_path,
            charset,
            dataset_name,
            split,
            transform,
            img_height,
            img_width,
            max_length,
        )
    elif dataset_name == "Synthetic":
        return SyntheticTextGenerator(
            charset=charset,
            img_height=img_height,
            img_width=img_width,
            max_length=max_length,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_transforms(is_training=True, img_height=32, img_width=100):
    """
    Get data transforms

    Args:
        is_training: Whether for training or testing
        img_height: Target image height
        img_width: Target image width

    Returns:
        transforms: Composed transforms
    """
    if is_training:
        return transforms.Compose(
            [
                transforms.Resize((img_height, img_width)),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((img_height, img_width)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


if __name__ == "__main__":
    print("Testing Datasets...")

    # Test synthetic dataset
    print("\n1. Testing Synthetic Dataset:")
    charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

    synthetic_dataset = SyntheticTextGenerator(
        num_samples=100, charset=charset, img_height=32, img_width=100, max_length=10
    )

    print(f"Dataset size: {len(synthetic_dataset)}")

    sample = synthetic_dataset[0]
    print(f"Sample image shape: {sample['image'].shape}")
    print(f"Sample label shape: {sample['label'].shape}")
    print(f"Sample text: {sample['text']}")

    # Test DataLoader
    print("\n2. Testing DataLoader:")
    dataloader = DataLoader(
        synthetic_dataset, batch_size=4, shuffle=True, num_workers=0
    )

    batch = next(iter(dataloader))
    print(f"Batch images shape: {batch['image'].shape}")
    print(f"Batch labels shape: {batch['label'].shape}")
    print(f"Batch texts: {batch['text']}")

    print("\nDataset tests passed!")
