"""
ASTER: Attentional Scene Text Recognition
Configuration file
"""


class Config:
    """Configuration class for ASTER OCR model"""

    # Model Architecture
    NUM_FIDUCIAL = 20  # Number of fiducial points for TPS
    IMG_HEIGHT = 32  # Input image height
    IMG_WIDTH = 100  # Input image width
    NUM_CHANNELS = 3  # Number of input channels (RGB)

    # CNN Feature Extractor
    CNN = "ResNet"  # ResNet-18 as feature extractor
    HIDDEN_SIZE = 256  # Hidden size for LSTM

    # Attention Mechanism
    ATTENTION_DIM = 256  # Attention dimension

    # Decoder
    EMBEDDING_DIM = 256  # Character embedding dimension
    MAX_SEQ_LENGTH = 25  # Maximum sequence length

    # Training
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    NUM_EPOCHS = 100

    # Dataset
    # Supported datasets: ['Synth90K', 'SynthText', 'IIIT5K', 'SVT', 'IC13', 'IC15']
    DATASET_NAME = "Synth90K"  # Default training dataset
    DATA_PATH = "./data"

    # Character Set
    # Full alphanumeric + punctuation (ASCII 32-126)
    CHARACTERS = """0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ """
    NUM_CLASSES = len(CHARACTERS) + 1  # +1 for CTC blank/unknown

    # Special Tokens
    SOS_TOKEN = 0  # Start of sequence
    EOS_TOKEN = 1  # End of sequence
    PAD_TOKEN = 2  # Padding

    # Device
    CUDA = True

    # Checkpointing
    SAVE_FREQ = 1  # Save every epoch
    CHECKPOINT_DIR = "./checkpoints"

    # Evaluation
    EVAL_FREQ = 1  # Evaluate every epoch

    # Visualization
    VISUALIZE = False
    VIS_DIR = "./visualizations"


class Synth90KConfig(Config):
    """Configuration for Synth90K dataset"""

    DATASET_NAME = "Synth90K"
    NUM_IMAGES = 7236848  # Total number of images


class SynthTextConfig(Config):
    """Configuration for SynthText dataset"""

    DATASET_NAME = "SynthText"
    NUM_IMAGES = 800000  # ~8 million synthetic text images


class IIIT5KConfig(Config):
    """Configuration for IIIT5K dataset (evaluation)"""

    DATASET_NAME = "IIIT5K"
    NUM_TRAIN = 2000
    NUM_TEST = 3000


class SVTConfig(Config):
    """Configuration for Street View Text dataset (evaluation)"""

    DATASET_NAME = "SVT"
    NUM_TRAIN = 100
    NUM_TEST = 249


class IC13Config(Config):
    """Configuration for ICDAR 2013 dataset (evaluation)"""

    DATASET_NAME = "IC13"
    NUM_TRAIN = 848
    NUM_TEST = 1095


class IC15Config(Config):
    """Configuration for ICDAR 2015 dataset (evaluation)"""

    DATASET_NAME = "IC15"
    NUM_TRAIN = 4468
    NUM_TEST = 2077


# Map dataset names to configurations
DATASET_CONFIGS = {
    "Synth90K": Synth90KConfig,
    "SynthText": SynthTextConfig,
    "IIIT5K": IIIT5KConfig,
    "SVT": SVTConfig,
    "IC13": IC13Config,
    "IC15": IC15Config,
}


def get_config(dataset_name="Synth90K"):
    """Get configuration for specific dataset"""
    if dataset_name in DATASET_CONFIGS:
        return DATASET_CONFIGS[dataset_name]()
    return Config()
