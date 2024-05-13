import torch


class Hyperparams:
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    BATCH_SIZE = 16
    NUM_EPOCHS = 3
    NUM_WORKERS = 2
    IMAGE_HEIGHT = 160
    IMAGE_WIDTH = 240
    PIN_MEMORY = True
    LOAD_MODEL = True
    TRAIN_IMG_DIR = "data/train_images/"
    TRAIN_MASK_DIR = "data/train_masks/"
    VAL_IMG_DIR = "data/val_images/"
    VAL_MASK_DIR = "data/val_masks/"
