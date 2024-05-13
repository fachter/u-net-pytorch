import torch
import torchvision
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F


from src.data.carvana_dataset import CarvanaDataset
from torch.utils.data import DataLoader
from src.hyperparameters.hyperparams import Hyperparams


def save_checkpoint(state, filename='../checkpoints/checkpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint_filename, model):
    print(f"=> Loading checkpoint <{checkpoint_filename}>")
    checkpoint = torch.load(checkpoint_filename)
    model.load_state_dict(checkpoint['state_dict'])


def get_loaders(hyperparameters: Hyperparams, train_transform, validation_transform):
    train_dataset = CarvanaDataset(
        image_dir=hyperparameters.TRAIN_IMG_DIR,
        mask_dir=hyperparameters.TRAIN_MASK_DIR,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters.BATCH_SIZE,
        num_workers=hyperparameters.NUM_WORKERS,
        pin_memory=hyperparameters.PIN_MEMORY,
        shuffle=True
    )

    validation_dataset = CarvanaDataset(
        image_dir=hyperparameters.VAL_IMG_DIR,
        mask_dir=hyperparameters.VAL_MASK_DIR,
        transform=validation_transform
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=hyperparameters.BATCH_SIZE,
        num_workers=hyperparameters.NUM_WORKERS,
        pin_memory=hyperparameters.PIN_MEMORY,
        shuffle=False
    )

    return train_loader, validation_loader


def get_accuracy_and_dice_score(loader, model, device):
    num_correct = 0
    num_pixel = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds >= 0.5).float()
            num_correct += (preds == y).sum()
            num_pixel += torch.numel(preds)

            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    acc = (num_correct / num_pixel) * 100
    final_dice = (dice_score / len(loader))
    print(
        f"{num_correct} / {num_pixel} correct pixels "
        f"with Acc: {acc :.2f}"
    )
    print(f"Dice Score: {final_dice :.2f}")
    model.train()
    return acc, final_dice


def save_predictions_as_images(
        loader, model, folder="../saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds >= 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/{idx:03d}_prediction.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx:03d}_ground_truth.png")
        for input_image, mask_prediction in zip(x, preds):
            image_with_mask = (
                torchvision.utils.draw_segmentation_masks(
                    input_image, mask_prediction.bool(),
                    alpha=0.4, colors=(0, 220, 255))
            )
            torchvision.utils.save_image(image_with_mask, f"{folder}/{idx:03d}_masked.png")

    model.train()
