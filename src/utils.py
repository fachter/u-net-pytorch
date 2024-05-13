import torch
import torchvision
from src.data.carvana_dataset import CarvanaDataset
from torch.utils.data import DataLoader
from src.hyperparameters.hyperparams import Hyperparams


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
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


def check_acc(loader, model, device="cuda"):
    num_correct = 0
    num_pixel = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds >= 0.5).float()
            num_correct += (preds == y).sum()
            num_pixel += torch.numel(preds)

            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(
        f"{num_correct} / {num_pixel} correct pixels "
        f"with Acc: {(num_correct / num_pixel) * 100:.2f}"
    )
    print(f"Dice Score: {(dice_score / len(loader)):.2f}")
    model.train()


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

    model.train()
