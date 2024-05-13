import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model.unet import UNet
from src.hyperparameters.hyperparams import Hyperparams
from src.utils import get_loaders, save_checkpoint, load_checkpoint, get_accuracy_and_dice_score, \
    save_predictions_as_images, get_validation_transform

hyperparams = Hyperparams()


def train_fn(
        loader, model: nn.Module, optimizer: torch.optim.Optimizer,
        loss_function: nn.Module, scaler: torch.GradScaler
):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=hyperparams.DEVICE)
        targets = targets.float().unsqueeze(1).to(device=hyperparams.DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_function(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())


def main():
    print('Using Device: {}'.format(hyperparams.DEVICE))
    train_transform = A.Compose([
        A.Resize(height=hyperparams.IMAGE_HEIGHT, width=hyperparams.IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    val_transform = get_validation_transform(hyperparams)

    model = UNet(in_channels=3, out_channels=1).to(device=hyperparams.DEVICE)
    loss_fn = nn.BCEWithLogitsLoss().to(device=hyperparams.DEVICE)  # cross entropy loss for multi-class output
    optimizer = optim.Adam(model.parameters(), lr=hyperparams.LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        hyperparams, train_transform, val_transform
    )

    if hyperparams.LOAD_MODEL:
        load_checkpoint("../checkpoints/checkpoint.pth.tar", model)
        get_accuracy_and_dice_score(val_loader, model, device=hyperparams.DEVICE)
        save_predictions_as_images(val_loader, model, device=hyperparams.DEVICE)
    if hyperparams.TRAIN_MODEL:
        scaler = torch.cuda.amp.GradScaler()
        best_acc = 0
        best_dice = 0
        for epoch in range(hyperparams.NUM_EPOCHS):
            train_fn(train_loader, model, optimizer, loss_fn, scaler)

            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            acc, dice = get_accuracy_and_dice_score(val_loader, model, device=hyperparams.DEVICE)
            save = False
            if acc > best_acc:
                best_acc = acc
                save = True
            if dice > best_dice:
                best_dice = dice
                save = True
            if save:
                save_checkpoint(checkpoint, f"../checkpoints/checkpoint_{epoch:03d}.pth.tar")

            save_predictions_as_images(
                val_loader, model, device=hyperparams.DEVICE
            )


if __name__ == '__main__':
    main()
