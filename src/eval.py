from model.unet import UNet
from src.hyperparameters.hyperparams import Hyperparams
from src.utils import load_checkpoint, get_test_loader, get_validation_transform, predict_and_save_mask

hyperparams = Hyperparams()


def main():
    test_transform = get_validation_transform(hyperparams)

    model = UNet(in_channels=3, out_channels=1).to(device=hyperparams.DEVICE)
    test_loader = get_test_loader(hyperparams, test_transform)
    load_checkpoint("../checkpoints/checkpoint.pth.tar", model)
    for idx, (image, _) in enumerate(test_loader):
        image = image.to(device=hyperparams.DEVICE)
        predict_and_save_mask("../saved_test_images/", idx, model, image)


if __name__ == '__main__':
    main()
