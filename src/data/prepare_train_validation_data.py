import random
from pathlib import Path


def move_n_files_to_validation_set(train_folder: Path, validation_folder: Path, n: int):
    images = list(train_folder.glob("*.jpg"))
    masks_folder = (train_folder.parent / (train_folder.stem + "_masks"))
    validation_mask_folder = (validation_folder.parent / (validation_folder.stem + "_masks"))
    validation_folder.mkdir(exist_ok=True)
    validation_mask_folder.mkdir(exist_ok=True)

    unique_image_ids = list({image.stem.split("_")[0] for image in images})

    random_ids = random.sample(unique_image_ids, n)
    for random_id in random_ids:
        img_files_to_move = list(train_folder.glob(f"{random_id}*.jpg"))
        mask_files_to_move = list(masks_folder.glob(f"{random_id}*.gif"))
        for img_file, mask_file in zip(img_files_to_move, mask_files_to_move):
            img_file.rename(validation_folder / img_file.name)
            mask_file.rename(validation_mask_folder / mask_file.name)


if __name__ == '__main__':
    data_folder = Path(__file__).parent.parent.parent / 'data'
    move_n_files_to_validation_set(data_folder / "train", data_folder / "validation", 4)
