import os
import unittest
from pathlib import Path

from src.data.prepare_train_validation_data import move_n_files_to_validation_set


def _folder_has_files(path):
    return path.exists() and len(list(path.glob("*"))) > 0


file_path = os.path.dirname(os.path.abspath(__file__))


class PrepareDataTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.train_folder = Path(file_path + '/../data/train')
        self.train_masks_folder = Path(file_path + '/../data/train_masks')
        self.validation_folder = Path(file_path + '/../data/validation')
        self.validation_masks_folder = Path(file_path + '/../data/validation_masks')

    def tearDown(self):
        for file in self.validation_folder.iterdir():
            file.rename(self.train_folder / file.name)
        for file in self.validation_masks_folder.iterdir():
            file.rename(self.train_masks_folder / file.name)

        self.assertTrue(len(list(self.validation_folder.glob('*.jpg'))) == 0)
        self.assertTrue(len(list(self.validation_masks_folder.glob('*.gif'))) == 0)

        self.validation_folder.rmdir()
        self.validation_masks_folder.rmdir()

    def test_with_one_car(self):
        image_folder_has_files = _folder_has_files(self.validation_folder)
        mask_folder_has_files = _folder_has_files(self.validation_masks_folder)

        move_n_files_to_validation_set(self.train_folder, self.validation_folder, 1)

        self.assertEqual(image_folder_has_files, False)
        self.assertEqual(mask_folder_has_files, False)
        self.assertEqual(_folder_has_files(self.validation_folder), True)
        self.assertEqual(_folder_has_files(self.validation_masks_folder), True)
        self.assertEqual(len(list(self.validation_folder.glob('*.jpg'))), 16)
        self.assertEqual(len(list(self.validation_masks_folder.glob('*.gif'))), 16)

    def test_with_three_cars(self):
        n = 3
        image_folder_has_files = _folder_has_files(self.validation_folder)
        mask_folder_has_files = _folder_has_files(self.validation_masks_folder)

        move_n_files_to_validation_set(self.train_folder, self.validation_folder, n)

        self.assertEqual(image_folder_has_files, False)
        self.assertEqual(mask_folder_has_files, False)
        self.assertEqual(_folder_has_files(self.validation_folder), True)
        self.assertEqual(_folder_has_files(self.validation_masks_folder), True)
        self.assertEqual(len(list(self.validation_folder.glob('*.jpg'))), 16 * n)
        self.assertEqual(len(list(self.validation_masks_folder.glob('*.gif'))), 16 * n)

