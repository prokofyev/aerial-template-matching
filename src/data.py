from typing import List, Optional, Tuple, Union, Any

import os
import sys
from itertools import product
from itertools import cycle
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2
from PIL import Image
from tqdm import tqdm

from .utils import rotate_image, offset_coordinates

if __name__ == '__main__':
    pass


class ImageDataset(Dataset):
    """
    Class to create pytorch dataset for the aerial template matching task
    """

    def __init__(self,
                 data_df: pd.DataFrame,
                 background_image_paths: List[str] = None,
                 add_data: bool = True,
                 transform: Optional[A.core.composition.Compose] = None) -> None:
        """
        Create the pytorch dataset for the aerial template matching task

        Parameters
        ----------
        data_df (pd.DataFrame): pre-cropped images and labels
        background_image_paths (List[str]): paths of backgrounds to create new data from
        add_data (bool): whether to augment with data generated from provided backgrounds
        transform (albumentations.Compose): albumentations transforms
        """
        self.data_df = data_df
        self.transform = transform
        self._add_data = add_data
        self._data_len = len(self.data_df)
        self._points = np.array([[0, 1024],
                                 [0, 0],
                                 [1024, 0],
                                 [1024, 1024]], dtype="float32")
        self._coords_grid = [i * 422 + 1024 for i in range(21)]
        self._coords_grid = list(product(self._coords_grid, self._coords_grid))

        if background_image_paths:
            self.background_image_paths = background_image_paths
            self.background_images = [cv2.imread(path) for path in self.background_image_paths]
            self._background_images_len = len(self.background_image_paths)
            self._cycle = cycle(range(self._background_images_len))

    def __len__(self) -> int:
        """
        Returns
        ----------
        length (int): dataset length
        """
        if self._add_data:
            return self._data_len * 2
        else:
            return self._data_len

    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        """
        Gets images and labels by id, generates new images and labels on-the-fly if needed

        Parameters
        ----------
        idx (int): image id

        Returns
        ----------
        result (Tuple): image and its labels
        """
        if idx < self._data_len:
            template_image_path = self.data_df.iloc[idx]['path']
            template_image = cv2.imread(template_image_path)
            template_labels = [self.data_df.iloc[idx]['left_top_x'] / 10496,
                               self.data_df.iloc[idx]['left_top_y'] / 10496,
                               self.data_df.iloc[idx]['right_bottom_x'] / 10496,
                               self.data_df.iloc[idx]['right_bottom_y'] / 10496,
                               self.data_df.iloc[idx]['angle'] / 360]
        else:
            idx = np.random.randint(low=0, high=441, dtype=int)
            left_top_x = self._coords_grid[idx][0] + np.random.randint(low=-100, high=100, dtype=int)
            left_top_y = self._coords_grid[idx][1] + np.random.randint(low=-100, high=100, dtype=int)

            offset = (left_top_x, left_top_y)
            angle = np.random.choice([np.random.randint(low=0, high=45, dtype=int),
                                      np.random.randint(low=315, high=360, dtype=int)])
            theta = np.radians(angle)
            rotated_mask = np.array([offset_coordinates(rotate_image(coordinates, theta), offset)
                                     for coordinates in self._points], dtype='float32')
            transformation_matrix = cv2.getPerspectiveTransform(rotated_mask, self._points)
            template_image = cv2.warpPerspective(self.background_images[next(self._cycle)],
                                                 transformation_matrix,
                                                 (1024, 1024))

            right_bottom_x = int(rotated_mask[3][0])
            right_bottom_y = int(rotated_mask[3][1])

            template_labels = [left_top_x / 10496,
                               left_top_y / 10496,
                               right_bottom_x / 10496,
                               right_bottom_y / 10496,
                               angle / 360]

        template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)
        template_image = template_image.astype('float32')
        template_image = template_image / 255.

        if self.transform:
            template_image = self.transform(image=template_image)['image']

        return template_image, torch.tensor(template_labels).float()


def create_and_save_backgrounds(ids: List[List[int]],
                                data_df: pd.DataFrame,
                                background_path: str,
                                save_path: str) -> None:
    """
    Generates new backgrounds from training set and saves them

    Parameters
    ----------
    ids (List): list of lists of image ids for each new background
    data_df (pd.DataFrame): pre-cropped images and labels
    background_path (str): background to paste images on
    save_path (str): where to save the files
    """
    with tqdm(total=sum([len(x) for x in ids]), leave=True, file=sys.stdout) as t:
        for i, fold in enumerate(ids):
            background = Image.open(background_path)
            for idx in fold:
                template_image = cv2.imread(data_df.iloc[idx]['path'])
                template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGBA)
                points = np.array([[0, 1024],
                                   [0, 0],
                                   [1024, 0],
                                   [1024, 1024]], dtype="float32")

                rotated_mask = np.array([data_df.iloc[idx][['left_bottom_x', 'left_bottom_y']],
                                         data_df.iloc[idx][['left_top_x', 'left_top_y']],
                                         data_df.iloc[idx][['right_top_x', 'right_top_y']],
                                         data_df.iloc[idx][['right_bottom_x', 'right_bottom_y']]], dtype="float32")
                transformation_matrix = cv2.getPerspectiveTransform(points, rotated_mask)
                overlay = cv2.warpPerspective(template_image, transformation_matrix, (10496, 10496))

                overlay = Image.fromarray(overlay)
                background.paste(overlay, mask=overlay)
                t.update()

            background.save(f'{save_path}background-{i}.tiff')


def read_json_from_dir(json_dir: str, image_dir: str) -> pd.DataFrame:
    """
    Creates pre-cropped images and labels dataframe from json files

    Parameters
    ----------
    json_dir (str): folder with json files
    image_dir (str): folder with image files
    save_path (str): where to save the files

    Returns
    ----------
    data_df (pd.DataFrame): images and labels dataframe
    """
    data_df = pd.DataFrame({'path': [],
                            'left_top_x': [],
                            'left_top_y': [],
                            'right_top_x': [],
                            'right_top_y': [],
                            'left_bottom_x': [],
                            'left_bottom_y': [],
                            'right_bottom_x': [],
                            'right_bottom_y': [],
                            'angle': []})

    for _, _, files in os.walk(json_dir):
        for file in files:
            if file.endswith('.json'):
                data = json.load(open(json_dir + file))
                new_row = {'path': f'{image_dir}' + file.split('.')[0] + '.png',
                           'left_top_x': data['left_top'][0],
                           'left_top_y': data['left_top'][1],
                           'right_top_x': data['right_top'][0],
                           'right_top_y': data['right_top'][1],
                           'left_bottom_x': data['left_bottom'][0],
                           'left_bottom_y': data['left_bottom'][1],
                           'right_bottom_x': data['right_bottom'][0],
                           'right_bottom_y': data['right_bottom'][1],
                           'angle': data['angle']}
                data_df = data_df.append(new_row, ignore_index=True)

    return data_df


def get_transforms(image_width: int = 512,
                   image_height: int = 512,
                   add_augmentations: bool = False) -> A.core.composition.Compose:
    """
    Creates albumentations transforms

    Parameters
    ----------
    image_width (int): desired image width
    image_height (int): desired image height
    add_augmentations (bool): if true, returns transforms with augmentations

    Returns
    -------
    loaders (A.Compose): albumentations transforms
    """
    transforms_augment_list = [
        A.OneOf([A.CoarseDropout(max_holes=10, max_height=256, max_width=256, min_holes=4, min_height=128,
                                 min_width=128, fill_value=1, mask_fill_value=0, p=0.5),
                 A.GridDropout(ratio=0.5, random_offset=True, fill_value=1, p=0.5)], p=0.5),
        A.PixelDropout(dropout_prob=0.05, per_channel=False, drop_value=1, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.25, 0.25), p=0.5),
        A.MultiplicativeNoise(multiplier=(0.75, 1.25), per_channel=True, p=0.5),
    ]
    transforms_resize_list = [
        A.LongestMaxSize(image_width),
        A.PadIfNeeded(image_height, image_width, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
        A.pytorch.ToTensorV2()
    ]

    if add_augmentations:
        transforms = A.Compose([*transforms_augment_list, *transforms_resize_list])
    else:
        transforms = A.Compose(transforms_resize_list)

    return transforms
