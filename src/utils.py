from typing import Tuple, List, Union

import numpy as np
import os

if __name__ == '__main__':
    pass


def rotate_image(coordinates: Union[List[float, float], Tuple[float, float]], theta: float) -> Tuple[float, float]:
    """
    Calculates the new coordinates after rotating image by theta radians

    Parameters
    ----------
    coordinates (Tuple): coordinates before the rotation
    theta (float): rotation angle in radians

    Returns
    ----------
    new_coordinates (Tuple): new coordinates after rotation
    """
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    new_coordinates = (coordinates[0] * cos_theta - coordinates[1] * sin_theta,
                       coordinates[0] * sin_theta + coordinates[1] * cos_theta)
    return new_coordinates


def offset_coordinates(coordinates: Tuple[float, float],
                       offset: Tuple[float, float]) -> Tuple[float, float]:
    """
    Offsets the coordinates

    Parameters
    ----------
    coordinates (Tuple): coordinates before the offset
    offset (Tuple): required offset

    Returns
    ----------
    new_coordinates (Tuple): new coordinates after offset
    """
    return coordinates[0] + offset[0], coordinates[1] + offset[1]


def compute_metric(data_true: np.ndarray, data_pred: np.ndarray, out_image_w: int = 10496,
                   out_image_h: int = 10496) -> np.ndarray:
    """
    Computes the custom competition metric

    Parameters
    ----------
    data_true (List): true labels
    data_pred (List): predicted labels
    out_image_w (int): background width
    out_image_h (int): background height

    Returns
    ----------
    metric (float): calculated metric
    """
    x_center_true = np.array((data_true[0] + data_true[2]) / 2).astype(int)
    y_center_true = np.array((data_true[1] + data_true[3]) / 2).astype(int)

    x_metr = x_center_true - np.array((data_pred[0] + data_pred[2]) / 2).astype(int)
    y_metr = y_center_true - np.array((data_pred[1] + data_pred[3]) / 2).astype(int)

    metric = 1 - (0.7 * 0.5 * (abs(x_metr) / out_image_h + abs(y_metr) / out_image_w) + 0.3 * min(
        abs(data_true[4] - data_pred[4]), abs(abs(data_true[4] - data_pred[4]) - 360)) / 360)
    return metric


class RunningAverage:
    """
    Class to save and update the running average
    """

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0

    def update(self, n: float) -> None:
        """
        Updates running average with new value

        Parameters
        ----------
        n (float): value to add to the running average
        """
        self.total += n
        self.count += 1

    def __call__(self) -> float:
        """
        Returns current running average

        Returns
        -------
        running_avg (float): Current running average
        """
        running_avg = self.total / (self.count + 1e-15)
        return running_avg


def get_file_paths(folder: str) -> List[str]:
    """
    Gets files list from the folder and every sub-folder

    Parameters
    ----------
    folder (str): root folder

    Returns
    ----------
    files_list (List): files list
    """
    path_list = os.listdir(folder)
    files_list = list()
    for entry in path_list:
        path = os.path.join(folder, entry)
        if os.path.isdir(path):
            files_list = files_list + get_file_paths(path)
        else:
            files_list.append(path)
    return files_list
