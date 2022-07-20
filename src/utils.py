import numpy as np
import os


def rotate_image(coordinates, theta):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    new_coordinates = (coordinates[0] * cos_theta - coordinates[1] * sin_theta,
                       coordinates[0] * sin_theta + coordinates[1] * cos_theta)
    return new_coordinates


def offset_coordinates(coordinates, offset):
    return coordinates[0] + offset[0], coordinates[1] + offset[1]


def compute_metric(data_true, data_pred, out_image_w=10496, out_image_h=10496):
    x_center_true = np.array((data_true[0] + data_true[2]) / 2).astype(int)
    y_center_true = np.array((data_true[1] + data_true[3]) / 2).astype(int)

    x_metr = x_center_true - np.array((data_pred[0] + data_pred[2]) / 2).astype(int)
    y_metr = y_center_true - np.array((data_pred[1] + data_pred[3]) / 2).astype(int)

    metr = 1 - (0.7 * 0.5 * (abs(x_metr) / out_image_h + abs(y_metr) / out_image_w) + 0.3 * min(
        abs(data_true[4] - data_pred[4]), abs(abs(data_true[4] - data_pred[4]) - 360)) / 360)
    return metr


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


def get_file_paths(folder):
    path_list = os.listdir(folder)
    files_list = list()
    for entry in path_list:
        path = os.path.join(folder, entry)
        if os.path.isdir(path):
            files_list = files_list + get_file_paths(path)
        else:
            files_list.append(path)
    return files_list
