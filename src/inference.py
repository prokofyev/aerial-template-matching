import json
import torch
import numpy as np
import pandas as pd
import cv2

from .utils import offset_coordinates, rotate_image


def predict_and_save(model: torch.nn.Module, transform, data_df: pd.DataFrame, path: str, device: torch.device):
    model.eval()
    for index, row in data_df.iterrows():
        with torch.inference_mode():
            test_image = cv2.imread(row['path'])
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            test_image = test_image.astype('float32')
            test_image = test_image / 255.
            test_image = transform(image=test_image)['image']
            test_image = test_image.to(device)
            model_pred = model(test_image.unsqueeze(0))

            left_top_x_pred = model_pred.squeeze(0)[0].cpu() * 10496
            left_top_y_pred = model_pred.squeeze(0)[1].cpu() * 10496
            right_bot_x_pred = model_pred.squeeze(0)[2].cpu() * 10496
            right_bot_y_pred = model_pred.squeeze(0)[3].cpu() * 10496
            angle_pred = model_pred.squeeze(0)[4].cpu() * 360

        offset = (left_top_x_pred, left_top_y_pred)
        theta = np.radians(angle_pred)
        left_bot_x_pred, left_bot_y_pred = offset_coordinates(rotate_image([0, 1024], theta), offset)
        right_top_x_pred, right_top_y_pred = offset_coordinates(rotate_image([1024, 0], theta), offset)

        left_top_x_pred, left_top_y_pred = np.round(left_top_x_pred.item(), 4), np.round(left_top_y_pred.item(), 4)
        right_top_x_pred, right_top_y_pred = np.round(right_top_x_pred.item(), 4), np.round(right_top_y_pred.item(), 4)
        left_bot_x_pred, left_bot_y_pred = np.round(left_bot_x_pred.item(), 4), np.round(left_bot_y_pred.item(), 4)
        right_bot_x_pred, right_bot_y_pred = np.round(right_bot_x_pred.item(), 4), np.round(right_bot_y_pred.item(), 4)
        angle_pred = np.round(angle_pred.item(), 4)

        prediction_json = {
            'left_top': [left_top_x_pred, left_top_y_pred],
            'right_top': [right_top_x_pred, right_top_y_pred],
            'left_bottom': [left_bot_x_pred, left_bot_y_pred],
            'right_bottom': [right_bot_x_pred, right_bot_y_pred],
            'angle': angle_pred
        }

        with open(f'{path}{row["id"]}.json', 'w') as f:
            json.dump(prediction_json, f)
