import torch
import config
import os
import utils
from tqdm import tqdm
from data import YoloDataset
from models import *
from torch.utils.data import DataLoader


MODEL_DIR = 'models/YOLOv1/03_08_2024/09_38_54'


def plot_test_images():
    classes = utils.load_class_array()

    dataset = YoloDataset('test', normalize=False, augment=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = TinyYOLOv1()
    model.eval()
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'weights', 'final')))

    count = 0
    with torch.no_grad():
        for image, labels, original in tqdm(loader):
            predictions = model.forward(image)
            predictions = predictions.reshape(-1, config.S, config.S, config.B*5 + config.C)
            for i in range(image.size(dim=0)):
                utils.plot_boxes(
                    original[i, :, :, :],
                    predictions[i, :, :, :],
                    classes,
                    file=os.path.join('results', f'{count}')
                )
                # utils.plot_boxes(
                #     original[i, :, :, :],
                #     labels[i, :, :, :],
                #     classes,
                #     color='green'
                # )
                count += 1


if __name__ == '__main__':
    plot_test_images()