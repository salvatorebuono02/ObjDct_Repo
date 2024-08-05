import os
import torch
import config
import utils
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image


class YoloDataset(Dataset):
    def __init__(self, set_type, normalize=False, augment=False):
        assert set_type in {'train', 'test'}
        self.image_dir = os.path.join(config.DATA_PATH, 'images', set_type)
        self.label_dir = os.path.join(config.DATA_PATH, 'labels', set_type)
        self.image_paths = sorted([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)])
        self.label_paths = sorted([os.path.join(self.label_dir, f) for f in os.listdir(self.label_dir)])
        self.normalize = normalize
        self.augment = augment
        self.classes = utils.load_class_dict()

    def __getitem__(self, i):
        # Load image and labels
        image_path = self.image_paths[i]
        label_path = self.label_paths[i]
        image = Image.open(image_path).convert('L')
        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                labels.append([float(x) for x in line.strip().split()])

        # Apply transformations
        image = T.ToTensor()(image)
        original_data = T.Resize(config.IMAGE_SIZE)(image)
        if self.augment:
            x_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[0])
            y_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[1])
            scale = 1 + 0.2 * random.random()
            image = TF.affine(original_data, angle=0.0, scale=scale, translate=(x_shift, y_shift), shear=0.0)
            image = TF.adjust_hue(image, 0.2 * random.random() - 0.1)
            image = TF.adjust_saturation(image, 0.2 * random.random() + 0.9)
        else:
            image = original_data

        if self.normalize:
            image = TF.normalize(image, mean=[0.485], std=[0.229])

        grid_size_x = config.IMAGE_SIZE[0] / config.S
        grid_size_y = config.IMAGE_SIZE[1] / config.S

        # Process bounding boxes into the SxSx(5*B+C) ground truth tensor
        ground_truth = torch.zeros((config.S, config.S, 5 * config.B + config.C))
        for label in labels:
            class_idx, x_center, y_center, width, height = label
            class_idx = int(class_idx)
            if self.augment:
                half_width = config.IMAGE_SIZE[0] / 2
                half_height = config.IMAGE_SIZE[1] / 2
                x_center = utils.scale_bbox_coord(x_center, half_width, scale) + x_shift
                y_center = utils.scale_bbox_coord(y_center, half_height, scale) + y_shift
                width *= scale
                height *= scale

            col = int(x_center * config.S)
            row = int(y_center * config.S)

            if 0 <= col < config.S and 0 <= row < config.S:
                cell = (row, col)

                one_hot = torch.zeros(config.C)
                one_hot[class_idx] = 1.0
                ground_truth[row, col, :config.C] = one_hot

                bbox_truth = (
                    x_center * config.S - col,
                    y_center * config.S - row,
                    width * config.S,
                    height * config.S,
                    1.0
                )

                for bbox_index in range(config.B):
                    bbox_start = 5 * bbox_index + config.C
                    if ground_truth[row, col, bbox_start + 4] == 0:  # Empty slot (confidence == 0)
                        ground_truth[row, col, bbox_start:bbox_start + 5] = torch.tensor(bbox_truth)
                        break

        return image, ground_truth, original_data

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    # Display data
    obj_classes = utils.load_class_array()
    print(obj_classes)
    train_set = YoloDataset('train', normalize=False, augment=False)

    negative_labels = 0
    smallest = 0
    largest = 0
    for data, label, _ in train_set:
        print(label.shape)
        negative_labels += torch.sum(label < 0).item()
        smallest = min(smallest, torch.min(data).item())
        largest = max(largest, torch.max(data).item())
        utils.plot_boxes(data, label, obj_classes, max_overlap=float('inf'))
        break
    # print('num_negatives', negative_labels)
    # print('dist', smallest, largest)

