import os
import numpy as np
from PIL import Image
import torch as t
from skimage import transform as sktsf
import random
from torchvision import transforms as tvtsf

# Utility functions

def read_image(file_path, dtype=np.float32, color=False):
    """Utility function to load an image.
    
    Args:
        file_path (str): Path to the image file.
        dtype (np.dtype): Data type of the loaded image.
        color (bool): Whether to load the image in color. Default is False (grayscale).

    Returns:
        numpy.ndarray: Loaded image in CHW format.
    """
    img = Image.open(file_path)
    if not color:
        img = img.convert('L')  # Convert to grayscale
    img = np.array(img, dtype=dtype)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    else:
        img = img.transpose(2, 0, 1)
    return img / 255.0

# def resize_bbox(bbox, in_size, out_size):
#     y_scale = float(out_size[0]) / in_size[0]
#     x_scale = float(out_size[1]) / in_size[1]
#     bbox[:, 0] = y_scale * bbox[:, 0]
#     bbox[:, 2] = y_scale * bbox[:, 2]
#     bbox[:, 1] = x_scale * bbox[:, 1]
#     bbox[:, 3] = x_scale * bbox[:, 3]
#     return bbox

# def flip_bbox(bbox, size, y_flip=False, x_flip=False):
#     H, W = size
#     if y_flip:
#         y_max = H - bbox[:, 0]
#         y_min = H - bbox[:, 2]
#         bbox[:, 0] = y_min
#         bbox[:, 2] = y_max
#     if x_flip:
#         x_max = W - bbox[:, 1]
#         x_min = W - bbox[:, 3]
#         bbox[:, 1] = x_min
#         bbox[:, 3] = x_max
#     return bbox

# def random_flip(img, y_random=False, x_random=False, return_param=False, copy=False):
#     y_flip, x_flip = False, False
#     if y_random:
#         y_flip = random.choice([True, False])
#     if x_random:
#         x_flip = random.choice([True, False])
#     if y_flip:
#         img = img[:, ::-1, :]
#     if x_flip:
#         img = img[:, :, ::-1]
#     if copy:
#         img = img.copy()
#     if return_param:
#         return img, {'y_flip': y_flip, 'x_flip': x_flip}
#     else:
#         return img

def pytorch_normalize(img):
    normalize = tvtsf.Normalize(mean=[0.485], std=[0.229])
    img = normalize(t.from_numpy(img))
    return img.numpy()

# def caffe_normalize(img):
#     img = img[[2, 1, 0], :, :]  # RGB to BGR
#     img = img * 255
#     mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
#     img = (img - mean).astype(np.float32, copy=True)
#     return img

def preprocess(img, min_size=600, max_size=1000, caffe_pretrain=False):
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.0
    img = sktsf.resize(img, (C, int(H * scale), int(W * scale)), mode='reflect', anti_aliasing=False)
    normalize = pytorch_normalize
    return normalize(img)

# Dataset Classes

class ShipDataset:
    """Custom Ship Detection Dataset for Bounding Box Detection."""
    
    def __init__(self, data_dir, split='train', transform=None, caffe_pretrain=False):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.caffe_pretrain = caffe_pretrain
        self.image_dir = os.path.join(data_dir, 'images', split)
        self.label_dir = os.path.join(data_dir, 'labels', split)
        self.ids = [os.path.splitext(f)[0] for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        self.label_names = ['ship']

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        id_ = self.ids[i]
        img_file = os.path.join(self.image_dir, id_ + '.jpg')
        img = read_image(img_file, color=False)
        label_file = os.path.join(self.label_dir, id_ + '.txt')
        labels = []
        bbox = []
        with open(label_file, 'r') as f:
            for line in f:
                label, x_center, y_center, width, height = map(float, line.split())
                xmin = x_center - width / 2
                ymin = y_center - height / 2
                xmax = x_center + width / 2
                ymax = y_center + height / 2
                bbox.append([ymin, xmin, ymax, xmax])
                labels.append(label+1)
        bbox = np.stack(bbox).astype(np.float32)
        labels = np.stack(labels).astype(np.int32)

        if self.transform:
            img, bbox = self.transform((img, bbox))

        return img, bbox, labels , 1.0 # scale is always 1.0

    __getitem__ = get_example

# class Transform(object):
#     """Apply transformations like resizing and flipping to the dataset."""
    
#     def __init__(self, min_size=600, max_size=1000, caffe_pretrain=False):
#         self.min_size = min_size
#         self.max_size = max_size
#         self.caffe_pretrain = caffe_pretrain

#     def __call__(self, in_data):
#         img, bbox = in_data
#         _, H, W = img.shape
#         img = preprocess(img, self.min_size, self.max_size, self.caffe_pretrain)
#         _, o_H, o_W = img.shape
#         scale = o_H / H
#         bbox = resize_bbox(bbox, (H, W), (o_H, o_W))
#         img, params = random_flip(img, x_random=True, return_param=True)
#         bbox = flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])
#         return img, bbox

# Usage example
# transform = Transform(caffe_pretrain=opt.caffe_pretrain)

dataset = ShipDataset(data_dir='/home/buono/ObjDct_Repo/data/ShipDataset', split='train')
print(len(dataset))
print(dataset[0])