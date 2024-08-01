import os
import torchvision.transforms as T


DATA_PATH = 'data/ShipDataset'
CLASSES_PATH = '/home/buono/ObjDct_Repo/models/YOLOv1/classes.json'

BATCH_SIZE = 64
EPOCHS = 135
WARMUP_EPOCHS = 0
LEARNING_RATE = 1E-4

EPSILON = 1E-6
IMAGE_SIZE = (256, 256)

S = 7       # Divide each image into a SxS grid
B = 2       # Number of bounding boxes to predict
C = 1      # Number of classes in the dataset