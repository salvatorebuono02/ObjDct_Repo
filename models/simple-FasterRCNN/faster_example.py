
import torch
import torchvision
from dataset import ShipDataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
# For training
images = []
targets = []
dataset = ShipDataset(data_dir='data/ShipDataset', split='train')
for i in range(len(dataset)):
    image, box, label, scale = dataset.get_example(i)
    images.append(torch.from_numpy(image))
    targets.append({'boxes': torch.from_numpy(box), 'labels': torch.from_numpy(label)})
output = model(images, targets)
# For inference
model.eval()
test_img = []
test_target = []
test_dataset = ShipDataset(data_dir='data/ShipDataset', split='test')
for i in range(len(test_dataset)):
    image, box, label, scale = test_dataset.get_example(i)
    test_img.append(torch.from_numpy(image))
    test_target.append({'boxes': torch.from_numpy(box), 'labels': torch.from_numpy(label)})

predictions = model(test_img)
print(predictions[0])
# # optionally, if you want to export the model to ONNX:
# torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)