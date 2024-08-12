from PIL import Image
import os
import numpy as np

# Path to the folder containing the images
image_folder = '/home/buono/ObjDct_Repo/data/ShipDataset/images/train'

# Path to the folder where numpy data will be saved
numpy_folder = '/home/buono/ObjDct_Repo/data/calibration_data'

# Create the numpy folder if it doesn't exist
os.makedirs(numpy_folder, exist_ok=True)

# Loop through each image in the image folder
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Read the image using OpenCV
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)
        image = image.convert("L")
        image = image.resize((88, 88))


        # Convert the image to numpy format
        numpy_data = np.array(image)

        # Save the numpy data to the numpy folder
        numpy_filename = os.path.splitext(filename)[0] + '.npy'
        numpy_path = os.path.join(numpy_folder, numpy_filename)
        np.save(numpy_path, numpy_data)

        print(f'Saved {numpy_filename} in {numpy_folder}')