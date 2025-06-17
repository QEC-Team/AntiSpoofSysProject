

"""



"""

import os
import argparse
import random
from shutil import copy2
from glob import glob

# Setup argument parser
parser = argparse.ArgumentParser(description='Split dataset into training and validation sets.')
parser.add_argument("-image_dir", help='Path to the directory containing images.', default='/media/muhannad/wdd3/thamer/perfect/sdp4/AntiSpoofSys/License_Plate_Recognation_Dataset', type=str)
parser.add_argument("-train_dir", help='Directory to save training images.', default='/media/muhannad/wdd3/thamer/perfect/sdp4/AntiSpoofSys/LPRNet/data/LPR_Data_Train', type=str)
parser.add_argument("-val_dir", help='Directory to save validation images.', default='//media/muhannad/wdd3/thamer/perfect/sdp4/AntiSpoofSys/LPRNet/data/LPR_Data_Test', type=str)
parser.add_argument("-val_ratio", help='Ratio of validation to total images.', default=0.1, type=float)
args = parser.parse_args()

# List all images
image_paths = glob(os.path.join(args.image_dir, '*.jpg'))  # Adjust pattern as needed
random.shuffle(image_paths)

# Calculate split index
num_val = int(len(image_paths) * args.val_ratio)
val_paths = image_paths[:num_val]
train_paths = image_paths[num_val:]

# Function to copy files
def copy_files(file_paths, destination):
    for path in file_paths:
        try:
            copy2(path, destination)
        except Exception as e:
            print(f"Error copying file {path}: {e}")

# Copy files to their respective directories
copy_files(train_paths, args.train_dir)
copy_files(val_paths, args.val_dir)

print(f'Done! {len(train_paths)} training images and {len(val_paths)} validation images have been copied.')

