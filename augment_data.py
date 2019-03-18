import math
import numpy as np
import os
import glob
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow import keras
import densenet
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from imgaug import augmenters as iaa

#original data directory
DATADIR = "/Users/laurenyang/Desktop/testing" 

#augmented data directory
AUGDIR = "/Users/laurenyang/Desktop/train-1-aug"

# Create imgaug custom function 
def custom_aug(image):
    seq = iaa.Sequential([
        # Gaussian blur with random sigma between 1.0 and 2.0, blur 50% of images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(1.0, 2.0))
        ),
        iaa.Sometimes(0.5,
            iaa.GammaContrast((1.0, 2.0)), # contrasts 50% of images.
        ),
        iaa.ContrastNormalization((0.75, 3)),
        iaa.Multiply((0.75, 1.25), per_channel=.5) #changes brightness + color by 75-125% of original value, 30% of the time for each channel
    ], random_order=True)
    images_aug = seq.augment_image(image)
    return images_aug

# Create Augmented Data Generator
aug_datagen = ImageDataGenerator(
    brightness_range=[0,1.5],
    channel_shift_range=5,
    preprocessing_function=custom_aug)

augment_generator = aug_datagen.flow_from_directory(
    directory=DATADIR,
    target_size=(316, 475),
    color_mode="rgb",
    batch_size=1,
    class_mode="binary",
    shuffle=False,
    seed=42,
    save_to_dir=AUGDIR, 
    save_prefix='aug', 
    save_format='jpeg',
)

i = 0
for batch in augment_generator:
    #print out orig name 
    idx = (augment_generator.batch_index - 1) * augment_generator.batch_size
    orig_path = augment_generator.filenames[idx : idx + augment_generator.batch_size]
    orig_name_extracted = orig_path[0][orig_path[0].rfind('/')+1:]
    list_of_files = glob.glob('/Users/laurenyang/Desktop/train-1-aug/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print("latest file is: " + latest_file)
    os.rename(latest_file, latest_file[:latest_file.rfind('_')-1]+orig_name_extracted)

    i = i + 1
    if i == 25402:
        break  # otherwise the generator would loop indefinitely

