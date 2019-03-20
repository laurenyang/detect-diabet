import math
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow import keras
import densenet
import matplotlib.cm as cm
from vis.utils import utils
from keras import activations
from vis.visualization import visualize_cam, visualize_saliency, overlay
from vis.utils import utils
import pickle

#opens pickled model 
model = pickle.load(open('model.pickle', 'rb'))

#gets desired layer
last_layer = utils.find_layer_idx(model, "conv2d_120")

#loads images to get CAMs of
img1 = utils.load_img('/Users/laurenyang/Desktop/retinopathy-dataset-master/symptoms/1032_right.jpeg',target_size=(475, 316, 3))
img2 = utils.load_img('/Users/laurenyang/Desktop/github_images1/nosymptoms/119_right.jpeg', target_size=(475, 316, 3))
print(utils.get_img_shape(img1))
f, ax = plt.subplots(1, 2)
ax[0].imshow(img1)
ax[1].imshow(img2)
#displays original images
plt.show()

for modifier in [None]:
    plt.figure()
    f, ax = plt.subplots(1, 2)
    plt.suptitle("vanilla" if modifier is None else modifier)
    for i, img in enumerate([img1, img2]):    
        grads = visualize_cam(model, layer_idx=last_layer, filter_indices=[4], 
                              seed_input=img)         
        #Overlays the heatmap onto original image.    
        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
        plt.imshow(overlay(grads, img))
        plt.show()








