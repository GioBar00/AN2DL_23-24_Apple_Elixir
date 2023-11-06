import os
import warnings as wr
import numpy as np
import logging
import random as rnd
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.python.keras import layers as tfkl
# from plot_image import plot_image

DATASET = np.load("public_data_clean.npz", allow_pickle=True)
KEYS = list(DATASET.keys())
IMG = DATASET[KEYS[0]]
LBL = DATASET[KEYS[1]]

print(IMG.shape)
print(LBL.shape)


# for all images in the dataset
aug_imgs = np.empty((len(IMG)*7, 96, 96, 3))
aug_lbls = [None] * len(IMG)*7

for i in range(len(IMG)):
    # for all rotations
    for j in range(3):
        # rotate image
        aug_imgs[i*7+j] = np.rot90(IMG[i], j + 1)
    # flip image
    aug_imgs[i*7+3] = np.flipud(IMG[i])
    # for all rotations
    for j in range(3):
        # rotate image
        aug_imgs[i*7+4+j] = np.rot90(aug_imgs[i*7+3], j + 1)
    # add 7 labels
    aug_lbls[i*7:i*7+7] = [LBL[i]] * 7


print(aug_imgs.shape)
#print(aug_lbls.shape)


# Display a sample of images from the training-validation dataset
ROWS = 7
COLS = 7
fig, axes = plt.subplots(ROWS, COLS, figsize=(96,96))

# Iterate through the selected number of images
for i in range(ROWS):
    for j in range(COLS):
        index = i*ROWS + j + 33901
        img_norm = aug_imgs[index]/255               
        axes[i,j].imshow(np.clip(img_norm, 0,1))
        axes[i,j].set_title(f'{aug_lbls[index]}, {index}')  

# Adjust layout and display the images
plt.tight_layout()
plt.show()


# save the augmented dataset
#np.savez_compressed("public_data_rot_flip", aug_imgs, aug_lbls)
