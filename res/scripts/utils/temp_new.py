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
from plot_image import plot_image

# MAIN
if __name__ == "__main__":
    DATASET = np.load("./res/npz/public_data.npz", allow_pickle=True)
    KEYS = list(DATASET.keys())
    IMG = DATASET[KEYS[0]]/255
    LBL = DATASET[KEYS[1]]


    image = np.array(IMG[0])
    # Rotate 90° + Flip Hz + Flip
    # for i in range(3):
    #     image = np.rot90(image)
    #     plot_image(image, LBL[0])

    #     image = np.fliplr(image)
    #     plot_image(image, LBL[0])

    #     image = np.flipud(image)
    #     plot_image(image, LBL[0])



    ROWS = 3
    COLS = 4
    LABELS = [
        ["Normal", "Normal +90°", "Normal +180°", "Normal +270°"],
        ["Flipped V", "Flipped V +90°", "Flipped V +180°", "Flipped V +270°"],
        ["Flipped H", "Flipped H +90°", "Flipped H +180°", "Flipped H +270°"]
    ]        
    fig, axes = plt.subplots(ROWS, COLS, figsize=(96, 96))
    for i in range(ROWS):
        for j in range(COLS):
            axes[i,j].imshow(np.clip(image, 0,1))                        # Show the image
            axes[i,j].set_title(f'{LABELS[i][j]}')              # Show the corresponding digit label
            image = np.rot90(image)
        if i == 0:
            image = np.flipud(image)    
        if i == 1:
            image = np.flipud(image)    
            image = np.fliplr(image)

    # Adjust layout and display the images
    #plt.tight_layout()
    #plt.savefig(f'./res/img/{z}.jpg')
    plt.show()