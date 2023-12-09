import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
import keras as tfk
import keras.layers as tkl

def plot_images(images, labels, index=0, rows=1, cols=1, normalized=False):
  data_rescaling = tkl.Rescaling(scale=1./255)
  if normalized==False:
    IMGS_= data_rescaling(images)
  size = rows*cols
  diff = len(images)-(size+index)
  if diff <= 0:
    size = len(images)-index
    plt.figure(figsize=(cols,rows), dpi=200)
    for i in range(size):
      ax = plt.subplot(rows, cols, i + 1)
      plt.imshow(IMGS_[index + i])
      plt.title(labels[i], fontdict={"fontsize":5}, pad=2)
      plt.axis("off")

# -------- HOW TO USE --------
#
# images = array of images
# labels = array of labels 
# index = index of the first image to plot
# rows = number of rows
# cols = number of columns
# normalized = if True images are in 0..1 if False images are 0..255
#
# plot_images([IMGS[0]], [LBLS[0]])                                             # will plot the image IMGS[0] (notice you need to pass it as an array [] of one image)
# plot_images(IMGS[:10], LBLS[:10], 3, 2, 5)                                    # will plot the images from 3 to 10 displayed in a 2 rows x 5 columns grid -> 5 picture first row , 2 pictures second row
#
# -----------------------------