from keras.preprocessing.image import ImageDataGenerator
import numpy as np

DATASET = np.load("public_data.npz", allow_pickle=True)
KEYS = list(DATASET.keys())
images = DATASET[KEYS[0]]
labels = DATASET[KEYS[1]]

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

gen_images = 2000 
  # Number of images that has to be generated
for img in datagen.flow(images,labels,batch_size = 1):
  gen_images -= 1
  images = np.insert(images,0,img[0][0],axis=0)
  labels = np.insert(labels,0,img[1][0],axis=0)
  if gen_images < 0:
    break

