import tensorflow as tf
from tensorflow import keras as tfk
from keras import layers as tfkl

IMG_SIZE = 96

batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

resize = tf.keras.Sequential([
  tfkl.Resizing(IMG_SIZE, IMG_SIZE),
])

data_augmentation = tf.keras.Sequential([
  tfkl.RandomFlip("horizontal_and_vertical"),
  tfkl.RandomRotation(0.2),
])

def prepare(ds, shuffle=False, augment=False):
  # Resize and rescale all datasets.
  ds = ds.map(lambda x, y: (resize(x), y), 
              num_parallel_calls=AUTOTUNE)

  if shuffle:
    ds = ds.shuffle(1000)

  # Batch all datasets.
  ds = ds.batch(batch_size)

  # Use data augmentation only on the training set.
  if augment:
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=AUTOTUNE)

  # Use buffered prefetching on all datasets.
  return ds.prefetch(buffer_size=AUTOTUNE)