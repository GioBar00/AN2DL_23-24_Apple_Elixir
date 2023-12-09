import numpy as np
import matplotlib.pyplot as plt
import keras_cv as kcv
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
from sklearn.model_selection import train_test_split

RND = True
if not RND:
    seed = 7234562

images, labels = np.load('public_data_clean_balanced.npz', allow_pickle=True).values()

print('Images shape: ', images.shape)
print('Labels shape: ', labels.shape)

# Split the dataset into a combined training and validation set, and a separate test set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    images,
    labels,
    test_size = 0.15,
    **({"random_state":seed} if not RND else {}),
    stratify = labels
)

# Further split the combined training and validation set into a training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size = 0.15,
    **({"random_state":seed} if not RND else {}),
    stratify = y_train_val
)

# create a dataset object for each set
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (96, 96)

def to_dict(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, 2)
    return {"images": image, "labels": label}


def prepare_trainset(dataset):
    return (
            dataset.shuffle(10 * BATCH_SIZE)
            .map(to_dict, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
        )
    
def prepare_valset(dataset):
    return dataset.map(to_dict, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)


train_dataset = prepare_trainset(train_dataset)
val_dataset = prepare_valset(val_dataset)

def visualize_dataset(dataset, title):
    plt.figure(figsize=(6, 6)).suptitle(title, fontsize=18)
    for i, samples in enumerate(iter(dataset.take(9))):
        images = samples["images"]
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[0].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()


visualize_dataset(train_dataset, title="Before Augmentation")

rand_augment = kcv.layers.RandAugment(
    value_range=(0, 255),
    augmentations_per_image=3,
    magnitude=0.3,
    magnitude_stddev=0.2,
    rate=0.5,
)


def apply_rand_augment(inputs):
    inputs["images"] = rand_augment(inputs["images"])
    return inputs

res = train_dataset.map(apply_rand_augment, num_parallel_calls=AUTOTUNE)

visualize_dataset(res, title="After RandAugment")

mix_up = kcv.layers.MixUp()

def apply_mix_up(inputs):
    inputs = mix_up(inputs, training=True)
    return inputs

res = train_dataset.map(apply_mix_up, num_parallel_calls=AUTOTUNE)

visualize_dataset(res, title="After MixUp")

# custom pipeline
pipeline = kcv.layers.RandomAugmentationPipeline(
    layers=[kcv.layers.GridMask(), kcv.layers.Grayscale(output_channels=3)],
    augmentations_per_image=1,
)

def apply_pipeline(inputs):
    inputs["images"] = pipeline(inputs["images"])
    return inputs

res = train_dataset.map(apply_pipeline, num_parallel_calls=AUTOTUNE)

visualize_dataset(res, title="After RandomAugmentationPipeline")


def preprocess_for_model(inputs):
    images, labels = inputs["images"], inputs["labels"]
    images = tf.cast(images, tf.float32)
    return images, labels


train_dataset = (
    train_dataset
    .map(apply_rand_augment, num_parallel_calls=AUTOTUNE)
    .map(apply_mix_up, num_parallel_calls=AUTOTUNE)
    .map(preprocess_for_model, num_parallel_calls=AUTOTUNE)
)

val_dataset = (
    val_dataset
    .map(preprocess_for_model, num_parallel_calls=AUTOTUNE)
)

train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)

input_shape = IMAGE_SIZE + (3,)

print("Input shape: ", input_shape)