import tensorflow as tf
from tensorflow import keras as tfk
from keras import layers as tfkl
import numpy as np

IMAGE_SIZE = 96

def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

    Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.

    Returns:
        An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.

    Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.

    Returns:
        A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(img, _):
    img_min, img_max = img.min(), img.max()
    img = (img - img_min) / (img_max - img_min) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def equalize(img, _):
    img_flat = img.flatten()
    hist, bins = np.histogram(img_flat, bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    img_equalized = np.interp(img_flat, bins[:-1], cdf_normalized * 255).reshape(img.shape)
    return np.clip(img_equalized, 0, 255).astype(np.uint8)


def posterize(img, level):
    level = int_parameter(sample_level(level), 4)
    # Implement posterize logic without using PIL
    step = 256 // (4 - level)
    return ((img // step) * step).astype(np.uint8)


def rotate(img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                            Image.AFFINE, (1, 0, 0, level, 1, 0),
                            resample=Image.BILINEAR)


def translate_x(img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                            Image.AFFINE, (1, 0, level, 0, 1, 0),
                            resample=Image.BILINEAR)


def translate_y(img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                            Image.AFFINE, (1, 0, 0, 0, 1, level),
                            resample=Image.BILINEAR)

# Available operations
augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

def normalize(image):
  """Normalize input image channel-wise to zero mean and unit variance."""
  image = image.transpose(2, 0, 1)  # Switch to channel-first
  mean, std = np.array(MEAN), np.array(STD)
  image = (image - mean[:, None, None]) / std[:, None, None]
  return image.transpose(1, 2, 0)


def apply_op(image, op, severity):
  image = np.clip(image * 255., 0, 255).astype(np.uint8)
  image = op(image, severity)
  return image / 255.

def augmix(image, severity=3, width=3, depth=-1, alpha=1.):
    ws = np.float32(
      np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image)
    for i in range(width):
        image_aug = image.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op = np.random.choice(augmentations)
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * normalize(image_aug)

    mixed = (1 - m) * normalize(image) + m * mix
    return mixed



batch_size = 32
AUTO = tf.data.experimental.AUTOTUNE

trainloader = tf.data.Dataset.from_tensor_slices((XT, YT)).shuffle(1024).map(preprocess_image, num_parallel_calls=AUTO)

trainloader = (
    trainloader
    .shuffle(1024)
    .map(lambda x, y: augmix(x), num_parallel_calls=AUTO)
    .batch(batch_size)
    .prefetch(AUTO)
)