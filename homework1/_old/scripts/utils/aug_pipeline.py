import keras_cv as kcv
import numpy as np

value_range = (0, 255)
height=96
width=96
crop_area_factor=(0.08, 1.0)
aspect_ratio_factor=(3 / 4, 4 / 3)
grayscale_rate=0.2
color_jitter_rate=0.8
brightness_factor=0.2
contrast_factor=0.8
saturation_factor=(0.3, 0.7)
hue_factor=0.2

[
    kcv.preprocessing.RandomFlip("horizontal"),
    kcv.preprocessing.RandomCropAndResize(
        target_size=(height, width),
        crop_area_factor=crop_area_factor,
        aspect_ratio_factor=aspect_ratio_factor,
    ),
    kcv.preprocessing.RandomApply(
        kcv.preprocessing.Grayscale(output_channels=3),
        rate=grayscale_rate,
    ),
    kcv.preprocessing.RandomApply(
        kcv.preprocessing.RandomColorJitter(
            value_range=value_range,
            brightness_factor=brightness_factor,
            contrast_factor=contrast_factor,
            saturation_factor=saturation_factor,
            hue_factor=hue_factor,
        ),
        rate=color_jitter_rate,
    ),
],

class Random90RotFlip(kcv.layers.BaseImageAugmentationLayer):
    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.seed = seed
        self._ops = [lambda x: x]
        for j in range(1, 4):
            self._ops.append(lambda x: np.rot90(x, j + 1))
        self._ops.append(lambda x: np.flipud(x))
        for j in range(1, 4):
            self._ops.append(lambda x: np.rot90(np.flipud(x), j + 1))

    def augment_image(self, image, transformation, **kwargs):
        # randomly apply one of the 8 possible transformations
        if self.seed is not None:
            np.random.seed(self.seed)
        return self._ops[np.random.randint(8)](image)
    
    def augment_label(self, label, transformation, **kwargs):
        return label

    def get_config(self):
        config = {
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class KerasToCV(kcv.layers.BaseImageAugmentationLayer):
    def __init__(self, keras_layer, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.seed = seed
        self.keras_layer = keras_layer

    def augment_image(self, image, transformation, **kwargs):
        # randomly decide whether to apply the augmentation
        return self.keras_layer(image)
    
    def augment_label(self, label, transformation, **kwargs):
        return label

    def get_config(self):
        config = {
            "seed": self.seed,
            "keras_layer": self.keras_layer,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))