import numpy as np

from "./res/scripts/utils/plot_image" import plot_image

DATASET = np.load("./res/npz/public_data.npz", allow_pickle=True)
KEYS = list(DATASET.keys())
IMG = DATASET[KEYS[0]]
LBL = DATASET[KEYS[1]]


image = np.array(IMG[0])