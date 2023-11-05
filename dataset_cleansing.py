import numpy as np

DATASET = np.load("./res/npz/public_data.npz", allow_pickle=True)
KEYS = list(DATASET.keys())
IMAGES = DATASET[KEYS[0]]
LABELS = DATASET[KEYS[1]]

indices_meme = []

shrek = IMAGES[506]
trololo = IMAGES[529]
for i, image in enumerate(IMAGES):
    if  np.array_equal(IMAGES[i], shrek) or np.array_equal(IMAGES[i], trololo):
        indices_meme.append(i)

IMAGES = np.delete(IMAGES, indices_meme, axis=0)

np.savez("./res/npz/public_data_clean.npz", IMAGES, LABELS)