import numpy as np

DATASET = np.load("public_data.npz", allow_pickle=True)
KEYS = list(DATASET.keys())
IMAGES = DATASET[KEYS[0]]
LABELS = DATASET[KEYS[1]]

shrek = IMAGES[506]
trololo = IMAGES[529]

# remove duplicates
dup_indexes = np.unique(IMAGES, axis=0, return_index=True)[1]
IMAGES = [IMAGES[index] for index in sorted(dup_indexes)]
LABELS = [LABELS[index] for index in sorted(dup_indexes)]

# remove shrek and trololo
meme_indexes = []
for i, image in enumerate(IMAGES):
    if  np.array_equal(IMAGES[i], shrek) or np.array_equal(IMAGES[i], trololo):
        meme_indexes.append(i)
IMAGES = np.delete(IMAGES, meme_indexes, axis=0)
LABELS = np.delete(LABELS, meme_indexes, axis=0)

print(len(IMAGES))
print(len(LABELS))

np.savez("public_data_clean.npz", IMAGES, LABELS)