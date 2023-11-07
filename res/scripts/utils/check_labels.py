import numpy as np

dataset = np.load("public_data.npz", allow_pickle=True)
keys = list(dataset.keys())
images = dataset[keys[0]]
labels = dataset[keys[1]]

dataset_clean = np.load("public_data_clean.npz", allow_pickle=True)
keys_clean = list(dataset_clean.keys())
images_clean = dataset_clean[keys_clean[0]]
labels_clean = dataset_clean[keys_clean[1]]


shrek = images[506]
trololo = images[529]

meme_indexes = []
for i, image in enumerate(images):
    if  np.array_equal(images[i], shrek) or np.array_equal(images[i], trololo):
        meme_indexes.append(i)

meme_indexes = np.array(meme_indexes)

valid_indexes = np.unique(images, axis=0, return_index=True)[1]

# remove meme_indexes from valid_indexes
valid_indexes = np.setdiff1d(valid_indexes, meme_indexes)
print(len(valid_indexes))

# check if labels are the same
print(np.array_equal(labels[valid_indexes], labels_clean))
# check if images are the same
print(np.array_equal(images[valid_indexes], images_clean))