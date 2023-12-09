import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def plot_images(images, labels, index=0, rows=1, cols=1, normalized=False):
  size = rows*cols
  diff = len(images)-(size+index)
  if diff <= 0:
    size = len(images)-index
    plt.figure(figsize=(cols,rows), dpi=200)
    for i in range(size):
      ax = plt.subplot(rows, cols, i + 1)
      plt.imshow(images[index + i] if normalized else images[index + i]/255, cmap='gray')
      plt.title(labels[i], fontdict={"fontsize":5}, pad=2)
      plt.axis("off")
    plt.tight_layout()
    plt.show()

def random_90_rotation_flip(images, labels, num_gen):
    gen_images_shape = (num_gen, images.shape[1], images.shape[2], images.shape[3])
    gen_images = np.zeros(gen_images_shape)
    # array of labels (strings) for generated images
    gen_labels = np.empty(num_gen, dtype=object)
    ops = []
    for j in range(1, 4):
        ops.append(lambda x: np.rot90(x, j + 1))
    ops.append(lambda x: np.flipud(x))
    for j in range(1, 4):
        ops.append(lambda x: np.rot90(np.flipud(x), j + 1))
    # random choice of num_gen operations from len(gen_images)*len(ops) operations
    idxs = range(len(gen_images) * len(ops))
    idxs = np.random.choice(idxs, num_gen, replace=False)
    i = 0
    for idx in idxs:
        gen_images[i] = ops[idx % len(ops)](images[idx // len(ops)])
        gen_labels[i] = labels[idx // len(ops)]
        i += 1
    return gen_images, gen_labels


images, labels = np.load('public_data_clean.npz', allow_pickle=True).values()

print('Images: ', len(images))
print('Labels: ', len(labels))

print('Images shape: ', images.shape)
print('Labels shape: ', labels.shape)

healthy_images = images[labels == 'healthy']
unhealthy_images = images[labels == 'unhealthy']

print('Healthy images: ', len(healthy_images))
print('Unhealthy images: ', len(unhealthy_images))
diff = len(healthy_images) - len(unhealthy_images)
print('Difference: ', diff)

if abs(diff) > 0:
    if len(healthy_images) < len(unhealthy_images):
        gen_images, gen_labels = healthy_images, labels[labels == 'healthy']
    else:
        gen_images, gen_labels = unhealthy_images, labels[labels == 'unhealthy']

    gen_images, gen_labels = random_90_rotation_flip(gen_images, gen_labels, abs(diff))
    # plot_images(gen_images, gen_labels, rows=3, cols=3, normalized=False)
    print('Generated images: ', len(gen_images))
    print('Generated labels: ', len(gen_labels))
    images = np.concatenate((images, gen_images))
    labels = np.concatenate((labels, gen_labels))

    print('Images: ', len(images))
    print('Labels: ', len(labels))
    print('Healthy images: ', len(images[labels == 'healthy']))
    print('Unhealthy images: ', len(images[labels == 'unhealthy']))

# shuffle images and labels in the same way
# idxs = np.arange(len(images))
# np.random.shuffle(idxs)
# images = images[idxs]
# labels = labels[idxs]

# split dataset into train-validation and test sets
#X_tr_val, X_test, y_tr_val, y_test = train_test_split(images, labels, test_size=0.15, stratify=labels)

# save dataset as dictionary

dataset = {
    'images': images,
    'labels': labels
}

np.savez_compressed('public_data_clean_balanced.npz', **dataset)

# save train-validation and test sets
#np.savez_compressed('public_data_clean_balanced_splitted.npz', X_tr_val, y_tr_val, X_test, y_test)