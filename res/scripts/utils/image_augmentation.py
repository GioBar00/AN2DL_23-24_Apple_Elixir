from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

DATASET = np.load("public_data_clean.npz", allow_pickle=True)
KEYS = list(DATASET.keys())
images = DATASET[KEYS[0]]
labels = DATASET[KEYS[1]]

# Split the dataset into a combined training and validation set, and a separate test set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    images,
    labels,
    test_size = 0.1,
    stratify = labels
)

# Further split the combined training and validation set into a training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size = 0.20,
    stratify = y_train_val
)

datagen = ImageDataGenerator(
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=(0.2, 0.8)
    )

# create new array with shape of X_train, y_train
gen_X_train = np.empty_like(X_train)
gen_y_train = np.empty_like(y_train)

gen_images = 20
  # Number of images that has to be generated
for img in datagen.flow(X_train,y_train,batch_size = 1):
  gen_images -= 1
  gen_X_train = np.insert(gen_X_train,0,img[0][0],axis=0) 
  gen_y_train = np.insert(gen_y_train,0,img[1][0],axis=0)
  if gen_images < 0:
    break


# Display a sample of images from the training-validation dataset
num_img = 20
fig, axes = plt.subplots(1, num_img, figsize=(96, 96))

# Iterate through the selected number of images
for i in range(num_img):
    ax = axes[i % num_img]
    ax.imshow(gen_X_train[i]/255, cmap='gray')
    ax.set_title(f'{gen_y_train[i]}', fontsize=10)  # Show the corresponding label

# Adjust layout and display the images
plt.tight_layout()
plt.show()

exit()

# save dataset
# union X_train, y_train with gen_X_train, gen_y_train
X_train_a = np.concatenate((X_train, gen_X_train))
y_train_a = np.concatenate((y_train, gen_y_train))

# union X_val, y_val with X_test, y_test
X_val_test = np.concatenate((X_val, X_test))
y_val_test = np.concatenate((y_val, y_test))

np.savez_compressed("public_data_augmented.npz", X_train_a, y_train_a, X_val_test, y_val_test)

# load X_train, y_train, X_val, y_val, X_test, y_test
X_train_a, y_train_a, X_val_test, y_val_test = np.load("public_data_augmented.npz", allow_pickle=True).values()

# check the shape of the data
print("X_train shape: ", X_train_a.shape)
print("y_train shape: ", y_train_a.shape)
print("X_val_test shape: ", X_val_test.shape)
print("y_val_test shape: ", y_val_test.shape)



