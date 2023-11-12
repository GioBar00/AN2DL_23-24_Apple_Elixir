import tensorflow_probability as tfp

def preprocess_image(image, label):
  image = tf.image.resize(image, (96, 96))
  image = tf.image.convert_image_dtype(image, tf.float32)
  return image, label

tfd = tfp.distributions
# For Beta distribution.
alpha = [0.2]
beta = [0.2]

def mixup(a, b):
  
  (image1, label1), (image2, label2) = a, b

  dist = tfd.Beta(alpha, beta)
  l = dist.sample(1)[0][0]
  
  img = l*image1+(1-l)*image2
  lab = l*label1+(1-l)*label2

  return img, lab

AUTO = tf.data.experimental.AUTOTUNE
batch_size = 64


trainloader1 = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).map(preprocess_image, num_parallel_calls=AUTO)
trainloader2 = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).map(preprocess_image, num_parallel_calls=AUTO)

trainloader = tf.data.Dataset.zip((trainloader1, trainloader2))
trainloader = (
    trainloader
    .shuffle(1024)
    .map(mixup, num_parallel_calls=AUTO)
    .batch(batch_size)
    .prefetch(AUTO)
)

testloader = tf.data.Dataset.from_tensor_slices((X_test, y_test))
testloader = (
    testloader
    .map(preprocess_image, num_parallel_calls=AUTO)
    .batch(batch_size)
    .prefetch(AUTO)
)

# PRINT IMAGES
for images, labels in trainloader.take(1):
  print('images.shape: ', images.shape)
  print('labels.shape: ', labels.shape)
  
  # Display a sample of images from the training-validation dataset
  num_img = 10
  fig, axes = plt.subplots(1, num_img, figsize=(96, 96))

  # Iterate through the selected number of images
  for i in range(num_img):
      ax = axes[i % num_img]
      ax.imshow(images[i]/255, cmap='gray')
      #ax.set_title(f'{labels_map[np.argmax(labels[i], axis=-1)]}', fontsize=40)  # Show the corresponding label
      ax.set_title(f'{labels[i]}', fontsize=40)

  # Adjust layout and display the images
  plt.tight_layout()
  plt.show()

  # Train the model and save its history
history = model.fit(
    trainloader,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=testloader,
    callbacks=callbacks
).history