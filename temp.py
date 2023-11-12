# Deafault Imports
import os
import logging
import warnings as wr
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras as tfk
from keras import layers as tfkl
from keras import models as tfkm
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


# Random Configuration - All
RND = False
if not RND:
  SEED = 76998669
  os.environ['PYTHONHASHSEED'] = str(SEED)
  tf.compat.v1.set_random_seed(SEED)
  tf.random.set_seed(SEED)
  np.random.seed(SEED)
  rnd.seed(SEED)

# OS Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'

# Warning Congiguration
wr.simplefilter(action='ignore', category=FutureWarning)
wr.simplefilter(action='ignore', category=Warning)

# TensorFlow Configuration
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Initializate
dataset = np.load('./res/dataset/public_data_clean.npz', allow_pickle=True)
TEST_PERC = 0.15
VALI_PERC = 0.15

# Split into Images and Labels + Normalize the Image Data Set
KEYS = list(dataset.keys())
IMG = np.array(dataset[KEYS[0]])/255
LBL = np.array(dataset[KEYS[1]])
LIMG = len(IMG)
LLBL = len(LBL)

print('Images: ', LIMG)
print('Labels: ', LLBL)
print()

# Balances
print(pd.DataFrame(LBL, columns=['label'])['label'].value_counts())
print()

MPLBL = {0: "healthy", 1: "unhealthy"}
RMLBL = {"healthy": 0, "unhealthy": 1}
LBL = np.array([RMLBL[l] for l in LBL])
LBL = tf.keras.utils.to_categorical(LBL)

# Split the Data Set into Training XT YT, Validation XV YV, Test XTE YTE
Xtrv, XTE, Ytrv, YTE = train_test_split(
    IMG, 
    LBL, 
    test_size=int(TEST_PERC * LIMG), 
    **({"random_state":SEED} if not RND else {}), 
    stratify=LBL
)

XT, XV, YT, YV = train_test_split(
    Xtrv, 
    Ytrv, 
    test_size=int(VALI_PERC*LIMG), 
    **({"random_state":SEED} if not RND else {}), 
    stratify=Ytrv
)

# Shapes
print('Training Set Shape: ', XT.shape, YT.shape)
print('Validation Set Shape: ', XV.shape, YV.shape)
print('Test Set Shape: ', XTE.shape, YTE.shape)
print()

# Balances
print(pd.DataFrame(YV).value_counts())
print()

# Get the Shape of IN OUT
input_shape = XT.shape[1:]
output_shape = YT.shape[1]

print(f'Input shape of the Network: {input_shape}')
print(f'Output shape of the Network: {output_shape}')
print()

def apple_elixir_model(input_shape, output_shape, seed=SEED):

  tf.random.set_seed(seed)

  # Build the neural network layer by layer
  input_layer = tfkl.Input(shape=input_shape, name='Input')

  x = tfkl.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',name='conv0')(input_layer)
  x = tfkl.MaxPooling2D(name='mp0')(x)

  x = tfkl.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu',name='conv1')(x)
  x = tfkl.MaxPooling2D(name='mp1')(x)
  
  x = tfkl.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu',name='conv2')(x)
  x = tfkl.MaxPooling2D(name='mp2')(x)

  x = tfkl.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu',name='conv3')(x)
  x = tfkl.MaxPooling2D(name='mp3')(x)

  x = tfkl.Flatten()(x)
  x = tfkl.Dense(units = 512, activation='relu')(x)

  output_layer = tfkl.Dense(units=output_shape ,activation='softmax', name='Output')(x)

  model = tfk.Model(inputs=input_layer, outputs=output_layer, name='Convnet')
  model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(weight_decay=5e-4), metrics=['accuracy'])
  return model

BTC = 16
EPO = 100
NAME_PLOT = "test_push"
NAME_MODEL = "model_push"
CALLBACKS = [
    # tfk.callbacks.EarlyStopping(
    #   monitor='val_accuracy',
    #   mode='max',
    #   patience=20,
    #   restore_best_weights=True
    # ),
    # tfk.callbacks.ReduceLROnPlateau(
    #   monitor="val_accuracy",
    #   factor=0.1,
    #   patience=20,
    #   min_lr=1e-5,
    #   mode='max'
    # )
]


# MAIN
if __name__ == "__main__":
    model = apple_elixir_model(input_shape=input_shape, output_shape=output_shape)
    model.summary()

    history = model.fit(
    x = XT,                                                                         # We need to apply the preprocessing thought for the MobileNetV2 network
    y = YT,
    batch_size = BTC,
    epochs = EPO,
    validation_data = (XV, YV),                                               # We need to apply the preprocessing thought for the MobileNetV2 network
    callbacks = CALLBACKS
    ).history


    # Pre-processing the image 
    img = image.load_img(image_path, target_size = (150, 150)) 
    img_tensor = image.img_to_array(img) 
    img_tensor = np.expand_dims(img_tensor, axis = 0) 
    img_tensor = img_tensor / 255.
    
    # Print image tensor shape 
    print(img_tensor.shape) 
    
    # Print image 
    import matplotlib.pyplot as plt 
    plt.imshow(img_tensor[0]) 
    plt.show()

    # Outputs of the 8 layers, which include conv2D and max pooling layers 
    layer_outputs = [layer.output for layer in model.layers[:8]] 
    activation_model = r.Model(inputs = model.input, outputs = layer_outputs) 
    activations = activation_model.predict(img_tensor) 
    
    # Getting Activations of first layer 
    first_layer_activation = activations[0] 
    
    # shape of first layer activation 
    print(first_layer_activation.shape) 
    
    # 6th channel of the image after first layer of convolution is applied 
    plt.matshow(first_layer_activation[0, :, :, 6], cmap ='viridis') 
    
    # 15th channel of the image after first layer of convolution is applied 
    plt.matshow(first_layer_activation[0, :, :, 15], cmap ='viridis') 


























































    # DATASET = np.load("./res/dataset/public_data_rot_flip.npz", allow_pickle=True)
    # KEYS = list(DATASET.keys())
    # IMG = DATASET[KEYS[0]]
    # LBL = DATASET[KEYS[1]]

    # # print(len(IMG), len(LBL))

    # # Display a sample of images from the training-validation dataset
    # ROWS = 7
    # COLS = 7
    # fig, axes = plt.subplots(ROWS, COLS, figsize=(96,96))

    # # Iterate through the selected number of images
    # for i in range(ROWS):
    #     for j in range(COLS):
    #         index = (i*ROWS + j) + 33901
    #         img_norm = IMG[index]/255               
    #         axes[i,j].imshow(np.clip(img_norm, 0,1))
    #         axes[i,j].set_title(f'{LBL[index]}, {index}')  

    # # Adjust layout and display the images
    # plt.tight_layout()
    # plt.show()


    # # agum = []
    # # agum_lbl = []
    # # ROWS = 2
    # # COLS = 4
    # # LABELS = [
    # #     ["Normal", "Normal +90°", "Normal +180°", "Normal +270°"],
    # #     ["Flipped V", "Flipped V +90°", "Flipped V +180°", "Flipped V +270°"],
    # # ]     

    # # for z in range(len(IMG)):
    # #     print("Loop: ", z)
    # #     image = np.array(IMG[z])
    # #     for i in range(ROWS):
    # #         for j in range(COLS):
    # #             image = np.rot90(image)
    # #             if i==0 and j==3:
    # #                 continue
    # #             agum.append(image)
    # #             agum_lbl.append(LBL[z])
    # #         if i == 0:
    # #             image = np.flipud(image) 
    
    # # print(len(agum))
    # # print(len(agum_lbl))

    # # np.savez_compressed("public_data_rot_flip.npz", agum, agum_lbl)










    # # image = np.array(IMG[0])
    # # # Rotate 90° + Flip Hz + Flip
    # # # for i in range(3):
    # # #     image = np.rot90(image)
    # # #     plot_image(image, LBL[0])

    # # #     image = np.fliplr(image)
    # # #     plot_image(image, LBL[0])

    # # #     image = np.flipud(image)
    # # #     plot_image(image, LBL[0])


    # # agum = []
    # # agum_lbl = []
    # # ROWS = 2
    # # COLS = 4
    # # LABELS = [
    # #     ["Normal", "Normal +90°", "Normal +180°", "Normal +270°"],
    # #     ["Flipped V", "Flipped V +90°", "Flipped V +180°", "Flipped V +270°"],
    # #     # ["Flipped H", "Flipped H +90°", "Flipped H +180°", "Flipped H +270°"]
    # # ]        
    # # fig, axes = plt.subplots(ROWS, COLS, figsize=(96, 96))
    # # for i in range(ROWS):
    # #     for j in range(COLS):
    # #         image = np.rot90(image)
    # #         agum.append(image)
    # #         if i==0 and j==3:
    # #             continue
    # #         axes[i,j].imshow(np.clip(image/255, 0,1))                        # Show the image
    # #         axes[i,j].set_title(f'{LABELS[i][j]}')              # Show the corresponding digit label
    # #     if i == 0:
    # #         image = np.flipud(image)    
    # #     # if i == 1:
    # #         # image = np.flipud(image)    
    # #         # image = np.fliplr(image)

    # # # Adjust layout and display the images
    # # #plt.tight_layout()
    # # #plt.savefig(f'./res/img/{z}.jpg')
    # # plt.show()
    