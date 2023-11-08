<img src="./res/img/cover.jpeg" style="border-radius: 3em">



# :memo: About the Project
The scope of this project is to track the progress of the Homeworks for AN2DL at PoliMI accademic year 23-24

<br />

## :book: Homework 1
### :clock3: Deadlines
#### :computer: Dev. Phase - Start: 02/11/2023 - 23:00
#### :triangular_flag_on_post: Final Phase - Start: 15/11/2023 - 23:00
#### :crossed_flags: Competition Ends: 17/11/2023 - 22:59  

<br />

### :dart: Task
Classify plants that are divided into two categories according to their state of health. It is a binary classification problem, so the goal is to predict the correct class label in {0, 1}.

<img style="display: block; margin: 0 auto" src="./res/img/healthy_unhealthy_example.png" />
<table style="margin: 0 auto;">
    <tr>
        <th style="width:106px">0 - Healthy</th>
        <th style="width:106px">1 - Unhealthy</th>
    </tr>
</table>

### :floppy_disk: Dataset Details
- Image Size: 96x96
- Color space: RGB
- File Format: npz
- Number of classes: 2
- Classes:
    - 0: "healthy"
    - 1: "unhealthy"

Inside the ./res/npz/ folder you'll find the public_data.npz which contains:
- "data" : 3-dimensional numpy array of shape 5100x96x96x3, containing the RGB images
- "labels" : 1-dimensioanl numpy array of shape 5200 with values in {'healthy', 'unhealthy'}

<br />

To Read the Data use:
```
    numpy.load('public_data.npz', allow_pickle=True)
```
:warning: no automatic validation set is provided.


<br />

### :globe_with_meridians: External Data
#### :white_check_mark: Allowed
- Libraries not seen during lectures and labs (bearing in mind that you will be limited to those selected by us during submission)
- Models pre-trained on imagenet by keras.applications (for transfer learning and/or fine tuning)
#### :x: Not Allowed
- Any source of data that has not been provided by us, especially if to be used for training purposes
- Any pre-trained model not belonging to keras.applications and/or not pre-trained on imagenet 


<br />

### :headphones: Playlist Suggestions

- [Apeiron](https://open.spotify.com/playlist/4n1ospIm5afsGRvWvCt0Ab?si=416f37db8a70413f)
- [Balance](https://open.spotify.com/playlist/4W3rpOJGsJeoEY2HFz3GNf?si=f2be91354aaa4f31)
- [Lair of the white rabbit](https://open.spotify.com/playlist/58m1g8X3E41wU5Do1A5trZ?si=095c94e653c2447f)

<br />

## :round_pushpin: Roadmap
- [ ] Data Cleaning / Data Cleansing :hourglass:
- [ ] Rotate / Reflect the images to enlarge the dataset
- [ ] Choose the Network for the feature exctraction
- [ ] Build the Classifier


<br />
<br />
<br />
<br />
<br />
<br />
<br />

# Table of Contents

# 1. Task and Goal
## 1.1 Task Description
## 1.2 Goal 
<br/>

# 2. Development
## 2.1 Brainstorming
## 2.2 Data Cleansing
## 2.3 Models
### 2.3.1 Model A
### 2.3.2 Model B
### 2.3.3 Model C
## 2.4 Other Functions
<br />

# 3. Tests and Final Evaluation
## 3.1 Test Model A
## 3.2 Test Model B
## 3.3 Test Model C
## 3.4 Comparisons
## 3.5 Final Evaluation
<br />



## Transfer Learning
Take a CNN already trained (with high performance) and attach the top layer (Dense NN):

:hourglass: - Ongoing Test Alex

:clock3: - Ongoing Test Andrea

:moon: - Ongoing Test Giovanni

- [ ] Example NOT TESTED -
- [X] Example TESTED - 0.5 val_acc - 

<br />

- [ ] [Xception](https://keras.io/api/applications/xception) - no smaller than 71x71x3 - ongoing testing @Andrea 

- [ ] [VGG16](https://keras.io/api/applications/vgg/#vgg16-function) - no smaller than 32x32x3 -
- [ ] [VGG19](https://keras.io/api/applications/vgg/#vgg19-function) - no smaller than 32x32x3 -

- [ ] [ResNet50](https://keras.io/api/applications/resnet/#resnet50-function) - no smaller than 32x32x3 -
- [ ] [ResNet50V2](https://keras.io/api/applications/resnet/#resnet50v2-function) - no smaller than 32x32x3 -
- [ ] [ResNet101](https://keras.io/api/applications/resnet/#resnet101-function) - no smaller than 32x32x3 -
- [ ] [ResNet101V2](https://keras.io/api/applications/resnet/#resnet101v2-function) - no smaller than 32x32x3 -
- [ ] [ResNet152](https://keras.io/api/applications/resnet/#resnet152-function) - no smaller than 32x32x3 -
- [ ] [ResNet152V2](https://keras.io/api/applications/resnet/#resnet152v2-function) - no smaller than 32x32x3 -

- [ ] [InceptionV3](https://keras.io/api/applications/inceptionv3) - no smaller than 75x75x3 -
- [ ] [InceptionResNetV2](https://keras.io/api/applications/inceptionresnetv2) - no smaller than 75x75x3 -

- [ ] [MobileNet](https://keras.io/api/applications/mobilenet/#mobilenet-function) - no smaller than 32x32x3 -
- [x] [MobileNetV2](https://keras.io/api/applications/mobilenet/#mobilenetv2-function) - no smaller than ?x?x3 - 0.78 score on codalab
- [ ] [MobileNetV3Small](https://keras.io/api/applications/mobilenet/#mobilenetv3small-function) - no smaller than ?x?x3 -
- [ ] [MobileNetV3Large](https://keras.io/api/applications/mobilenet/#mobilenetv3large-function) - no smaller than ?x?x3 -

- [ ] [DenseNet121](https://keras.io/api/applications/densenet/#densenet121-function) - no smaller than 32x32x3 -
- [ ] [DenseNet169](https://keras.io/api/applications/densenet/#densenet169-function) - no smaller than 32x32x3 -
- [ ] [DenseNet201](https://keras.io/api/applications/densenet/#densenet201-function) - no smaller than 32x32x3 -

- [ ] [NASNetMobile](https://keras.io/api/applications/nasnet/#nasnetmobile-function) - no smaller than 32x32x3 -
- [ ] [NASNetLarge](https://keras.io/api/applications/nasnet/#nasnetlarge-function) - no smaller than 32x32x3 -

- [x] [EfficientNetB0](https://keras.io/api/applications/efficientnet/#efficientnetb0-function) - no smaller than ?x?x3 - 0.5 on validation
- [] [EfficientNetB1](https://keras.io/api/applications/efficientnet/#efficientnetb1-function) - no smaller than ?x?2x3 -
- [x] [EfficientNetB2](https://keras.io/api/applications/efficientnet/#efficientnetb2-function) - no smaller than ?x?x3 - 0.5 on validation
- [ ] [EfficientNetB3](https://keras.io/api/applications/efficientnet/#efficientnetb3-function) - no smaller than ?x?x3 -
- [x] [EfficientNetB4](https://keras.io/api/applications/efficientnet/#efficientnetb4-function) - no smaller than ?x?x3 - 0.5 on validation
- [x] [EfficientNetB5](https://keras.io/api/applications/efficientnet/#efficientnetb5-function) - no smaller than ?x?x3 - 0.5 on validation
- [ ] [EfficientNetB6](https://keras.io/api/applications/efficientnet/#efficientnetb6-function) - no smaller than ?x?x3 -
- [ ] [EfficientNetB7](https://keras.io/api/applications/efficientnet/#efficientnetb7-function) - no smaller than ?x?x3 -

- [ ] [EfficientNetV2B0](https://keras.io/api/applications/efficientnet_v2/#efficientnetv2b0-function) - no smaller than - 
- [ ] [EfficientNetV2B1](https://keras.io/api/applications/efficientnet_v2/#efficientnetv2b1-function) - no smaller than - 
- [ ] [EfficientNetV2B2](https://keras.io/api/applications/efficientnet_v2/#efficientnetv2b2-function) - no smaller than - 
- [ ] [EfficientNetV2B3](https://keras.io/api/applications/efficientnet_v2/#efficientnetv2b3-function) - no smaller than - 

- [ ] [EfficientNetV2S](https://keras.io/api/applications/efficientnet_v2/#efficientnetv2s-function) - no smaller than - 
- [ ] [EfficientNetV2M](https://keras.io/api/applications/efficientnet_v2/#efficientnetv2m-function) - no smaller than - 
- [ ] [EfficientNetV2L](https://keras.io/api/applications/efficientnet_v2/#efficientnetv2l-function) - no smaller than - 

- [ ] [ConvNeXtTiny](https://keras.io/api/applications/convnext/#convnexttiny-function)
- [ ] [ConvNeXtSmall](https://keras.io/api/applications/convnext/#convnextsmall-function)
- [ ] [ConvNeXtBase](https://keras.io/api/applications/convnext/#convnextbase-function)
- [ ] [ConvNeXtLarge](https://keras.io/api/applications/convnext/#convnextlarge-function)
- [ ] [ConvNeXtXLarge](https://keras.io/api/applications/convnext/#convnextxlarge-function)
