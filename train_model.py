# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from tensorflow.keras.optimizers import SGD
import os 
import zipfile 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.image as mpimg
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
# initialize the number of training epochs and batch size
NUM_EPOCHS = 25
BS = 20
# derive the path to the directories containing the training,
# validation, and testing splits, respectively
TRAIN_PATH = os.path.sep.join(['/content/drive/My Drive/splitdata', "training"])
VAL_PATH = os.path.sep.join(['/content/drive/My Drive/splitdata', "validation"])
TEST_PATH = os.path.sep.join(['/content/drive/My Drive/splitdata', "testing"])
# determine the total number of image paths in training, validation,
# and testing directories
totalTrain = len(list(paths.list_images(TRAIN_PATH)))
totalVal = len(list(paths.list_images(VAL_PATH)))
totalTest = len(list(paths.list_images(TEST_PATH)))

# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=40,
	zoom_range=0.2,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	horizontal_flip=True,
	fill_mode="nearest")
# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	TRAIN_PATH,
	class_mode="binary",
	target_size=(150, 150),
	color_mode="rgb",
	shuffle=True,
	batch_size=20)
# initialize the validation generator
valGen = valAug.flow_from_directory(
	VAL_PATH,
	class_mode='binary',
	target_size=(150, 150),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)
# initialize the testing generator
testGen = valAug.flow_from_directory(
	TEST_PATH,
	class_mode='binary',
	target_size=(150, 150),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize our Keras implementation of ResNet model and compile it
model = InceptionV3(input_shape = (150, 150, 3), include_top = False, weights = 'imagenet')

for layer in model.layers:
    layer.trainable = False
x = layers.Flatten()(model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])


# train our Keras model
H = model.fit(trainGen, validation_data = valGen, steps_per_epoch = 60, epochs = 10)
