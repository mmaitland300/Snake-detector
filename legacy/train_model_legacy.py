# Legacy script archived from early Colab prototype.

import os

import tensorflow as tf
from imutils import paths
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

NUM_EPOCHS = 25
BS = 20

TRAIN_PATH = os.path.sep.join(["/content/drive/My Drive/splitdata", "training"])
VAL_PATH = os.path.sep.join(["/content/drive/My Drive/splitdata", "validation"])
TEST_PATH = os.path.sep.join(["/content/drive/My Drive/splitdata", "testing"])

totalTrain = len(list(paths.list_images(TRAIN_PATH)))
totalVal = len(list(paths.list_images(VAL_PATH)))
totalTest = len(list(paths.list_images(TEST_PATH)))

trainAug = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=40,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
valAug = ImageDataGenerator(rescale=1 / 255.0)

trainGen = trainAug.flow_from_directory(
    TRAIN_PATH,
    class_mode="binary",
    target_size=(150, 150),
    color_mode="rgb",
    shuffle=True,
    batch_size=20,
)
valGen = valAug.flow_from_directory(
    VAL_PATH,
    class_mode="binary",
    target_size=(150, 150),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS,
)
testGen = valAug.flow_from_directory(
    TEST_PATH,
    class_mode="binary",
    target_size=(150, 150),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS,
)

model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights="imagenet")
for layer in model.layers:
    layer.trainable = False
x = layers.Flatten()(model.output)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.models.Model(model.input, x)
model.compile(optimizer=RMSprop(lr=0.0001), loss="binary_crossentropy", metrics=["acc"])
H = model.fit(trainGen, validation_data=valGen, steps_per_epoch=60, epochs=10)
