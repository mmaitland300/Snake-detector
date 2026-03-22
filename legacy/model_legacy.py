# Legacy script archived from early Colab prototype.

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPooling2D,
    ZeroPadding2D,
    add,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


class ResNet:
    @staticmethod
    def residual_module(data, num_filters, stride, chan_dim, red=False, reg=0.0001):
        shortcut = data
        bn1 = BatchNormalization(axis=chan_dim)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(num_filters * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

        bn2 = BatchNormalization(axis=chan_dim)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(
            int(num_filters * 0.25),
            (3, 3),
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2(reg),
        )(act2)

        bn3 = BatchNormalization(axis=chan_dim)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(num_filters, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        if red:
            shortcut = Conv2D(num_filters, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

        return add([conv3, shortcut])

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=0.0001):
        input_shape = (height, width, depth)
        chan_dim = -1
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chan_dim = 1
        inputs = Input(shape=input_shape)
        x = BatchNormalization(axis=chan_dim)(inputs)
        x = Conv2D(filters[0], (5, 5), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = Activation("relu")(x)
        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        for i in range(0, len(stages)):
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride, chan_dim, red=True, reg=reg)
            for _ in range(0, stages[i] - 1):
                x = ResNet.residual_module(x, filters[i + 1], (1, 1), chan_dim, reg=reg)
        x = BatchNormalization(axis=chan_dim)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)
        return Model(inputs, x, name="resnet")
