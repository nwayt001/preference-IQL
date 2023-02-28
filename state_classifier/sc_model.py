from keras.losses import SparseCategoricalCrossentropy
from tensorflow import keras

from keras import Model, Input
from keras.layers import Flatten, BatchNormalization, Layer, Conv2D, MaxPooling2D, Dropout, Dense, TimeDistributed


class ImageEncoder(Layer):
    # Input size: (128, 128, 3)

    def __init__(self, image_dim: int = 128, activation='relu', kernel_initializer='he_normal', conv1_filters=128,
                 conv1_kernel_size=(4, 4), conv2_filters=128, conv2_kernel_size=(4, 4), conv3_filters=128,
                 conv3_kernel_size=(4, 4), **kwargs):
        super().__init__(**kwargs)
        self.image_dim = image_dim
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.conv1_filters = conv1_filters
        self.conv1_kernel_size = conv1_kernel_size
        self.conv2_filters = conv2_filters
        self.conv2_kernel_size = conv2_kernel_size
        self.conv3_filters = conv3_filters
        self.conv3_kernel_size = conv3_kernel_size

        self.conv1 = Conv2D(self.conv1_filters, self.conv1_kernel_size, activation=self.activation, padding='same',
                            kernel_initializer=self.kernel_initializer, strides=[1, 1])
        self.conv2 = Conv2D(self.conv2_filters, self.conv2_kernel_size, activation=self.activation, padding='same',
                            kernel_initializer=self.kernel_initializer, strides=[1, 1])
        self.conv3 = Conv2D(self.conv3_filters, self.conv3_kernel_size, activation=self.activation, padding='same',
                            kernel_initializer=self.kernel_initializer, strides=[1, 1])
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.pool1 = MaxPooling2D((2, 2))
        self.pool2 = MaxPooling2D((2, 2))
        self.pool3 = MaxPooling2D((2, 2))

    def __call__(self, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


class Classifier(Layer):
    def __init__(self, num_classes, activation='relu', output_activation='softmax', kernel_initializer='he_normal',
                 output_kernel_initializer='glorot_uniform', dense1_size=256, dense2_size=256, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.activation = activation
        self.output_activation = output_activation
        self.kernel_initializer = kernel_initializer
        self.output_kernel_initializer = output_kernel_initializer
        self.dense1_size = dense1_size
        self.dense2_size = dense2_size

        self.dense1 = Dense(self.dense1_size, activation=self.activation, kernel_initializer=self.kernel_initializer)
        self.dense2 = Dense(self.dense2_size, activation=self.activation, kernel_initializer=self.kernel_initializer)
        self.dense3 = Dense(self.num_classes, activation=self.output_activation) # , kernel_initializer=self.output_kernel_initializer)

        self.dropout1 = Dropout(0.5)
        self.dropout2 = Dropout(0.5)

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()

    def __call__(self, x, *args, **kwargs):
        # x = self.dropout1(x)
        x = self.dense1(x)
        # x = self.bn1(x)
        # x = self.dropout2(x)
        x = self.dense2(x)
        # x = self.bn2(x)
        x = self.dense3(x)
        return x


def build_state_classifier_model(num_classes, image_size=128):
    encoder = ImageEncoder()
    classifier = Classifier(num_classes)

    _input = Input(shape=(image_size, image_size, 3))
    x = encoder(_input)
    x = Flatten()(x)
    output = classifier(x)

    model = Model(inputs=_input, outputs=output, name="state_classifier")
    optimizer = keras.optimizers.get("adam")
    optimizer.learning_rate = 1e-3
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model
