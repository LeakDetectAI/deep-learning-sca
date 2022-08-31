import numpy as np
from keras.utils import to_categorical
from keras.layers import Flatten, Dense, Input, AveragePooling2D, Conv2D, Activation
from keras.models import Model
from keras.optimizers import RMSprop
from keras.regularizers import l2

from deepscapy.core.sca_nn_model import SCANNModel
from deepscapy.constants import TWOD_CNN_SQR
import logging


# Some of the Model Code is from ASCAD Paper Github Repository (https://github.com/ANSSI-FR/ASCAD)
class CustomLeNet5Square(SCANNModel):
    def __init__(self, model_name, num_classes, input_dim=1400, loss_function='categorical_crossentropy',
                 kernel_regularizer=l2(l=1e-5), kernel_initializer="he_uniform",
                 optimizer=RMSprop(learning_rate=1e-4), metrics=['accuracy'], weight_averaging=False, **kwargs):
        input_dim = self.get_reshaped_input_dim(input_dim)
        self.logger = logging.getLogger(CustomLeNet5Square.__name__)
        self.logger.info("New Reshaped input size {}".format(input_dim))
        super(CustomLeNet5Square, self).__init__(model_name=model_name, num_classes=num_classes,
                                                 input_dim=input_dim, model_type=TWOD_CNN_SQR,
                                                 loss_function=loss_function, kernel_regularizer=kernel_regularizer,
                                                 kernel_initializer=kernel_initializer, optimizer=optimizer,
                                                 metrics=metrics, weight_averaging=weight_averaging, **kwargs)

    def _construct_model_(self, **kwargs):
        img_input = Input(shape=self.input_dim)
        # Block 1
        x = Conv2D(64, (5, 5), activation='selu', padding='same', name='block1_conv2d', **kwargs)(img_input)
        x = AveragePooling2D(name='block1_averagepool2d')(x)

        # Block 2
        x = Conv2D(128, (5, 5), activation='selu', padding='same', name='block2_conv2d', **kwargs)(x)
        x = AveragePooling2D(name='block2_averagepool2d')(x)

        # Block 3
        # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2d', **kwargs)(x)
        # x = AveragePooling2D(name='block3_averagepool2d')(x)

        # Classification Block
        x = Flatten(name='flatten_block_lenet_rect_2d')(x)
        x = Dense(512, activation='selu', name='fc1_lenet_5_rect', **kwargs)(x)
        x = Dense(128, activation='selu', name='fc2_lenet_5_rect', **kwargs)(x)
        scores = Dense(self.num_classes, activation=None, name='scores_lenet_5_rect',
                       kernel_regularizer=self.kernel_regularizer)(x)
        predictions = Activation('softmax', name='predictions_lenet_5_rect')(scores)

        model = Model(img_input, predictions, name='cnn_lenet_2d')
        scoring_model = Model(img_input, scores, name='cnn_lenet_2d')

        # optimizer = RMSprop(learning_rate=0.00001)
        # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model, scoring_model

    def get_reshaped_input_dim(self, features):
        image_size = int(np.floor(np.sqrt(features)) + 1)
        input_shape = (image_size, image_size, 1)
        return input_shape

    def reshape_inputs(self, X, y):
        features = X.shape[-1]
        new_features = self.input_dim[0]
        n = new_features * new_features - features
        constant_values = np.mean(X)
        X = np.lib.pad(X, ((0, 0), (0, n)), 'constant', constant_values=constant_values)
        X = X.reshape(-1, new_features, new_features, 1)
        y = to_categorical(y, num_classes=self.num_classes)
        return X, y

    def fit(self, X, y, epochs=200, batch_size=100, verbose=1, **kwargs):
        return super().fit(X, y, epochs, batch_size, verbose, **kwargs)

    def predict_scores(self, X, verbose=0, **kwargs):
        return super().predict_scores(X, verbose, **kwargs)

    def evaluate(self, X, y, verbose=1, **kwargs):
        return super().evaluate(X, y, verbose, **kwargs)

    def summary(self, **kwargs):
        super().summary(**kwargs)
