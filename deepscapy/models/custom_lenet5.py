import math

from keras.layers import Flatten, Dense, Input, AveragePooling2D, Conv2D, Activation, Dropout, \
    BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop
from keras.regularizers import l2

from deepscapy.constants import ASCAD_DESYNC0, ASCAD_DESYNC50, ASCAD_DESYNC100, TWOD_CNN_SQR
from deepscapy.core.sca_nn_model import SCANNModel


# Some of the Model Code is from ASCAD Paper Github Repository (https://github.com/ANSSI-FR/ASCAD)
class CustomLeNet5(SCANNModel):
    def __init__(self, model_name, num_classes, input_dim=625, loss_function='categorical_crossentropy',
                 kernel_regularizer=l2(l=1e-5), kernel_initializer="he_uniform", optimizer=RMSprop(learning_rate=1e-5),
                 metrics=['accuracy'], dataset_type=ASCAD_DESYNC0, weight_averaging=False, **kwargs):
        self.de_synchronization = (dataset_type == ASCAD_DESYNC50 or dataset_type == ASCAD_DESYNC100)
        image_size = int(math.sqrt(input_dim))
        input_dim = (image_size, image_size)
        super(CustomLeNet5, self).__init__(model_name=model_name, num_classes=num_classes, input_dim=input_dim,
                                           model_type=TWOD_CNN_SQR, loss_function=loss_function,
                                           kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                                           optimizer=optimizer, metrics=metrics, weight_averaging=weight_averaging,
                                           **kwargs)

    def _construct_model_(self, **kwargs):
        input_shape = (self.input_dim[0], self.input_dim[1], 1)
        img_input = Input(shape=input_shape)

        # Block 1
        x = Conv2D(64, (5, 5), activation='relu', padding='same', name='block1_conv2d', **kwargs)(img_input)
        x = BatchNormalization()(x)
        x = AveragePooling2D(name='block1_averagepool2d')(x)

        # Block 2
        x = Conv2D(128, (5, 5), activation='relu', padding='same', name='block2_conv2d', **kwargs)(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D(name='block2_averagepool2d')(x)

        # Add block 3 for complex datasets
        # if self.dataset_type == ASCAD_DESYNC50 or self.dataset_type == ASCAD_DESYNC100:
        #     x = Conv2D(512, (5, 5), activation='relu', padding='same', name='block3_conv2d', **kwargs)(x)
        #     x = AveragePooling2D(name='block3_averagepool2d')(x)

        # Classification block
        x = Flatten(name='flatten_block_lenet_5')(x)
        x = Dense(1000, activation='relu', name='fc1_lenet_5', **kwargs)(x)
        x = Dropout(0.25)(x)
        x = Dense(750, activation='relu', name='fc2_lenet_5', **kwargs)(x)
        x = Dropout(0.25)(x)
        scores = Dense(self.num_classes, activation=None, name='scores_lenet_5', **kwargs)(x)
        predictions = Activation('softmax', name='predictions_lenet_5')(scores)

        model = Model(img_input, predictions, name='cnn_lenet_5')
        scoring_model = Model(img_input, scores, name='cnn_lenet_5_scorer')

        # optimizer = RMSprop(learning_rate=0.00001)
        # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model, scoring_model

    def fit(self, X, y, epochs=200, batch_size=100, verbose=1, **kwargs):
        return super().fit(X, y, epochs, batch_size, verbose, **kwargs)

    def predict_scores(self, X, verbose=0, **kwargs):
        return super().predict_scores(X, verbose, **kwargs)

    def evaluate(self, X, y, verbose=1, **kwargs):
        return super().evaluate(X, y, verbose, **kwargs)

    def summary(self, **kwargs):
        super().summary(**kwargs)
