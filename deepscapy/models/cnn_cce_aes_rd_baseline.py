from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, Activation, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

from deepscapy.constants import ONED_CNN
from deepscapy.core.sca_nn_model import SCANNModel

class CNNCCEAESRDBaseline(SCANNModel):
    def __init__(self, model_name, num_classes, input_dim, loss_function='categorical_crossentropy',
                 kernel_regularizer=None, kernel_initializer="he_uniform", optimizer=Adam(learning_rate=1e-2),
                 metrics=['accuracy'], weight_averaging=False, **kwargs):
        super(CNNCCEAESRDBaseline, self).__init__(model_name=model_name, num_classes=num_classes, input_dim=input_dim,
                                                  model_type=ONED_CNN, loss_function=loss_function,
                                                  kernel_regularizer=kernel_regularizer,
                                                  kernel_initializer=kernel_initializer, optimizer=optimizer,
                                                  metrics=metrics, weight_averaging=weight_averaging, **kwargs)

    def _construct_model_(self, **kwargs):
        # Designing input layer
        input_shape = (self.input_dim, 1)
        img_input = Input(shape=input_shape)

        # 1st convolutional block
        x = Conv1D(8, 1, activation='selu', padding='same', name='block1_conv1_aes_rd', **kwargs)(
            img_input)
        x = BatchNormalization(name='block1_batchnorm_aes_rd')(x)
        x = AveragePooling1D(2, strides=2, name='block1_pool_aes_rd')(x)

        # 2nd convolutional block
        x = Conv1D(16, 50, activation='selu', padding='same', name='block2_conv1_aes_rd', **kwargs)(x)
        x = BatchNormalization(name='block2_batchnorm_aes_rd')(x)
        x = AveragePooling1D(50, strides=50, name='block2_pool_aes_rd')(x)

        # 3rd convolutional block
        x = Conv1D(32, 3, activation='selu', padding='same', name='block3_conv1_aes_rd', **kwargs)(x)
        x = BatchNormalization(name='block3_batchnorm_aes_rd')(x)
        x = AveragePooling1D(7, strides=7, name='block3_pool_aes_rd')(x)

        x = Flatten(name='flatten_aes_rd')(x)

        # Classification layer
        x = Dense(10, activation='selu', name='fc1_aes_rd', **kwargs)(x)
        x = Dense(10, activation='selu', name='fc2_aes_rd', **kwargs)(x)

        # Logits layer
        score_layer = Dense(self.num_classes, activation=None, name='score_aes_rd', **kwargs)(x)
        predictions = Activation('softmax', name='prediction_aes_rd')(score_layer)

        # Create model
        inputs = img_input
        model = Model(inputs, predictions, name='cnn_aes_rd')

        scoring_model = Model(inputs, score_layer, name='cnn_aes_rd_scorer')

        return model, scoring_model

    def fit(self, X, y, verbose=1, **kwargs):
        return super().fit_lr(X=X, y=y, epochs=50, batch_size=50, max_lr=1e-2, verbose=verbose, **kwargs)

    def predict_scores(self, X, verbose=0, **kwargs):
        return super().predict_scores(X, verbose, **kwargs)

    def evaluate(self, X, y, verbose=1, **kwargs):
        return super().evaluate(X, y, verbose, **kwargs)

    def summary(self, **kwargs):
        super().summary(**kwargs)
