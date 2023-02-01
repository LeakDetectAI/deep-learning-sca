from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, Activation, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

from deepscapy.core.sca_nn_model import SCANNModel, ONED_CNN


class CNNCCEDPACV4Baseline(SCANNModel):
    def __init__(self, model_name, num_classes, input_dim, loss_function='categorical_crossentropy',
                 kernel_regularizer=None, kernel_initializer="he_uniform", optimizer=Adam(learning_rate=1e-3),
                 metrics=['accuracy'], weight_averaging=False, **kwargs):
        super(CNNCCEDPACV4Baseline, self).__init__(model_name=model_name, num_classes=num_classes, input_dim=input_dim,
                                                   model_type=ONED_CNN, loss_function=loss_function,
                                                   kernel_regularizer=kernel_regularizer,
                                                   kernel_initializer=kernel_initializer, optimizer=optimizer,
                                                   metrics=metrics, weight_averaging=weight_averaging, **kwargs)

    def _construct_model_(self, **kwargs):
        # Designing input layer
        input_shape = (self.input_dim, 1)
        img_input = Input(shape=input_shape)

        # 1st convolutional block
        x = Conv1D(2, 1, activation='selu', padding='same', name='block1_conv1_dpacv4', **kwargs)(
            img_input)
        x = BatchNormalization(name='block1_batchnorm_dpacv4')(x)
        x = AveragePooling1D(2, strides=2, name='block1_pool_dpacv4')(x)

        x = Flatten(name='flatten_dpacv4')(x)

        # Classification layer
        x = Dense(2, activation='selu', name='fc1_dpacv4', **kwargs)(x)

        # Logits layer
        score_layer = Dense(self.num_classes, activation=None, name='score_dpacv4', **kwargs)(x)
        predictions = Activation('softmax', name='prediction_dpacv4')(score_layer)

        # Create model
        inputs = img_input
        model = Model(inputs, predictions, name='cnn_dpacv4')

        scoring_model = Model(inputs, score_layer, name='cnn_dpacv4_scorer')

        return model, scoring_model

    def fit(self, X, y, verbose=1, **kwargs):
        kwargs['epochs'] = 50
        kwargs['batch_size'] = 50
        return super().fit(X=X, y=y, verbose=verbose, **kwargs)

    def predict_scores(self, X, verbose=0, **kwargs):
        return super().predict_scores(X, verbose, **kwargs)

    def evaluate(self, X, y, verbose=1, **kwargs):
        return super().evaluate(X, y, verbose, **kwargs)

    def summary(self, **kwargs):
        super().summary(**kwargs)
