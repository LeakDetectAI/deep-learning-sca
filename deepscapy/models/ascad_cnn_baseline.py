from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, Activation
from keras.models import Model
from keras.optimizers import RMSprop

from deepscapy.core.sca_nn_model import SCANNModel
from deepscapy.constants import ONED_CNN


# Model Code from ASCAD Paper Github Repository (https://github.com/ANSSI-FR/ASCAD)
class ASCADCNNBaseline(SCANNModel):
    def __init__(self, model_name, num_classes, input_dim, loss_function='categorical_crossentropy',
                 kernel_regularizer=None, kernel_initializer="glorot_uniform", optimizer=RMSprop(learning_rate=0.00001),
                 metrics=['accuracy'], weight_averaging=False, **kwargs):
        super(ASCADCNNBaseline, self).__init__(model_name=model_name, num_classes=num_classes, input_dim=input_dim,
                                               model_type=ONED_CNN, loss_function=loss_function,
                                               kernel_regularizer=kernel_regularizer,
                                               kernel_initializer=kernel_initializer, optimizer=optimizer,
                                               metrics=metrics, weight_averaging=weight_averaging, **kwargs)

    def _construct_model_(self, **kwargs):
        # From VGG16 design
        input_shape = (self.input_dim, 1)
        trace_input = Input(shape=input_shape, dtype='float32')
        # Block 1
        x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1', **kwargs)(trace_input)
        x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
        # Block 2
        x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1', **kwargs)(x)
        x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
        # Block 3
        x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1', **kwargs)(x)
        x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
        # Block 4
        x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1', **kwargs)(x)
        x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
        # Block 5
        x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1', **kwargs)(x)
        x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1', **kwargs)(x)
        x = Dense(4096, activation='relu', name='fc2', **kwargs)(x)
        scores = Dense(self.num_classes, activation=None, name='scores', kernel_regularizer=self.kernel_regularizer)(x)
        predictions = Activation('softmax', name='predictions')(scores)

        # Create model.
        model = Model(inputs=trace_input, outputs=predictions, name='cnn_baseline')
        # model.compile(loss=self.loss_fn, optimizer=self.optimizer, metrics=self.metrics)
        scoring_model = Model(inputs=trace_input, outputs=scores, name='cnn_baseline_scorer')
        return model, scoring_model

    def fit(self, X, y, epochs=200, batch_size=100, verbose=1, **kwargs):
        return super().fit(X=X, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, **kwargs)

    def predict_scores(self, X, verbose=0, **kwargs):
        return super().predict_scores(X, verbose, **kwargs)

    def evaluate(self, X, y, verbose=1, **kwargs):
        return super().evaluate(X, y, verbose, **kwargs)

    def summary(self, **kwargs):
        super().summary(**kwargs)
