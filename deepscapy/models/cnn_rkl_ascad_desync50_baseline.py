from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, Activation, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

from deepscapy.constants import ONED_CNN
from deepscapy.core.sca_nn_model import SCANNModel
from deepscapy.losses.loss_functions import ranking_loss


# Model Code from Ranking Loss Paper Github Repository (https://github.com/gabzai/Ranking-Loss-SCA)
class CNNRankingLossASCADDesync50Baseline(SCANNModel):
    def __init__(self, model_name, num_classes, input_dim, loss_function=ranking_loss(), kernel_regularizer=None,
                 kernel_initializer="he_uniform", optimizer=Adam(learning_rate=5e-3), metrics=['accuracy'],
                 weight_averaging=False, **kwargs):
        super(CNNRankingLossASCADDesync50Baseline, self).__init__(model_name=model_name, num_classes=num_classes,
                                                                  input_dim=input_dim, model_type=ONED_CNN,
                                                                  loss_function=loss_function,
                                                                  kernel_regularizer=kernel_regularizer,
                                                                  kernel_initializer=kernel_initializer,
                                                                  optimizer=optimizer, metrics=metrics,
                                                                  weight_averaging=weight_averaging, **kwargs)

    def _construct_model_(self, **kwargs):
        input_shape = (self.input_dim, 1)
        img_input = Input(shape=input_shape)

        # 1st convolutional block
        x = Conv1D(32, 1, activation='selu', padding='same', name='block1_conv1_ascad_desync50', **kwargs)(
            img_input)
        x = BatchNormalization(name='block1_batchnorm_ascad_desync50')(x)
        x = AveragePooling1D(2, strides=2, name='block1_pool_ascad_desync50')(x)

        # 2nd convolutional block
        x = Conv1D(64, 25, activation='selu', padding='same', name='block2_conv1_ascad_desync50', **kwargs)(x)
        x = BatchNormalization(name='block2_batchnorm_ascad_desync50')(x)
        x = AveragePooling1D(25, strides=25, name='block2_pool_ascad_desync50')(x)

        # 3rd convolutional block
        x = Conv1D(128, 3, activation='selu', padding='same', name='block3_conv1_ascad_desync50', **kwargs)(x)
        x = BatchNormalization(name='block3_batchnorm_ascad_desync50')(x)
        x = AveragePooling1D(4, strides=4, name='block3_pool_ascad_desync50')(x)

        x = Flatten(name='flatten_ascad_desync50')(x)

        # Classification part
        x = Dense(15, activation='selu', name='fc1_ascad_desync50', **kwargs)(x)
        x = Dense(15, activation='selu', name='fc2_ascad_desync50', **kwargs)(x)
        x = Dense(15, activation='selu', name='fc3_ascad_desync50', **kwargs)(x)

        # Logits layer
        score_layer = Dense(self.num_classes, activation=None, name='score_ascad_desync50')(x)
        predictions = Activation('softmax', name='prediction_ascad_desync50')(score_layer)

        # Create model
        inputs = img_input
        model = Model(inputs, predictions, name='cnn_ascad_desync50')

        scoring_model = Model(inputs=inputs, outputs=score_layer, name='cnn_ascad_desync50_baseline_scorer')

        return model, scoring_model

    def fit(self, X, y, verbose=1, **kwargs):
        kwargs['epochs'] = 50
        kwargs['batch_size'] = 256
        return super().fit_lr(X=X, y=y, verbose=verbose, max_lr=5e-3, **kwargs)

    def predict_scores(self, X, verbose=0, **kwargs):
        return super().predict_scores(X, verbose, **kwargs)

    def evaluate(self, X, y, verbose=1, **kwargs):
        return super().evaluate(X, y, verbose, **kwargs)

    def summary(self, **kwargs):
        super().summary(**kwargs)
