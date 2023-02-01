from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, Activation, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

from deepscapy.constants import ONED_CNN
from deepscapy.core.sca_nn_model import SCANNModel
from deepscapy.losses.loss_functions import ranking_loss


# Model Code from Ranking Loss Paper Github Repository (https://github.com/gabzai/Ranking-Loss-SCA)
class CNNRankingLossASCADDesync0Baseline(SCANNModel):
    def __init__(self, model_name, num_classes, input_dim, loss_function=ranking_loss(),
                 kernel_regularizer=None, kernel_initializer="he_uniform", optimizer=Adam(learning_rate=5e-3),
                 metrics=['accuracy'], weight_averaging=False, **kwargs):
        super(CNNRankingLossASCADDesync0Baseline, self).__init__(model_name=model_name, num_classes=num_classes,
                                                                 input_dim=input_dim, model_type=ONED_CNN,
                                                                 loss_function=loss_function,
                                                                 kernel_regularizer=kernel_regularizer,
                                                                 kernel_initializer=kernel_initializer,
                                                                 optimizer=optimizer, weight_averaging=weight_averaging,
                                                                 metrics=metrics, **kwargs)

    def _construct_model_(self, **kwargs):
        input_shape = (self.input_dim, 1)
        img_input = Input(shape=input_shape)

        # 1st convolutional block
        x = Conv1D(4, 1, activation='selu', padding='same', name='block1_conv1_ascad_desync0', **kwargs)(
            img_input)
        x = BatchNormalization(name='block1_batchnorm_ascad_desync0')(x)
        x = AveragePooling1D(2, strides=2, name='block1_pool_ascad_desync0')(x)

        x = Flatten(name='flatten_ascad_desync0')(x)

        # Classification layer
        x = Dense(10, activation='selu', name='fc1_ascad_desync0', **kwargs)(x)
        x = Dense(10, activation='selu', name='fc2_ascad_desync0', **kwargs)(x)

        # Logits layer
        score_layer = Dense(self.num_classes, activation=None, name='score_ascad_desync0', **kwargs)(x)
        predictions = Activation('softmax', name='prediction_ascad_desync0')(score_layer)

        # Create model
        inputs = img_input
        model = Model(inputs, predictions, name='cnn_ascad_desync0')

        scoring_model = Model(inputs=inputs, outputs=score_layer, name='cnn_ascad_desync0_baseline_scorer')

        return model, scoring_model

    def fit(self, X, y, verbose=1, **kwargs):
        return super().fit_lr(X=X, y=y, epochs=50, batch_size=50, verbose=verbose, max_lr=5e-3, **kwargs)

    def predict_scores(self, X, verbose=0, **kwargs):
        return super().predict_scores(X, verbose, **kwargs)

    def evaluate(self, X, y, verbose=1, **kwargs):
        return super().evaluate(X, y, verbose, **kwargs)

    def summary(self, **kwargs):
        super().summary(**kwargs)
