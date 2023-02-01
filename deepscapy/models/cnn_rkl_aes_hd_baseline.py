from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, Activation, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

from deepscapy.constants import ONED_CNN
from deepscapy.core.sca_nn_model import SCANNModel
from deepscapy.losses.loss_functions import ranking_loss


# Model Code from Ranking Loss Paper Github Repository (https://github.com/gabzai/Ranking-Loss-SCA)
class CNNRankingLossAESHDBaseline(SCANNModel):
    def __init__(self, model_name, num_classes, input_dim, loss_function=ranking_loss(), kernel_regularizer=None,
                 kernel_initializer="he_uniform", optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'],
                 weight_averaging=False, **kwargs):
        super(CNNRankingLossAESHDBaseline, self).__init__(model_name=model_name, num_classes=num_classes,
                                                          input_dim=input_dim, model_type=ONED_CNN, loss_function=loss_function,
                                                          kernel_regularizer=kernel_regularizer,
                                                          kernel_initializer=kernel_initializer, optimizer=optimizer,
                                                          metrics=metrics, weight_averaging=weight_averaging, **kwargs)

    def _construct_model_(self, **kwargs):
        input_shape = (self.input_dim, 1)
        img_input = Input(shape=input_shape)

        # 1st convolutional block
        x = Conv1D(2, 1, activation='selu', padding='same', name='block1_conv1_aes_hd', **kwargs)(
            img_input)
        x = BatchNormalization(name='block1_batchnorm_aes_hd')(x)
        x = AveragePooling1D(2, strides=2, name='block1_pool_aes_hd')(x)

        x = Flatten(name='flatten_aes_hd')(x)

        # Classification layer
        x = Dense(2, activation='selu', name='fc1_aes_hd', **kwargs)(x)

        # Logits layer
        score_layer = Dense(self.num_classes, activation=None, name='score_aes_hd', **kwargs)(x)
        predictions = Activation('softmax', name='prediction_aes_hd')(score_layer)

        # Create model
        inputs = img_input
        model = Model(inputs, predictions, name='cnn_aes_hd')

        scoring_model = Model(inputs=inputs, outputs=score_layer, name='cnn_aes_hd_baseline_scorer')

        return model, scoring_model

    def fit(self, X, y, verbose=1, **kwargs):
        return super().fit(X=X, y=y, epochs=20, batch_size=256, verbose=verbose, **kwargs)

    def predict_scores(self, X, verbose=0, **kwargs):
        return super().predict_scores(X, verbose, **kwargs)

    def evaluate(self, X, y, verbose=1, **kwargs):
        return super().evaluate(X, y, verbose, **kwargs)

    def summary(self, **kwargs):
        super().summary(**kwargs)
