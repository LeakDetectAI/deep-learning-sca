from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras.optimizers import RMSprop

from deepscapy.constants import MLP
from deepscapy.core.sca_nn_model import SCANNModel


# Model Code from ASCAD Paper Github Repository (https://github.com/ANSSI-FR/ASCAD)
class ASCADMLPBaseline(SCANNModel):
    def __init__(self, model_name, num_classes, input_dim, n_units=200, n_hidden=5,
                 loss_function='categorical_crossentropy', kernel_regularizer=None, kernel_initializer="glorot_uniform",
                 optimizer=RMSprop(learning_rate=0.00001), metrics=['accuracy'], weight_averaging=False, **kwargs):
        self.n_units = n_units
        self.n_hidden = n_hidden
        super(ASCADMLPBaseline, self).__init__(model_name=model_name, num_classes=num_classes, input_dim=input_dim,
                                               model_type=MLP, loss_function=loss_function,
                                               kernel_regularizer=kernel_regularizer,
                                               kernel_initializer=kernel_initializer, optimizer=optimizer,
                                               metrics=metrics, weight_averaging=weight_averaging, **kwargs)

    def _construct_model_(self, **kwargs):
        self.input = Input(shape=self.input_dim, dtype='float32')
        self.hidden_layers = [Dense(self.n_units, name="hidden_{}".format(x), activation='relu', **kwargs) for x in
                              range(self.n_hidden)]
        self.score_layer = Dense(self.num_classes, activation=None, kernel_regularizer=self.kernel_regularizer)
        self.output_node = Activation('softmax', name='predictions')
        x = self.hidden_layers[0](self.input)
        for hidden in self.hidden_layers[1:]:
            x = hidden(x)
            # x = BatchNormalization()(x)
        scores = self.score_layer(x)
        output = self.output_node(scores)
        model = Model(inputs=self.input, outputs=output, name="mlp_baseline")
        scoring_model = Model(inputs=self.input, outputs=scores, name="mlp_baseline_scorer")
        # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model, scoring_model

    def fit(self, X, y, epochs=200, batch_size=100, verbose=1, **kwargs):
        return super().fit(X=X, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, **kwargs)

    def predict_scores(self, X, verbose=0, **kwargs):
        return super().predict_scores(X, verbose, **kwargs)

    def evaluate(self, X, y, verbose=1, **kwargs):
        return super().evaluate(X, y, verbose, **kwargs)

    def summary(self, **kwargs):
        super().summary(**kwargs)
