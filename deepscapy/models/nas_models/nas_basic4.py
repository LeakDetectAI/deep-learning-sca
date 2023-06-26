import autokeras as ak
from keras_tuner.engine import hyperparameters as hp

from deepscapy.constants import NAS_BASIC4
from deepscapy.core import *
from deepscapy.core.extension_blocks import DenseBlockExt


class NASBasic4(NASModel):
    def __init__(self, model_name, num_classes, input_dim, dataset, reshape_type,
                 loss_function='categorical_crossentropy',
                 loss_function_name='categorical_crossentropy', max_trials=100, objective='val_accuracy',
                 overwrite=False, metrics=['accuracy'], tuner='greedy', seed=1234, **auto_model_kwargs):
        super(NASBasic4, self).__init__(model_name=model_name, num_classes=num_classes, input_dim=input_dim,
                                        dataset=dataset, loss_function=loss_function,
                                        loss_function_name=loss_function_name, model_directory=NAS_BASIC4,
                                        reshape_type=reshape_type, max_trials=max_trials, objective=objective,
                                        overwrite=overwrite, metrics=metrics, tuner=tuner, seed=seed,
                                        **auto_model_kwargs)

    def fit(self, X, y, epochs=200, batch_size=100, validation_split=0.1, verbose=1, **kwargs):
        return super().fit(X=X, y=y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                           verbose=verbose, **kwargs)

    def predict_scores(self, X, verbose=0, **kwargs):
        return super().predict_scores(X=X, verbose=verbose, **kwargs)

    def evaluate(self, X, y, verbose=1, **kwargs):
        return super().evaluate(X=X, y=y, verbose=verbose, **kwargs)

    def summary(self, **kwargs):
        super().summary(**kwargs)

    def _construct_model_(self):
        input_node = ImageInputExt(one_dim=self.one_dim)

        # Block 1
        x = ConvBlockExt(num_blocks=hp.Choice(name='num_blocks', values=[1, 2, 3, 4]),
                         num_layers=hp.Choice(name='num_layers', values=[1]),
                         filters=hp.Choice("filters", self.n_filters),
                         kernel_size=hp.Choice("kernel_size", self.kernel_size))(input_node)

        # Classification Block
        x = ak.Flatten()(x)
        x = DenseBlockExt(num_layers=hp.Choice(name='num_layers', values=[1, 2, 3]),
                          num_units=hp.Choice(name='num_units', values=self.n_units),
                          dropout=hp.Choice("dropout", self.dropout, default=0.0))(x)
        output_node = ClassificationHeadFixed(num_classes=self.num_classes, loss=self.loss_function, metrics=self.metrics,
                                              dropout=hp.Choice("dropout", self.dropout, default=0.0))(x)

        auto_model = AutoModelExt(inputs=input_node, outputs=output_node, overwrite=self.overwrite,
                                  max_trials=self.max_trials, objective=self.objective,
                                  directory=self.directory, project_name=self.project_name,
                                  tuner=self.tuner, seed=self.seed, max_model_size=self.max_model_size,
                                  **self.auto_model_kwargs)

        return auto_model
