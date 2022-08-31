import autokeras as ak
import tensorflow as tf
from autokeras import graph as graph_module
from autokeras import keras_layers
from autokeras.engine import head as head_module
from autokeras.engine import node as node_module
from autokeras import ImageClassifier

class AutoModelExt(ak.AutoModel):

    def _build_graph(self):
        # Using functional API.
        if all([isinstance(output, node_module.Node) for output in self.outputs]):
            graph = GraphExt(inputs=self.inputs, outputs=self.outputs)
        # Using input/output API.
        elif all([isinstance(output, head_module.Head) for output in self.outputs]):
            graph = self._assemble()
            self.outputs = graph.outputs

        return graph


class GraphExt(graph_module.Graph):
    def _compile_keras_model(self, hp, model):
        # Specify hyperparameters from compile(...)
        optimizer_name = hp.Choice(
            "optimizer",
            ["adam", "adam_weight_decay"],
            default="adam",
        )
        learning_rate = hp.Choice("learning_rate", [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5], default=1e-3)
        if optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == "adam_weight_decay":
            steps_per_epoch = int(self.num_samples / self.batch_size)
            num_train_steps = steps_per_epoch * self.epochs
            warmup_steps = int(
                self.epochs * self.num_samples * 0.1 / self.batch_size
            )

            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=num_train_steps,
                end_learning_rate=0.0,
            )
            if warmup_steps:
                lr_schedule = keras_layers.WarmUp(
                    initial_learning_rate=learning_rate,
                    decay_schedule_fn=lr_schedule,
                    warmup_steps=warmup_steps,
                )

            optimizer = keras_layers.AdamWeightDecay(
                learning_rate=lr_schedule,
                weight_decay_rate=0.01,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-6,
                exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            )

        model.compile(
            optimizer=optimizer, metrics=self._get_metrics(), loss=self._get_loss()
        )
        return model
