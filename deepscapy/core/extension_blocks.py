import logging
from typing import Optional

import autokeras as ak
import tensorflow as tf
from autokeras import Input, keras_layers, adapters, analysers
from autokeras import hyper_preprocessors as hpps_module
from autokeras import preprocessors
from autokeras.blocks import reduction, ImageBlock
from autokeras.engine import head as head_module
from autokeras.utils import layer_utils, utils
from autokeras.utils import types
from keras import layers, losses, activations
from tensorflow._api.v2 import nest
from tensorflow.python.util import nest
import keras

class ImageInputExt(Input):
    """Input node for image data.

    The input data should be numpy.ndarray or tf.data.Dataset. The shape of the data
    should be (samples, width, height) or
    (samples, width, height, channels).

    # Arguments
        name: String. The name of the input node. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(self, name: Optional[str] = None, one_dim=False, **kwargs):
        self.one_dim = one_dim
        self.logger = logging.getLogger(ImageInputExt.__name__)

        super().__init__(name=name, **kwargs)

    def build(self, hp, inputs=None):
        inputs = super().build(hp, inputs)
        output_node = nest.flatten(inputs)[0]
        self.logger.info("Input Shape {} One-Dimensional {}".format(output_node.shape, self.one_dim))
        if len(output_node.shape) == 3 and not self.one_dim:
            output_node = keras_layers.ExpandLastDim()(output_node)
        return output_node

    def get_adapter(self):
        return adapters.ImageAdapter()

    def get_analyser(self):
        return analysers.ImageAnalyser()

    def get_block(self):
        return ImageBlock()


def get_average_pooling(shape):
    return [
        layers.AveragePooling1D,
        layers.AveragePooling2D,
        layers.AveragePooling3D,
    ][len(shape) - 3]


def get_max_pooling(shape):
    return [
        layers.MaxPooling1D,
        layers.MaxPooling2D,
        layers.MaxPooling3D,
    ][len(shape) - 3]


class ConvBlockExt(ak.ConvBlock):

    def build(self, hp, inputs=None):
        self.logger = logging.getLogger(ConvBlockExt.__name__)
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node
        conv = layer_utils.get_conv(input_node.shape)
        use_batch_norm = hp.Boolean("use_batch_norm", default=True)
        max_pooling = hp.Boolean("max_pooling", default=True)
        if max_pooling:
            pool = get_max_pooling(input_node.shape)
            pool_name = 'maxpPooling_'
        else:
            pool = get_average_pooling(input_node.shape)
            pool_name = 'averagePooling_'

        self.dropout = hp.Choice("dropout", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], default=0.0)
        self.logger.info("Batch Norm {}, Pooling Name {}, dropout {}".format(use_batch_norm, pool_name, self.dropout))
        activation = hp.Choice('activation', values=["relu", "selu", "elu"])
        if isinstance(conv, layers.Conv1D):
            strides_hp = hp.Choice('strides', [2, 3, 4, 5, 6, 7, 8, 9, 10], default=2)
            pool_size_hp = hp.Choice('pool_size', [2, 3, 4, 5], default=2)

            for i in range(utils.add_to_hp(self.num_blocks, hp)):
                for j in range(utils.add_to_hp(self.num_layers, hp)):
                    kernel_size = utils.add_to_hp(self.kernel_size, hp)
                    conv_padding = 'same'
                    filters = utils.add_to_hp(self.filters, hp, "filters_{i}_{j}".format(i=i, j=j)),
                    output_node = conv(filters, kernel_size, padding=conv_padding, activation=activation,
                                       kernel_initializer="he_uniform")(output_node)
                    if use_batch_norm:
                        output_node = layers.BatchNormalization()(output_node)

                    name = "{}{}_{}_size".format(pool_name, str(i + 1), str(j + 1))
                    pool_size = utils.add_to_hp(pool_size_hp, hp, name)
                    name = "{}{}_{}_stride".format(pool_name, str(i + 1), str(j + 1))
                    strides = utils.add_to_hp(strides_hp, hp, name)
                    pool_padding = "valid"
                    self.logger.info("For Layer {}, For block {} Kernel Size {}".format(i + 1, j + 1, kernel_size))
                    self.logger.info("Pool size {}, Strides {}, Padding {}".format(pool_size, strides, pool_padding))
                    output_node = pool(pool_size=pool_size, strides=strides, padding=pool_padding)(output_node)
                if utils.add_to_hp(self.dropout, hp) > 0:
                    output_node = layers.Dropout(utils.add_to_hp(self.dropout, hp))(output_node)
        else:
            kernel_size = utils.add_to_hp(self.kernel_size, hp)
            strides_hp = hp.Choice('strides', [2, 4], default=2)
            for i in range(utils.add_to_hp(self.num_blocks, hp)):
                for j in range(utils.add_to_hp(self.num_layers, hp)):
                    output_node = conv(
                        utils.add_to_hp(self.filters, hp, "filters_{i}_{j}".format(i=i, j=j)),
                        kernel_size,
                        padding=self._get_padding(kernel_size, output_node),
                        activation=activation,
                        kernel_initializer="he_uniform"
                    )(output_node)
                    if use_batch_norm:
                        output_node = layers.BatchNormalization()(output_node)
                    pool_size = kernel_size - 1
                    name = "{}{}_{}_stride".format(pool_name, str(i + 1), str(j + 1))
                    strides = utils.add_to_hp(strides_hp, hp, name)
                    padding = self._get_padding(kernel_size - 1, output_node)
                    self.logger.info("For Layer {}, For block {} Kernel Size {}".format(i + 1, j + 1, kernel_size))
                    self.logger.info("Pool size {}, Strides {}, Padding {}".format(pool_size, strides, padding))
                    output_node = pool(pool_size=pool_size, strides=strides, padding=padding)(output_node)
                if utils.add_to_hp(self.dropout, hp) > 0:
                    output_node = layers.Dropout(utils.add_to_hp(self.dropout, hp))(output_node)
            return output_node

        return output_node


class DenseBlockExt(ak.DenseBlock):
    def build(self, hp, inputs=None):
        activation = hp.Choice('activation', values=["relu", "selu", "elu"])
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node
        output_node = reduction.Flatten().build(hp, output_node)

        use_batchnorm = self.use_batchnorm
        if use_batchnorm is None:
            use_batchnorm = hp.Boolean("use_batchnorm", default=False)

        for i in range(utils.add_to_hp(self.num_layers, hp)):
            units = utils.add_to_hp(self.num_units, hp, "units_{i}".format(i=i))
            output_node = layers.Dense(units, kernel_initializer="he_uniform", activation=activation)(output_node)
            if use_batchnorm:
                output_node = layers.BatchNormalization()(output_node)
            if utils.add_to_hp(self.dropout, hp) > 0:
                output_node = layers.Dropout(utils.add_to_hp(self.dropout, hp))(output_node)
        return output_node

class SinglePoolBlock(ak.Block):
    def build(self, hp, inputs=None):
        # Get the input_node from inputs.
        input_node = inputs[0]
        output_node = layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), name='block1_pool')(input_node)
        return output_node


class ClassificationHeadExt(head_module.Head):
    """Classification Dense layers.

    Use sigmoid and binary crossentropy for binary classification and multi-label
    classification. Use softmax and categorical crossentropy for multi-class
    (more than 2) classification. Use Accuracy as metrics by default.

    The targets passing to the head would have to be tf.data.Dataset, np.ndarray,
    pd.DataFrame or pd.Series. It can be raw labels, one-hot encoded if more than two
    classes, or binary encoded for binary classification.

    The raw labels will be encoded to one column if two classes were found,
    or one-hot encoded if more than two classes were found.

    # Arguments
        num_classes: Int. Defaults to None. If None, it will be inferred from the
            data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use `binary_crossentropy` or
            `categorical_crossentropy` based on the number of classes.
        metrics: A list of Keras metrics. Defaults to use 'accuracy'.
        dropout: Float. The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(
            self,
            num_classes: Optional[int] = None,
            multi_label: bool = False,
            loss: Optional[types.LossType] = None,
            metrics: Optional[types.MetricsType] = None,
            dropout: Optional[float] = None,
            **kwargs
    ):
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.dropout = dropout
        if metrics is None:
            metrics = ["accuracy"]
        if loss is None:
            loss = self.infer_loss()
        super().__init__(loss=loss, metrics=metrics, **kwargs)
        # Infered from analyser.
        self._encoded = None
        self._encoded_for_sigmoid = None
        self._encoded_for_softmax = None
        self._add_one_dimension = False
        self._labels = None

    def infer_loss(self):
        if not self.num_classes:
            return None
        if self.num_classes == 2 or self.multi_label:
            return losses.BinaryCrossentropy()
        return losses.CategoricalCrossentropy()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "multi_label": self.multi_label,
                "dropout": self.dropout,
            }
        )
        return config

    def build(self, hp, inputs=None):
        self.logger = logging.getLogger(ClassificationHeadExt.__name__)
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        # Reduce the tensor to a vector.
        if len(output_node.shape) > 2:
            output_node = reduction.SpatialReduction().build(hp, output_node)

        self.dropout = hp.Choice("dropout", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], default=0.0)
        self.logger.info("Dropout {}".format(self.dropout))
        if utils.add_to_hp(self.dropout, hp) > 0:
            output_node = layers.Dropout(self.dropout)(output_node)
        output_node = layers.Dense(self.shape[-1])(output_node)
        self.logger.info(f"Loss function {self.loss}")
        if isinstance(self.loss, losses.BinaryCrossentropy):
            output_node = layers.Activation(activations.sigmoid, name=self.name)(output_node)
        elif 'ranking_loss' in str(self.loss):
            output_node = layers.Activation(activations.linear, name=self.name)(output_node)
        else:
            output_node = layers.Softmax(name=self.name)(output_node)

        return output_node

    def get_adapter(self):
        return adapters.ClassificationAdapter(name=self.name)

    def get_analyser(self):
        return analysers.ClassificationAnalyser(
            name=self.name, multi_label=self.multi_label
        )

    def config_from_analyser(self, analyser):
        super().config_from_analyser(analyser)
        self.num_classes = analyser.num_classes
        self.loss = self.loss
        self._encoded = analyser.encoded
        self._encoded_for_sigmoid = analyser.encoded_for_sigmoid
        self._encoded_for_softmax = analyser.encoded_for_softmax
        self._add_one_dimension = len(analyser.shape) == 1
        self._labels = analyser.labels

    def get_hyper_preprocessors(self):
        hyper_preprocessors = []

        if self._add_one_dimension:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.AddOneDimension())
            )

        if self.dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.CastToInt32())
            )

        if not self._encoded and self.dtype != tf.string:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.CastToString())
            )

        if self._encoded_for_sigmoid:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.SigmoidPostprocessor()
                )
            )
        elif self._encoded_for_softmax:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.SoftmaxPostprocessor()
                )
            )
        elif self.num_classes == 2:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.LabelEncoder(self._labels)
                )
            )
        else:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.OneHotEncoder(self._labels)
                )
            )
        return hyper_preprocessors



class ClassificationHead(head_module.Head):
    """Classification Dense layers.

    Use sigmoid and binary crossentropy for binary classification and multi-label
    classification. Use softmax and categorical crossentropy for multi-class
    (more than 2) classification. Use Accuracy as metrics by default.

    The targets passing to the head would have to be tf.data.Dataset, np.ndarray,
    pd.DataFrame or pd.Series. It can be raw labels, one-hot encoded if more than two
    classes, or binary encoded for binary classification.

    The raw labels will be encoded to one column if two classes were found,
    or one-hot encoded if more than two classes were found.

    # Arguments
        num_classes: Int. Defaults to None. If None, it will be inferred from the
            data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use `binary_crossentropy` or
            `categorical_crossentropy` based on the number of classes.
        metrics: A list of Keras metrics. Defaults to use 'accuracy'.
        dropout: Float. The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        multi_label: bool = False,
        loss: Optional[types.LossType] = None,
        metrics: Optional[types.MetricsType] = None,
        dropout: Optional[float] = None,
        **kwargs
    ):
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.dropout = dropout
        if metrics is None:
            metrics = ["accuracy"]
        if loss is None:
            loss = self.infer_loss()
        super().__init__(loss=loss, metrics=metrics, **kwargs)
        #print(f"super {self.loss}")
        # Infered from analyser.
        self._encoded = None
        self._encoded_for_sigmoid = None
        self._encoded_for_softmax = None
        self._add_one_dimension = False
        self._labels = None
        #print(f"self._labels {self.loss}")

    def infer_loss(self):
        if not self.num_classes:
            return None
        if self.num_classes == 2 or self.multi_label:
            return losses.BinaryCrossentropy()
        return losses.CategoricalCrossentropy()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "multi_label": self.multi_label,
                "dropout": self.dropout,
            }
        )
        return config

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        # Reduce the tensor to a vector.
        if len(output_node.shape) > 2:
            output_node = reduction.SpatialReduction().build(hp, output_node)

        if self.dropout is not None:
            dropout = self.dropout
        else:
            dropout = hp.Choice("dropout", [0.0, 0.25, 0.5], default=0)
        print(f"build {self.loss}")
        if utils.add_to_hp(self.dropout, hp) > 0:
            output_node = layers.Dropout(dropout)(output_node)
        output_node = layers.Dense(self.shape[-1])(output_node)
        if isinstance(self.loss, keras.losses.BinaryCrossentropy):
            output_node = layers.Activation(activations.sigmoid, name=self.name)(
                output_node
            )
        else:
            output_node = layers.Softmax(name=self.name)(output_node)
        return output_node

    def get_adapter(self):
        return adapters.ClassificationAdapter(name=self.name)

    def get_analyser(self):
        return analysers.ClassificationAnalyser(
            name=self.name, multi_label=self.multi_label
        )

    def config_from_analyser(self, analyser):
        super().config_from_analyser(analyser)
        self.num_classes = analyser.num_classes
        #print(f"before infer {self.loss}")
        self.loss = self.infer_loss()
        #print(f"after infer {self.loss}")
        self._encoded = analyser.encoded
        self._encoded_for_sigmoid = analyser.encoded_for_sigmoid
        self._encoded_for_softmax = analyser.encoded_for_softmax
        self._add_one_dimension = len(analyser.shape) == 1
        self._labels = analyser.labels

    def get_hyper_preprocessors(self):
        hyper_preprocessors = []

        if self._add_one_dimension:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.AddOneDimension())
            )

        if self.dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.CastToInt32())
            )

        if not self._encoded and self.dtype != tf.string:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.CastToString())
            )

        if self._encoded_for_sigmoid:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.SigmoidPostprocessor()
                )
            )
        elif self._encoded_for_softmax:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.SoftmaxPostprocessor()
                )
            )
        elif self.num_classes == 2:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.LabelEncoder(self._labels)
                )
            )
        else:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.OneHotEncoder(self._labels)
                )
            )
        return hyper_preprocessors