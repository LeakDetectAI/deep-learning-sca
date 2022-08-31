from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


class AutoencoderCNN(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dimension=700, latent_dimension=625, loss=losses.BinaryCrossentropy(),
                 optimizer='adam', **compile_kwargs):
        self.input_dimension = input_dimension
        self.latent_dimension = latent_dimension
        self.autoencoder_model = self._construct_feature_autoencoder()
        self.encoder_model = None
        self.decoder_model = None
        self.loss = loss
        self.optimizer = optimizer
        self.compile_kwargs = compile_kwargs

    def compile(self):
        self.autoencoder_model.compile(optimizer=self.optimizer, loss=self.loss, **self.compile_kwargs)

    def fit(self, X, y, batch_size=200, verbose=1, epochs=15, validation_data=None,
            save_autoencoder_file_name='deepscapy/trained_models/autoencoder.h5',
            save_encoder_file_name='deepscapy/trained_models/encoder.h5',
            save_decoder_file_name='deepscapy/trained_models/decoder.h5', **kwargs):
        save_autoencoder_model = ModelCheckpoint(save_autoencoder_file_name)
        autoencoder_callbacks = [save_autoencoder_model]

        self.autoencoder_model.fit(x=X, y=y, batch_size=batch_size, verbose=verbose, epochs=epochs,
                                   validation_data=validation_data, callbacks=autoencoder_callbacks, **kwargs)

        # Extract encoder and decoder models separately
        self.encoder_model = Model(self.autoencoder_model.input, self.autoencoder_model.layers[3].output,
                                   name='encoder')

        decoder_inputs = Input(shape=(self.latent_dimension,))
        self.decoder_model = Model(decoder_inputs, self._extract_decoder_output_layers(decoder_inputs), name='decoder')

        # To avoid the Model compile warnings
        self.encoder_model.compile(optimizer=self.optimizer, loss=self.loss, **self.compile_kwargs)
        self.decoder_model.compile(optimizer=self.optimizer, loss=self.loss, **self.compile_kwargs)

        # Save encoder and decoder models separately
        self.encoder_model.save(save_encoder_file_name)
        self.decoder_model.save(save_decoder_file_name)
        return self

    def predict(self, X, batch_size=200, verbose=0, **kwargs):
        predictions = self.encoder_model.predict_scores(x=X, batch_size=batch_size, verbose=verbose, **kwargs)
        return predictions

    def predict_decoder(self, X, batch_size=200, verbose=0, **kwargs):
        predictions_decoder = self.decoder_model.predict_scores(x=X, batch_size=batch_size, verbose=verbose, **kwargs)
        return predictions_decoder

    def predict_autoencoder(self, X, batch_size=200, verbose=0, **kwargs):
        predictions_autoencoder = self.autoencoder_model.predict(x=X, batch_size=batch_size, verbose=verbose, **kwargs)
        return predictions_autoencoder

    def evaluate(self, X, y, batch_size=200, verbose=1, **kwargs):
        model_metrics = self.autoencoder_model.evaluate(x=X, y=y, batch_size=batch_size, verbose=verbose, **kwargs)
        return model_metrics

    def summary(self, **kwargs):
        self.autoencoder_model.summary(**kwargs)
        self.encoder_model.summary(**kwargs)
        self.decoder_model.summary(**kwargs)

    def _extract_decoder_output_layers(self, decoder_inputs):
        decoder_layer_1 = self.autoencoder_model.layers[-3](decoder_inputs)
        decoder_layer_2 = self.autoencoder_model.layers[-2](decoder_layer_1)
        decoder_layer_3 = self.autoencoder_model.layers[-1](decoder_layer_2)
        return decoder_layer_3

    def _construct_feature_autoencoder(self):
        input_shape = (self.input_dimension)
        img_input = Input(shape=input_shape)
        inputs = img_input

        # Encoder
        x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(img_input)
        # kernel_size: Specifying the height and width of the 2D convolution window.

        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(x)
        encoded_output = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

        # Decoder
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_output)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        # Create the complete model, encoder model & decoder model
        autoencoder_model = Model(inputs, decoded_output, name='autoencoder')
        # encoder_model = Model(inputs, encoded_output, name='encoder')
        # decoder_model = Model(encoded_output, decoded_output, name='decoder')

        return autoencoder_model
