#!/usr/bin/env python
# # -*- coding: utf-8 -*-

from typing import List, Tuple
import keras
import keras_tuner
import math

# from keras import backend as K

class MultiSpectral(keras.layers.Layer):
    def __init__(self, spectrograms, output_size: Tuple[int, int], **kwargs):
        super(MultiSpectral, self).__init__(**kwargs)
        self.spectrograms = spectrograms
        self.output_shape = output_size
        self.pipelines = [
            keras.layers.Pipeline([
                spectrogram,
                keras.layers.Lambda(lambda x: keras.ops.expand_dims(x, axis=-1)), # Add an artificial color channel
                keras.layers.Resizing(
                    width=output_size[0],
                    height=output_size[1],
                    interpolation="bilinear",
                    data_format="channels_last"
                )
            ])
            for spectrogram in spectrograms
        ]
    def call(self, inputs, training=None):
        xs = [pipe(inputs, training=training) for pipe in self.pipelines]
        return keras.ops.concatenate(xs, axis=-1)
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + self.output_shape + (len(self.spectrograms), )
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "spectrograms": [
                spectrogram.get_config()
                for spectrogram in self.spectrograms
            ],
            "output_size": self.output_shape
        })
        return config
    @classmethod
    def from_config(cls, config):
        spectrogram_configs = config.pop("spectrograms")
        spectrograms = [keras.saving.deserialize_keras_object(config) for config in spectrogram_configs]
        
        output_shape = config.pop("output_size")
        return cls(spectrograms, output_shape, *config)
    def build(self, input_shape):
        for pipeline in self.pipelines:
            pipeline.build(input_shape)
    
# class SimplifiedSTFT(keras.layers.Layer):
#     def __init__(self, *args, **kwargs):
#         super(SimplifiedSTFT, self).__init__(*args, **kwargs)
#         self.stft = keras.layers.STFTSpectrogram(*args, **kwargs)
#     def call(self, inputs, training=None):
#         x = keras.ops.expand_dims(inputs, -1)
#         x = self.stft(x)
#         return x
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "stft": self.stft.get_config()
#         })
#         return config
#     @classmethod
#     def from_config(cls, config):
#         stft_config = config.pop("stft")
#         stft = keras.saving.deserialize_keras_object(stft_config)
#         layer = cls(*config)
#         layer.stft = stft
#         return layer
def SimplifiedSTFT(*args, **kwargs):
    return keras.layers.Pipeline([
        keras.layers.Lambda(lambda x: keras.ops.expand_dims(x, axis=-1)),
        keras.layers.STFTSpectrogram(*args, **kwargs)
    ])