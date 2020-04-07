## Utils.py -- Some utility functions
##
## Copyright (C) 2018, PaiShun Ting <paishun@umich.edu>
##                     Chun-Chen Tu <timtu@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the "supplementary license" folder present in the root directory.
##
## Modifications Copyright (c) 2019 IBM Corporation


import tensorflow as tf
import os
import numpy as np
import scipy.misc
from tensorflow.contrib.keras.api.keras.applications.resnet50 import ResNet50
from tensorflow.contrib.keras.api.keras.applications.vgg16 import VGG16
from tensorflow.contrib.keras.api.keras.layers import Input, Dense, Dropout, LeakyReLU, Activation
from tensorflow.contrib.keras.api.keras.models import Model, model_from_json, Sequential
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint
from tensorflow.contrib.keras.api.keras import metrics
from tensorflow.contrib.keras.api.keras import regularizers
from tensorflow.contrib.keras.api.keras.optimizers import SGD

def load_AE(codec_prefix, print_summary=False):

    saveFilePrefix = "models/AE_codec/" + codec_prefix + "_"

    decoder_model_filename = saveFilePrefix + "decoder.json"
    decoder_weight_filename = saveFilePrefix + "decoder.h5"

    if not os.path.isfile(decoder_model_filename):
        raise Exception("The file for decoder model does not exist:{}".format(decoder_model_filename))
    json_file = open(decoder_model_filename, 'r')
    decoder = model_from_json(json_file.read(), custom_objects={"tf": tf})
    json_file.close()

    if not os.path.isfile(decoder_weight_filename):
        raise Exception("The file for decoder weights does not exist:{}".format(decoder_weight_filename))
    decoder.load_weights(decoder_weight_filename)

    if print_summary:
        print("Decoder summaries")
        decoder.summary()

    return decoder

class CELEBAModel:
    def __init__(self, nn_type="resnet50", restore = None, session=None, use_imagenet_pretrain=False, use_softmax=True):
        self.image_size = 224
        self.num_channels = 3
        self.num_labels = 8

        input_layer = Input(shape=(self.image_size, self.image_size, self.num_channels))
        weights = "imagenet" if use_imagenet_pretrain else None
        if nn_type == "resnet50":
            base_model = ResNet50(weights=weights, input_tensor=input_layer)
        elif nn_type == "vgg16":
            base_model = VGG16(weights=weights, input_tensor=input_layer)
            # base_model = VGG16(weights=None, input_tensor=input_layer)
        x = base_model.output
        x = LeakyReLU()(x)
        x = Dense(1024)(x)
        x = Dropout(0.2)(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)
        x = Dense(8)(x)
        if use_softmax:
            x = Activation("softmax")(x)
        model = Model(inputs=base_model.input, outputs=x)

        # for layer in base_model.layers:
        # 	layer.trainable = False


        if restore:
            print("Load: {}".format(restore))
            model.load_weights(restore)

        self.model = model

    def predict(self, data):
        # this is used inside tf session, data should be a tensor
        return self.model(data)
