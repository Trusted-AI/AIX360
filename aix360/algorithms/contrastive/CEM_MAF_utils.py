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
import re
import numpy as np
import scipy.misc
import h5py
from contextlib import nullcontext as _nullcontext
from tensorflow.contrib.keras.api.keras.applications.resnet50 import ResNet50
from tensorflow.contrib.keras.api.keras.applications.vgg16 import VGG16
from tensorflow.contrib.keras.api.keras.layers import Input, Dense, Dropout, LeakyReLU, Activation
from tensorflow.contrib.keras.api.keras.models import Model, model_from_json, Sequential
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint
from tensorflow.contrib.keras.api.keras import metrics
from tensorflow.contrib.keras.api.keras import regularizers
from tensorflow.contrib.keras.api.keras.optimizers import SGD


def _legacy_resnet50_layer_remap(old):
    # Keras 2.2.x ResNet50 used res*/bn* naming; Keras 2.3+ uses conv*_block*.
    # Map old layer names to the modern equivalents so weights trained against
    # the legacy architecture can be loaded into the current model.
    m = re.match(r'^res(\d)([a-z])_branch2([abc])$', old)
    if m:
        return 'conv{}_block{}_{}_conv'.format(
            m.group(1), ord(m.group(2)) - ord('a') + 1,
            {'a': 1, 'b': 2, 'c': 3}[m.group(3)])
    m = re.match(r'^res(\d)([a-z])_branch1$', old)
    if m:
        return 'conv{}_block{}_0_conv'.format(
            m.group(1), ord(m.group(2)) - ord('a') + 1)
    m = re.match(r'^bn(\d)([a-z])_branch2([abc])$', old)
    if m:
        return 'conv{}_block{}_{}_bn'.format(
            m.group(1), ord(m.group(2)) - ord('a') + 1,
            {'a': 1, 'b': 2, 'c': 3}[m.group(3)])
    m = re.match(r'^bn(\d)([a-z])_branch1$', old)
    if m:
        return 'conv{}_block{}_0_bn'.format(
            m.group(1), ord(m.group(2)) - ord('a') + 1)
    return {
        'conv1': 'conv1_conv',
        'bn_conv1': 'conv1_bn',
        'fc1000': 'probs',
        'dense_1': 'dense',
        'dense_2': 'dense_1',
    }.get(old)


def _load_celebA_h5_with_remap(model, h5_path):
    # The shipped celebA weights were saved against the Keras 2.2.x ResNet50
    # naming. Match each old layer to the new model by name, then assign each
    # weight slot whose name and shape line up. Old conv layers had bias=True
    # while new ones have bias=False — bias slots without a destination are
    # silently skipped.
    new_layers = {l.name: l for l in model.layers}
    with h5py.File(h5_path, 'r') as f:
        old_names = [n.decode() if isinstance(n, bytes) else n
                     for n in f.attrs['layer_names']]
        for old in old_names:
            new_name = _legacy_resnet50_layer_remap(old)
            if new_name is None or new_name not in new_layers:
                continue
            target = new_layers[new_name]
            weight_names = [w.decode() if isinstance(w, bytes) else w
                            for w in f[old].attrs.get('weight_names', [])]
            old_weights = {wn.split('/')[-1].split(':')[0]: f[old][wn][...]
                           for wn in weight_names}
            new_w = []
            for v in target.weights:
                key = v.name.split('/')[-1].split(':')[0]
                if key in old_weights and old_weights[key].shape == tuple(v.shape.as_list()):
                    new_w.append(old_weights[key])
                else:
                    new_w.append(v.numpy() if hasattr(v, 'numpy')
                                 else tf.keras.backend.get_value(v))
            target.set_weights(new_w)

class CELEBAModel:
    def __init__(self, nn_type="resnet50", restore = None, session=None, use_imagenet_pretrain=False, use_softmax=True, device=None):
        self.image_size = 224
        self.num_channels = 3
        self.num_labels = 8

        # TF 1.15 was built against CUDA 10.0; cuBLAS 10's SGEMM has no kernels
        # for Ampere (sm_80), so ResNet50 forward pass on an A100 fails with
        # "Blas SGEMM launch failed". Pinning this model to CPU sidesteps that
        # while leaving the GAN free to use GPU for its NCHW Conv2D ops.
        device_ctx = tf.device(device) if device else _nullcontext()
        with device_ctx:
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

            if restore:
                print("Load: {}".format(restore))
                _load_celebA_h5_with_remap(model, restore)

        self.model = model

    def predict(self, data):
        # this is used inside tf session, data should be a tensor
        return self.model(data)
