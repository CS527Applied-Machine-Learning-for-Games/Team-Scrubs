import os, math
import numpy as np
import tensorflow as tf

import keras.backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *

from kerassurgeon import Surgeon, identify
from kerassurgeon.operations import delete_channels, delete_layer


def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    # model.compile(optimizer = Adam(lr = 1e-4), metrics = ['accuracy'], loss=weighted_binary_crossentropy(class_weights=[0.25,0.75]))
    
    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def get_filter_weights(model, layer=None):
    if layer or layer==0:
        weight_array = model.layers[layer].get_weights()[0]
    else:
        weights = [model.layers[layer_ix].get_weights()[0] for layer_ix in range(len(model.layers)) \
                   if 'conv' in model.layers[layer_ix].name]
        weight_array = [np.array(i) for i in weights]
    
    return weight_array


def get_filters_l1(model, layer=None):
    if layer or layer==0:
        weights = get_filter_weights(model, layer)
        num_filter = len(weights[0,0,0,:])
        norms_dict = {}
        norms = []
        for i in range(num_filter):
            l1_norm = np.sum(abs(weights[:,:,:,i]))
            norms.append(l1_norm)
    else:
        weights = get_filter_weights(model)
        max_kernels = max([layr.shape[3] for layr in weights])
        norms = np.empty((len(weights), max_kernels))
        norms[:] = np.NaN
        for layer_ix in range(len(weights)):
            # compute norm of the filters
            kernel_size = weights[layer_ix][:,:,:,0].size
            nb_filters = weights[layer_ix].shape[3]
            kernels = weights[layer_ix]
            l1 = [np.sum(abs(kernels[:,:,:,i])) for i in range(nb_filters)]
            # divide by shape of the filters
            l1 = np.array(l1) / kernel_size
            norms[layer_ix, :nb_filters] = l1
    return norms


def compute_pruned_count(model, perc=0.1, layer=None):
    if layer or layer ==0:
        # count nb of filters
        nb_filters = model.layers[layer].output_shape[3]
    else:
        nb_filters = np.sum([model.layers[i].output_shape[3] for i, layer in enumerate(model.layers) 
                                if 'conv' in model.layers[i].name])
            
    n_pruned = int(np.floor(perc*nb_filters))
    return n_pruned


def smallest_indices(array, N):
    idx = array.ravel().argsort()[:N]
    return np.stack(np.unravel_index(idx, array.shape)).T


def prune_one_layer(model, pruned_indexes, layer_ix):
    model_pruned = delete_channels(model, model.layers[layer_ix], pruned_indexes)
    model_pruned.compile(loss='binary_crossentropy',
                          optimizer=Adam(lr = 1e-4),
                          metrics=['accuracy'])
    return model_pruned


def prune_multiple_layers(model, pruned_matrix):
    conv_indexes = [i for i, v in enumerate(model.layers) if 'conv' in v.name]
    layers_to_prune = np.unique(pruned_matrix[:,0])
    surgeon = Surgeon(model, copy=True)
    to_prune = pruned_matrix
    to_prune[:,0] = np.array([conv_indexes[i] for i in to_prune[:,0]])
    layers_to_prune = np.unique(to_prune[:,0])
    for layer_ix in layers_to_prune :
        pruned_filters = [x[1] for x in to_prune if x[0]==layer_ix]
        pruned_layer = model.layers[layer_ix]
        surgeon.add_job('delete_channels', pruned_layer, channels=pruned_filters)
    
    model_pruned = surgeon.operate()
    model_pruned.compile(loss='binary_crossentropy',
              optimizer=Adam(lr = 1e-4),
              metrics=['accuracy'])
    
    return model_pruned


def prune_model(model, perc, layer=None):
    assert perc >=0 and perc <1, "Invalid pruning percentage"
    
    n_pruned = compute_pruned_count(model, perc, layer)
    norms = get_filters_l1(model)
    to_prune = smallest_indices(norms, n_pruned)
    if layer or layer ==0:
        model_pruned = prune_one_layer(model, to_prune, layer)
    else:
        model_pruned = prune_multiple_layers(model, to_prune)
    return model_pruned


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    
    from dataLoader import (
        pretrain_data,
        test_data,
        play_data,
        play_data_by_batch,
        get_data_x,
        play_data_by_batch_x,
    )
    print("Retrieving pre-train data...")
    X, Y = pretrain_data()
    print("Retrieving test data...")
    test_X, test_Y = test_data()
    m = unet()
    print("Train model...")
    H = m.fit(X, Y, batch_size=10, epochs=1, validation_split=0.8)
    m.save("./prune_models/model.h5")
    print("Model saved to prune_models/model.h5")
    print("Evaluating pre-trained model...")
    results = m.evaluate(test_X, test_Y, batch_size=10)
    print("test loss, test acc:", results)
    # print("Loading model...")
    # m = load_model("./prune_models/model.h5")
    print("Start pruning...")
    mp = prune_model(m, 0.2)
    mp.summary()
    print("Train pruned model...")
    H = mp.fit(X, Y, batch_size=10, epochs=1, validation_split=0.8)
    print("Saving pruned model to prune_models/model_prune.h5")
    mp.save("./prune_models/model_prune.h5")
    print("Evaluating pruned model...")
    results = mp.evaluate(test_X, test_Y, batch_size=10)
    print("test loss, test acc:", results)