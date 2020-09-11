import numpy as np
import os
import sys
import cv2
import h5py

from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import plot_model
import tensorflow as tf
from numpy import inf
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
import SimpleITK as sitk

# Change to your own directory
sys.path.append('../')

#from unet_train_2Dfor3D_mc_Encoding_beta_cell_4label import *

from utils.kerasutils import get_image, correct_data_format, save_model_summary, get_channel_axis
from utils.imageutils import map_label
from utils.input_data_mc import InputSegmentationArrays
from keras.models import load_model

import pandas as pd
from nets.unet_HW import build_net

modelPath = '/home/jpfrancis/models/mixlabel_z_12_19_19_model.h5'

image_weight = 0.0005
nlabel=5
ncluster=(np.zeros(nlabel,np.int) + 1)
unlabeled_IDs = pd.read_csv('/home/jpfrancis/3D/test_IDs', delimiter=' ')['ID'].values
predict_IDs = unlabeled_IDs

cc_weight=0
#image_weight=.0005
# interval=int(sys.argv[3])
output_folder= '/home/jpfrancis/3D/layer_outputs/'
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[0]
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

model_args = dict(
    # num_classes=input_arrays.get_num_classes(),
    num_classes=nlabel,
    base_num_filters=32,
    image_size=(512, 512),
    # image_size=(1024, 1024),
    dropout_rate=0.5,
    optimizer=Adam(lr=5e-5),
    conv_order='conv_first',
    kernel_size=3,
    kernel_cc_weight=0.0,
    activation='relu',
    net_depth=5,
    convs_per_depth=2,
    noise_std=1.0,
    ncluster=ncluster,
    # loss=combine_loss([EncodingLength_2D(image_weight=image_weight, L=nlabel), exp_categorical_crossentropy_Im_mc(exp=1.0, class_weights=class_weights,nclass=nlabel, ncluster=ncluster)], [0.5, 0.5])
    # loss=combine_loss([VoI_2D(image_weight=image_weight, sigma=sigma), exp_categorical_crossentropy_Im_mc(exp=exp, class_weights=class_weights,nclass=nclass, ncluster=ncluster)], [1.0-alpha, alpha])
)

net = build_net(**model_args)
AE_model = Model(inputs=net.input, outputs=net.get_layer('segmentation').output)
AE_model.load_weights(modelPath)
print(net.summary())

def transfer_image(img):
    # max_v=np.max(img)
    # min_v=np.min(img)
    #
    # new_img=(img-min_v)*255/(max_v-min_v+1E-10)
    # return new_img
    #img[img==3]=1
    #img[img==4]=3
    return img * 10000

def generator_prediction(id):
    prefix = '/home/jpfrancis/3D/Unlabeled_all/'+id+"/z/"+id
    flag=1
    length=0
    while flag>0:
        fn=prefix+'_2D_z_'+str(length)+'.nii.gz'
        # print(fn)
        if os.path.isfile(fn):
            length += 1
        else:
            flag=0

        # print(length)
        # kk
    imgs = np.zeros((length, 512, 512))
    c = 0
    for i in range(0,length):
        imfn = prefix+'_2D_z_'+str(i)+'.nii.gz'

        img = transfer_image(sitk.GetArrayFromImage(sitk.ReadImage(imfn)))

        imgs[c,:,:]=img
        # probs[c,:,:,:]=prob
        c=c+1

    #imgs_probs = to_categorical(imgs,4)
    imgs = np.expand_dims(imgs, axis=-1)
    print(imgs.shape)
    return imgs

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n),y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def get_segmentation(image, nclass, ncluster):
    seg_mc = AE_model.predict(image).argmax(get_channel_axis())
    seg = seg_mc.copy()
    ind=nclass
    for i in range(nclass):
        for j in range(ncluster[i]-1):
            seg[seg==ind] = i
            ind=ind+1
    return seg, seg_mc

def get_pd_mask(image, nclass, ncluster):
    seg_mask_mc = AE_model.predict(image)
    seg_mask = seg_mask_mc.copy()
    for i in range(nclass):
        for j in range(ncluster[i]-1):
            seg[seg==ind] = i
            ind = ind+1

    return seg_mask, seg_mask_mc

for i in range(len(predict_IDs)):
    with h5py.File(modelPath, "r") as f:
        a_group_key = list(f.keys())[0]
        layer_names = list(f[a_group_key])

    id=predict_IDs[i]
    print(id)

    imgs = generator_prediction(id)

    sz = imgs.shape

    #aseg = np.zeros((sz[0], sz[1], sz[2]), np.int32)
    aseg = np.zeros((sz[0], sz[1], sz[2], 5))

    #seg, seg_mc = get_segmentation(np.expand_dims(imgs_probs[j, :, :, :], axis=0),nlabel,ncluster)
    for layer_name in layer_names:
        if layer_name != 'input_1':
            AE_model = Model(inputs=net.input, outputs=net.get_layer(layer_name).output)
            layer_output = AE_model.predict(np.expand_dims(imgs[250,:,:,:], axis = 0))
  	
            print(layer_name)
            layer_output = np.rollaxis(layer_output, 3)
            layer_output = np.squeeze(layer_output)
            print(layer_output.shape)
    
            itkimage = sitk.GetImageFromArray(layer_output)
            sitk.WriteImage(itkimage, output_folder +str(id)+ '_' + layer_name + '.nii.gz', True)
            print('save seg to ' + output_folder +str(id)+'_'+ layer_name + '.nii.gz')

    itkimage = sitk.GetImageFromArray(np.squeeze(imgs[250]))
    sitk.WriteImage(itkimage, output_folder+str(id)+ '.nii.gz', True)
    print("original image saved")  

