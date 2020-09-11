import numpy as np
import os
import sys
import cv2

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

modelPath = './models/mixlabel_z_12_27_19_model.h5'

image_weight = 0.0005
nlabel=5
ncluster=(np.zeros(nlabel,np.int) + 1)
predict_IDs = pd.read_csv('../data/test_IDs', delimiter=' ')['ID'].values

cc_weight=0
image_weight=.0005
# interval=int(sys.argv[3])
output_folder= './predictions/'
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
    kernel_cc_weight=cc_weight,
    activation='relu',
    net_depth=5,
    convs_per_depth=2,
    noise_std=1.0,
    ncluster=ncluster,
    unsupervised_weight=image_weight
    # loss=combine_loss([EncodingLength_2D(image_weight=image_weight, L=nlabel), exp_categorical_crossentropy_Im_mc(exp=1.0, class_weights=class_weights,nclass=nlabel, ncluster=ncluster)], [0.5, 0.5])
    # loss=combine_loss([VoI_2D(image_weight=image_weight, sigma=sigma), exp_categorical_crossentropy_Im_mc(exp=exp, class_weights=class_weights,nclass=nclass, ncluster=ncluster)], [1.0-alpha, alpha])
)

net = build_net(**model_args)
segmentation_model = Model(inputs=net.input, outputs=net.get_layer('segmentation').output)
segmentation_model.load_weights(modelPath)

def transfer_image(img):
    # max_v=np.max(img)
    # min_v=np.min(img)
    #
    # new_img=(img-min_v)*255/(max_v-min_v+1E-10)
    # return new_img
    return img * 10000

def generator_prediction(id):
    prefix = '../data/Image_3D/'+id+"/z/"+id
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

    imgs = np.expand_dims(imgs, axis=-1)

    return imgs



for i in range(len(predict_IDs)):
    id=predict_IDs[i]
    print(id)

    imgs = generator_prediction(id)

    sz = imgs.shape

    asegZ = np.zeros((sz[0], sz[1], sz[2], 5))

    aseg = np.zeros((sz[0], sz[1], sz[2]), np.int8)
    aseg_slice = np.zeros((sz[1], sz[2]), np.int8)

    for j in range(sz[0]):
        seg_mc_Z = segmentation_model.predict(np.expand_dims(imgs[j, :, :, :], axis=0))
        asegZ[j, :, :, :] = seg_mc_Z[0, :, :, :]
        ##new for CS527
        slice = asegZ[j].argmax(get_channel_axis())
        for i in range(nlabel):
            for j in range(ncluster[i] - 1):
                slice[slice == ind] = i
                ind = ind + 1

        for row in range(sz[1]):
            aseg_slice[row,:] = slice[row,:]

        itkimage = sitk.GetImageFromArray(aseg_slice)
        sitk.WriteImage(itkimage, output_folder + str(id) + '_' + str(j) + '_autoseg.nii.gz', True)
        print('save seg slice to ' + output_folder + str(id) + '_' + str(j) + '_autoseg.nii.gz')


    seg_mc = ((asegZ+asegZ)/2).argmax(get_channel_axis())
    seg = seg_mc.copy()

    ind = nlabel
    for i in range(nlabel):
        for j in range(ncluster[i] - 1):
            seg[seg == ind] = i
            ind = ind + 1

    for j in range(sz[0]):
        aseg[j,:,:] = seg[j,:,:]

	
    print(aseg.shape)
    itkimage = sitk.GetImageFromArray(aseg)
    sitk.WriteImage(itkimage, output_folder +str(id)+'_autoseg.nii.gz', True)
    print('save seg to ' + output_folder +str(id)+'_autoseg.nii.gz')






