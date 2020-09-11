
# coding: utf-8

# ### Load the required libraries

# In[1]:

import numpy as np
import os
import sys

from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import plot_model
import tensorflow as tf
from numpy import inf
import matplotlib
matplotlib.use('agg')
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
import SimpleITK as sitk

# Change to your own directory
sys.path.append('../')

import os.path
from nets.unet_HW import build_net
# from nets.custom_losses import (exp_dice_loss, exp_categorical_crossentropy, combine_loss, correlation_crossentropy_2D)
from nets.custom_losses import *
from utils.segmentation_training_mc import segmentation_training
from utils.kerasutils import get_image, correct_data_format, save_model_summary, get_channel_axis
from utils.imageutils import map_label
from utils.input_data_mc import InputSegmentationArrays
# from IPython import get_ipython
import pandas as pd
import cv2
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from keras.preprocessing.image import ImageDataGenerator

# When displaying figures have them as inline with notebook and not as standalone popups
# get_ipython().magic(u'matplotlib inline')

# alpha=float(sys.argv[2])
# exp=float(sys.argv[3])
cc_weight=0
# image_weight=float(sys.argv[2])
# interval=int(sys.argv[3])
N_epoch=int(sys.argv[2])
N_per_epoch=int(sys.argv[3])
output_folder=sys.argv[4]
model_name=sys.argv[5]
N_labeled=int(sys.argv[6])
output_model_path=output_folder+'/'+model_name
print('model_name')
print(model_name)
# print('alpha:', alpha)
# print('exp: ',exp)
# print('cc_weight: ', cc_weight)
# print('image_weight: ', image_weight)
print('N_epoch',N_epoch)
print('N_per_epoch',N_per_epoch)
print('N_labeled',N_labeled)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# print('segmentation file',seg_name)
#
if K.backend() == 'tensorflow':
    # Use only gpu #X (with tf.device(/gpu:X) does not work)
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    # Automatically choose an existing and supported device if the specified one does not exist
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # To constrain the use of gpu memory, otherwise all memory is used
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

suffix = 'nlst'

nlabel=5
ncluster=(np.zeros(nlabel,np.int) + 1)
# ncluster[0]=5

class_weights=np.ones(nlabel,np.float32)
data_table = pd.read_csv('2D/mixlabel/train_list', delimiter=' ')
train_img_files = pd.read_csv('2D/mixlabel/train_list', delimiter=' ')['file'].values
train_seg_files = pd.read_csv('2D/mixlabel/train_label_list', delimiter=' ')['file'].values
test_img_files = pd.read_csv('2D/mixlabel/test_list', delimiter=' ')['file'].values
test_seg_files = pd.read_csv('2D/mixlabel/test_label_list', delimiter=' ')['file'].values
valid_img_files = pd.read_csv('2D/mixlabel/valid_list', delimiter=' ')['file'].values
valid_seg_files = pd.read_csv('2D/mixlabel/valid_label_list', delimiter=' ')['file'].values
# unlabeled_img_files = pd.read_csv('2D/mixlabel/unlabeled_list', delimiter=' ')['file'].values
train_IDs = pd.read_csv('2D/mixlabel/train_IDs_baseline', delimiter=' ')['ID'].values
train_flags = pd.read_csv('2D/mixlabel/train_IDs_baseline', delimiter=' ')['flag'].values
valid_IDs = pd.read_csv('2D/mixlabel/valid_IDs', delimiter=' ')['ID'].values
test_IDs = pd.read_csv('2D/mixlabel/test_IDs', delimiter=' ')['ID'].values

print(train_img_files)
print(valid_IDs)
print(test_IDs)

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def transfer_seg(seg):
    # seg[seg == 63] = 0
    # seg[seg == 95] = 1  # Membrane
    # seg[seg == 80] = 2  # Nucleus
    # seg[seg == 127] = 3  # Granules
    # seg[seg == 47] = 4  # Mito
    # seg[seg == 112] = 1  # Lipid
    #
    # seg[seg == 3] = 1  # Granules
    # seg[seg == 4] = 1  # Mito

    return seg
def transfer_image(img):
    # max_v=np.max(img)
    # min_v=np.min(img)
    #
    # new_img=(img-min_v)*255/(max_v-min_v+1E-10)
    # return new_img
    return img * 10000

def generator_train_from_IDlist(train_list, train_flags, view='x', angle=0.0, scale=0):
  print(train_list)
  list_L = len(train_list)
  print('list_L: ',list_L)
  while 1:
    index = random.randint(0, list_L-1)
    id = train_list[index]
    # flag=np.zeros((1,1,1,1))
    flag=train_flags[index]
    loc=random.randint(0, 511)
    segfn = '3D/Label_3D/' + id + '/' + view + '/' + id + '_labels_2D_' + view + '_' + str(loc) + '.nii.gz'
    seg = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(segfn)))
    while np.max(seg)==0:
        loc = random.randint(0, 511)
        segfn = '3D/Label_3D/' + id + '/' + view + '/' + id + '_labels_2D_' + view + '_' + str(
                loc) + '.nii.gz'
        seg = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(segfn)))

    imgfn = '3D/Image_3D/' + id + '/' + view + '/' + id + '_2D_' + view + '_' + str(loc) + '.nii.gz'
    segfn = '3D/Label_3D/' + id + '/' + view + '/' + id + '_labels_2D_' + view + '_' + str(loc) + '.nii.gz'
    seg_supervised = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(segfn)))
    img_supervised = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(imgfn)))

    rangle = np.random.random() * angle
    sc = (np.random.random() - 0.5) * 2 * scale + 1.0
    rows, cols = img_supervised.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rangle, sc)

    warped_img_supervised = transfer_image(cv2.warpAffine(img_supervised, M, (cols, rows), flags=1))
    warped_seg_supervised = cv2.warpAffine(seg_supervised, M, (cols, rows), flags=0)
    # warped_img_supervised = transfer_image(img_supervised)
    # warped_seg_supervised = seg_supervised

    warped_img_supervised = np.expand_dims(warped_img_supervised, axis=-1)
    warped_seg_supervised = to_categorical(warped_seg_supervised, num_classes=nlabel)
    warped_img_supervised = np.expand_dims(warped_img_supervised, axis=0)
    warped_seg_supervised = np.expand_dims(warped_seg_supervised, axis=0)

    yield warped_img_supervised, warped_seg_supervised, flag


'''
Generator for images and labels
'''
def generator_volume(id, view, nlabel):
    # prefix = '2D/Image_valid/'+id
    # seg_prefix = '2D/Label_valid/'+id
    # prefix = '3D/Image_3D/' + id + '/' + view + '/' + id + '_equalized_2D_' + view
    prefix = '3D/Image_3D/' + id + '/' + view + '/' + id + '_2D_' + view
    seg_prefix = '3D/Label_3D/' + id + '/' + view + '/' + id + '_labels_2D_' + view
    flag=1
    length=0
    while flag>0:
        fn=prefix+'_'+str(length)+'.nii.gz'
        # print(fn)
        if os.path.isfile(fn):
            length += 1
        else:
            flag=0

    # print(length)
    # kk
    imgs = np.zeros((length, 512, 512))
    segs = np.zeros((length, 512, 512))
    # probs = np.zeros((length, 512, 512, nlabel))
    c = 0
    for i in range(0,length):
        segfn= seg_prefix+'_'+str(i)+'.nii.gz'
        imfn = prefix+'_'+str(i)+'.nii.gz'

        seg=sitk.GetArrayFromImage(sitk.ReadImage(segfn))
        img = transfer_image(sitk.GetArrayFromImage(sitk.ReadImage(imfn)))

        imgs[c,:,:]=img
        segs[c,:,:]=seg
        c=c+1

    imgs = np.expand_dims(imgs, axis=-1)
    segs = transfer_seg(segs)
    probs = to_categorical(segs, nlabel)

    return imgs, probs, segs

'''
Generator for images and labels
'''
def generator_test(id):
    prefix = '2D/Image/'+id
    seg_prefix = '2D/Label/'+id
    flag=1
    length=0
    while flag>0:
        fn=prefix+'_'+str(length)+'.nii.gz'
        # print(fn)
        if os.path.isfile(fn):
            length += 1
        else:
            flag=0

    # print(length)
    # kk
    imgs = np.zeros((length, 512, 512))
    segs = np.zeros((length, 512, 512))
    probs = np.zeros((length, 512, 512, nlabel))
    c = 0
    for i in range(0,length):
        segfn= seg_prefix+'_labels_'+str(i)+'.nii.gz'
        imfn = prefix+'_'+str(i)+'.nii.gz'

        seg=sitk.GetArrayFromImage(sitk.ReadImage(segfn))
        img = transfer_image(sitk.GetArrayFromImage(sitk.ReadImage(imfn)))

        imgs[c,:,:]=img
        segs[c,:,:]=seg
        # probs[c,:,:,:]=prob
        c=c+1

    imgs = np.expand_dims(imgs, axis=-1)
    segs = transfer_seg(segs)
    probs = to_categorical(segs, nlabel)

    return imgs, probs, segs

# '''
# Generator for images and labels
# '''
# def generator_valid(id):
#     prefix = '2D/Image_valid/'+id
#     seg_prefix = '2D/Label_valid/'+id
#     flag=1
#     length=0
#     while flag>0:
#         fn=prefix+'_'+str(length)+'.nii.gz'
#         # print(fn)
#         if os.path.isfile(fn):
#             length += 1
#         else:
#             flag=0
#
#     # print(length)
#     # kk
#     imgs = np.zeros((length, 512, 512))
#     segs = np.zeros((length, 512, 512))
#     probs = np.zeros((length, 512, 512, nlabel))
#     c = 0
#     for i in range(0,length):
#         segfn= seg_prefix+'_labels_'+str(i)+'.nii.gz'
#         imfn = prefix+'_'+str(i)+'.nii.gz'
#
#         seg=sitk.GetArrayFromImage(sitk.ReadImage(segfn))
#         img = transfer_image(sitk.GetArrayFromImage(sitk.ReadImage(imfn)))
#
#         imgs[c,:,:]=img
#         segs[c,:,:]=seg
#         # probs[c,:,:,:]=prob
#         c=c+1
#
#     imgs = np.expand_dims(imgs, axis=-1)
#     segs = transfer_seg(segs)
#     probs = to_categorical(segs, nlabel)
#
#     return imgs, probs, segs

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
    # unsupervised_weight=image_weight
    # loss=combine_loss([EncodingLength_2D(image_weight=image_weight, L=nlabel), exp_categorical_crossentropy_Im_mc(exp=1.0, class_weights=class_weights,nclass=nlabel, ncluster=ncluster)], [0.5, 0.5])
    # loss=combine_loss([VoI_2D(image_weight=image_weight, sigma=sigma), exp_categorical_crossentropy_Im_mc(exp=exp, class_weights=class_weights,nclass=nclass, ncluster=ncluster)], [1.0-alpha, alpha])
)


net = build_net(**model_args)
segmentation_model = Model(inputs=net.input, outputs=net.get_layer('segmentation').output)

print(net.summary())

def get_segmentation(image, nclass, ncluster):
    seg_mc=segmentation_model.predict(image).argmax(get_channel_axis())
    seg=seg_mc.copy()
    ind = nclass
    for i in range(nclass):
        for j in range(ncluster[i] - 1):
            seg[seg == ind] = i
            ind = ind + 1
    return seg, seg_mc

def get_segmentation_from_prediction(predicted, nclass, ncluster):
    seg_mc=predicted.argmax(get_channel_axis())
    seg=seg_mc.copy()
    ind = nclass
    for i in range(nclass):
        for j in range(ncluster[i] - 1):
            seg[seg == ind] = i
            ind = ind + 1
    return seg, seg_mc

# N_unet_epoch=100
# n_per_epoch=1000
slice_neighbor=0
best_val_dice=0
for n_round in range(N_epoch):
    loss_supervised=0
    loss_unsupervised=0
    n_batches=0
    data = generator_train_from_IDlist(train_IDs, train_flags, view='z', angle=360, scale=0.2)
    # data = generator_train_from_filelist(train_img_files, train_seg_files, angle=30.0, scale=0.2)

    print('train supervised_loss unsupervised_loss')
    for i in range(N_per_epoch):
        img_supervised, seg_supervised, flag = next(data)
        seg_supervised[0,0,0,0]=flag
        # print(np.max(seg_supervised.argmax(axis=-1)))
        sloss = net.train_on_batch(img_supervised, seg_supervised)
        loss_supervised=loss_supervised+sloss
        n_batches += 1
        # print('\r'+str(n_batches)+' '+str(flag)+' '+str(sloss)+' '+str(loss_supervised/n_batches)+' '+str(loss_unsupervised/n_batches))

        sys.stdout.write('\r'+str(n_batches)+' '+str(flag)+' '+str(sloss)+' '+str(loss_supervised/n_batches)+' '+str(loss_unsupervised/n_batches))
        sys.stdout.flush()
    # print(loss_supervised, n_batches)
    loss_supervised /= n_batches
    loss_unsupervised /= n_batches
    print('train supervised loss: '+str(loss_supervised)+' unsupervised loss: '+str(loss_unsupervised))
    with open(output_folder + '/'+model_name+'_stdout.txt', 'a') as f:
        print('epoch: ', n_round, end="", file=f)
        print('train supervised loss: ', loss_supervised, end="", file=f)
        print('train unsupervised loss: ', loss_unsupervised, end="", file=f)

    with open(output_folder + '/'+model_name+'_res.txt', 'a') as f:
        print(' %f ' % loss_supervised, ' ', loss_unsupervised, end="", file=f)

    # continue;

    train_loss_epoch=0
    n_batches=0
    dice_train=0.0
    def dice_coef(seg1,seg2,nlabel):
        Dice=np.zeros(nlabel-1,np.float32)
        for i in range(nlabel):
            if i==0:
                continue
            L1=(seg1==i)
            L2=(seg2==i)
            v1=np.sum(L1)
            v2=np.sum(L2)
            ov=np.sum(np.multiply(L1,L2))
            Dice[i-1]=ov*2/(v1+v2+1E-10)

        return Dice

    # for i in range(1):
    #     id=train_IDs[15+i]
    #     print(id)
    #     # imgs, probs, segs= generator_test(id)
    #     imgs, probs, segs= generator_volume(id, view='z', nlabel=5)
    #     # print(imgs.shape, segs.shape)
    #
    #     sz=imgs.shape
    #     aseg=np.zeros((sz[0],sz[1],sz[2]),np.int32)
    #     for j in range(sz[0]):
    #         tloss = net.test_on_batch(np.expand_dims(imgs[j,:,:,:],axis=0), np.expand_dims(probs[j,],axis=0))
    #         train_loss_epoch += tloss
    #         seg, segmc = get_segmentation(np.expand_dims(imgs[j,:,:,:],axis=0), nlabel, ncluster)
    #         # print(seg.shape)
    #         aseg[j,:,:]=seg[0,:,:]
    #         n_batches += 1
    #         # sys.stdout.write('Validation id slice: \r' + str(id)+' '+str(j))
    #         # sys.stdout.flush()
    #     dice = dice_coef(segs, aseg, nlabel)
    #     itkimage = sitk.GetImageFromArray(aseg)
    #     sitk.WriteImage(itkimage, output_folder + '/'+str(id)+'_current_train_seg.nii.gz', True)
    #     print('save seg to ' + output_folder + '/'+str(id)+'_current_train_seg.nii.gz')
    #
    #     print(dice)
    #     dice_train+=dice
    # train_loss_epoch /= n_batches
    # dice_train /= 3
    # mean_dice_train = np.mean(dice_train)
    # print('val loss: '+str(train_loss_epoch)+' dice: '+str(dice_train)+' '+str(mean_dice_train))
    # with open(output_folder + '/'+model_name+'_stdout.txt', 'a') as f:
    #     print('valid loss: ', train_loss_epoch, end="", file=f)
    #     print('valid dice: ', mean_dice_train, end="", file=f)
    # with open(output_folder + '/'+model_name+'_res.txt', 'a') as f:
    #     print(' %f ' % train_loss_epoch, ' ', mean_dice_train, end="", file=f)


    val_loss_epoch=0
    n_batches=0
    dice_val=0.0

    for i in range(len(valid_IDs)):
        id=valid_IDs[i]
        print(id)
        imgs, probs, segs=generator_volume(id, view='z', nlabel=nlabel)

        # imgs, probs, segs= generator_test(id)
        # print(imgs.shape, segs.shape)

        sz=imgs.shape
        aseg=np.zeros((sz[0],sz[1],sz[2]),np.int32)
        for j in range(sz[0]):
            tloss = net.test_on_batch(np.expand_dims(imgs[j,:,:,:],axis=0), np.expand_dims(probs[j,],axis=0))
            val_loss_epoch += tloss
            seg, segmc = get_segmentation(np.expand_dims(imgs[j,:,:,:],axis=0), nlabel, ncluster)
            # print(seg.shape)
            aseg[j,:,:]=seg[0,:,:]
            n_batches += 1
            # sys.stdout.write('Validation id slice: \r' + str(id)+' '+str(j))
            # sys.stdout.flush()
        dice = dice_coef(segs, aseg, nlabel)
        itkimage = sitk.GetImageFromArray(aseg)
        sitk.WriteImage(itkimage, output_folder + '/'+str(id)+'_current_valid_seg.nii.gz', True)
        print('save seg to ' + output_folder + '/'+str(id)+'_current_valid_seg.nii.gz')

        print(dice)
        dice_val+=dice
    val_loss_epoch /= n_batches
    dice_val /= len(valid_IDs)
    mean_dice_val = np.mean(dice_val)
    print('val loss: '+str(val_loss_epoch)+' dice: '+str(dice_val)+' '+str(mean_dice_val)+' best dice: '+str(best_val_dice))
    with open(output_folder + '/'+model_name+'_stdout.txt', 'a') as f:
        print('valid loss: ', val_loss_epoch, end="", file=f)
        print('valid dice: ', mean_dice_val, end="", file=f)
    with open(output_folder + '/'+model_name+'_res.txt', 'a') as f:
        print(' %f ' % val_loss_epoch, ' ', mean_dice_val, end="", file=f)

    if best_val_dice >= mean_dice_val:
        with open(output_folder + '/' + model_name + '_stdout.txt', 'a') as f:
            print('\n', end="", file=f)
        with open(output_folder + '/' + model_name + '_res.txt', 'a') as f:
            print('\n', end="", file=f)
        continue;

    best_val_dice = mean_dice_val

    print('best_model_updated: ')
    # net.set_weights(best_model_weights)
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    net.save(output_model_path + '_model.h5')
    print('saved model to ' + output_model_path + '_model.h5')

    test_loss_epoch=0
    n_batches=0
    dice_test=0.0
    for i in range(len(test_IDs)):
        id=test_IDs[i]
        # imgs, probs, segs= generator_test(id)
        imgs, probs, segs=generator_volume(id, view='z', nlabel=nlabel)
        sz=imgs.shape
        aseg=np.zeros((sz[0],sz[1],sz[2]),np.int32)
        for j in range(sz[0]):
            loss = net.test_on_batch(np.expand_dims(imgs[j,:,:,:],axis=0), np.expand_dims(probs[j,],axis=0))
            test_loss_epoch += tloss
            seg, segmc = get_segmentation(np.expand_dims(imgs[j,:,:,:],axis=0), nlabel, ncluster)
            aseg[j,:,:]=seg[0,:,:]
            n_batches += 1

            # sys.stdout.write('Test id: %d  slice: %d\r'  + str(id)+' '+str(j))
            # sys.stdout.flush()

        dice = dice_coef(segs, aseg, nlabel)
        dice_test += dice
        itkimage = sitk.GetImageFromArray(aseg)
        sitk.WriteImage(itkimage, output_folder + '/' + str(id) + '_best_test_seg.nii.gz', True)
        print('save seg to ' + output_folder + '/' + str(id) + '_best_test_seg.nii.gz')
        print('test loss: '+str(test_loss_epoch/n_batches)+' dice: '+str(dice_test/len(test_IDs))+' '+str(np.mean(dice_test/len(test_IDs))))

    test_loss_epoch /= n_batches
    dice_test /= len(test_IDs)
    mean_dice_test = np.mean(dice_test)
    print('test loss: '+str(test_loss_epoch)+' dice: '+str(dice_test)+' '+str(mean_dice_test))
    with open(output_folder + '/'+model_name+'_stdout.txt', 'a') as f:
        print('test loss: ', test_loss_epoch, end="", file=f)
        print('test dice: ', mean_dice_test, end="", file=f)
        print('\n', end="", file=f)

    with open(output_folder + '/'+model_name+'_res.txt', 'a') as f:
        print(' %f ' % test_loss_epoch, ' ', mean_dice_test, end="", file=f)
        print('\n', end="", file=f)

