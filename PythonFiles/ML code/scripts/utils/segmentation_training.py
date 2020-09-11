import os
import sys
import numpy as np

from keras import backend as K
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
import time
import gc
from numpy import inf
import SimpleITK as sitk

from utils.dice import dice_coef, portion_wrong_image
from utils.kerasutils import rescale_intensity_batch
from utils.imageutils import colormap_31
from utils.input_data import InputSegmentationArrays
from .segmentation_testing import segmentation_testing

"""This module summarizes the common procedures for training a segmentation model. 
"""

__author__ = 'Ken C. L. Wong'


def segmentation_training(
        net,
        get_segmentation,
        input_arrays,
        output_path,
        output_model_path=None,
        output_image_path=None,
        rescale_types=None,
        rescale_probability=1.0,
        window_centers=None,
        window_widths=None,
        aug_args=None,
        batch_size=10,
        n_batches_per_epoch=None,
        valid_batches_per_epoch=None,
        n_epochs=100,
        print_freq=0.1,
        save_image_batches=1,
        class_weights_exp=0.5,
        cmap=colormap_31(),
        lr_schedule=None,
        use_lumped_dice=True,
        model_choose_portion=0.8,
        unique_labels=None
):
    """This function summarizes the common procedures for training a disease classification model.

    :param net: the network to be trained.
    :param get_segmentation: a function that gets segmentation results from net.
    :param InputSegmentationArrays input_arrays: input arrays of all training, validation, and testing images and
    labels.
    :param output_path: output directory.
    :param output_model_path: directory for saving the model.
    :param output_image_path: directory for saving testing results.
    :param rescale_types: if not None, contains a list of image intensity rescaling types to be performed.
    :param rescale_probability: probability to perform rescaling on an image batch.
    :param window_centers: if is dict, stores the window centers of different labels. Otherwise, the window center (
    scalar or list) used by all rescaling.
    :param window_widths: if is dict, stores the window widths of different labels. Otherwise, the window width (
    scalar or list) used by all rescaling.
    :param aug_args: arguments for image augmentation.
    :param batch_size: batch size.
    :param n_batches_per_epoch: number of training batches per epoch. None for all batches.
    :param valid_batches_per_epoch: number of validation batches per epoch. None for all batches.
    :param n_epochs: number of epochs.
    :param print_freq: Screen printing frequency. e.g. 0.1 means printing for every 10% of epochs. 0.0 to print all.
    None for no printing.
    :param save_image_batches: number of batches per testing image save. Valid only when output_image_path is not None.
    :param class_weights_exp: exponent for modifying class weights.
    :param cmap: colormap for image overlay.
    :param lr_schedule: a dict with keys as epoch numbers and values as learning rates.
    :param use_lumped_dice: if True, computes the Dice coefficients using all pixels from all images.
    :param model_choose_portion: the models after this portion of n_epochs are candidates for the final model.
    :param unique_labels: list of unique label arrays for remapping different modalities for visualization.
    0 (background) excluded.
    """

    #
    # IO and data formatting
    #

    pixels_per_image = np.prod(input_arrays.image_train.shape[1:-1])
    num_labels = input_arrays.get_num_classes()

    if aug_args is None:
        aug_args = {}

    # Make sure these arguments are valid
    batch_size = int(batch_size)
    if n_batches_per_epoch is not None:
        n_batches_per_epoch = int(n_batches_per_epoch)
    else:
        n_batches_per_epoch = input_arrays.get_train_num_batches(batch_size)
    if valid_batches_per_epoch is not None:
        valid_batches_per_epoch = int(valid_batches_per_epoch)
    n_epochs = int(n_epochs)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Empty file
    with open(output_path + '_stdout.txt', 'w'):
        pass

    #
    # Training
    #

    # Goal: make the more frequent labels weigh less
    class_weights = input_arrays.get_class_weights(class_weights_exp)
    class_weights[class_weights == inf] = 0
    print('Class weights: ', class_weights)
    with open(output_path + '_stdout.txt', 'a') as f:
        print('Class weights: ', class_weights, end="", file=f)
        # print >> f, 'Class weights: ', class_weights
        # print >> f, ''

    print('Started Experiment')

    start_time = time.time()

    # Maximum validation batches when no shuffle
    max_valid_batches = input_arrays.get_valid_num_batches(batch_size)
    # Set valid_batches_per_epoch to max_valid_batches if not specified
    if valid_batches_per_epoch is None:
        valid_batches_per_epoch = max_valid_batches
    # Shuffle during validation is required if not all validation images are used
    if valid_batches_per_epoch < max_valid_batches:
        valid_shuffle = True
    else:
        valid_shuffle = False

    print('n_batches_per_epoch: ', n_batches_per_epoch)
    print('valid_batches_per_epoch: ', valid_batches_per_epoch)

    train_loss = []
    val_loss = []
    test_loss = []
    dice = []
    max_dice = -float('inf')
    best_epoch = 0
    best_model_weights = None

    for epoch in range(n_epochs):
        best_model_updated=0
        if lr_schedule is not None:
            if epoch in lr_schedule:
                lr = lr_schedule[epoch]
                K.set_value(net.optimizer.lr, lr)
                print('**** Learning rate updated to ', lr)

        # Training phase

        train_loss_epoch = []
        n_batches = 0
        for image, label in input_arrays.get_train_flow(batch_size, aug_args, True):
            if rescale_types is not None and np.random.binomial(1, rescale_probability):
                rescale_intensity_batch(image, rescale_types, window_centers=window_centers,
                                        window_widths=window_widths)
            label = label.astype(int)
            Ls=np.unique(label)
            # print('unique label:',Ls)
            # print(len(np.unique(image)))
            # print(label.shape)
            # print(weights.shape)
            # weights = class_weights[label].reshape(len(image), pixels_per_image, 1)
            # label = to_categorical(label, num_labels).reshape(len(image), pixels_per_image, num_labels) * weights
            # weights = class_weights[label]
            # label = to_categorical(label, num_labels) * weights
            label = to_categorical(label, num_labels)
            # print(label.shape)
            # print(image.shape)
            # if len(Ls)==1:
            #     # print('using image for training')
            #     label[:,:,:,0] = image[:,:,:,0]
            # print("label shape: ", label.shape)
                # print(tf.unique(label).y)
                # print(len(tf.unique(label).y))
            # print(weights.shape)

            # print('in training size:')
            # print(image.shape)
            # print(label.shape)
            loss = net.train_on_batch(image, label)
            train_loss_epoch.append(loss)
            n_batches += 1

            if print_freq is not None:
                sys.stdout.write('Training n_batches: %d\r' % n_batches)
                sys.stdout.flush()

            if n_batches == n_batches_per_epoch:
                break

        train_loss_mean = np.mean(train_loss_epoch)
        if print_freq is not None and \
                (print_freq == 0.0 or epoch % int(n_epochs * print_freq) == 0 or epoch == n_epochs - 1):
            print('\n-------------------------')
            print('Epoch: ', epoch, '\ntrain loss: %f' % train_loss_mean)
        with open(output_path + '_stdout.txt', 'a') as f:
            print('\n-------------------------', end="", file=f)
            print('Epoch: ', epoch, '\ntrain loss: %f' % train_loss_mean, end="", file=f)
            print('\n', end="", file=f)
        with open(output_path + '_res.txt', 'a') as f:
            print('\n', end="", file=f)
            print(epoch, ' %f' % train_loss_mean, end="", file=f)

            # print >> f, '\n-------------------------'
            # print >> f, 'Epoch: ', epoch, '\ntrain loss: %f' % train_loss_mean

        train_loss.append(train_loss_mean)

        # Validation phase.

        val_loss_epoch = []
        seg_all = []
        label_all = []
        n_batches = 0
        sz=input_arrays.image_valid.shape
        i=0
        seg0=np.zeros((sz[0],sz[1],sz[2]),np.float32)
        for image, label in input_arrays.get_valid_flow(batch_size, valid_shuffle):
            if rescale_types is not None and np.random.binomial(1, rescale_probability, 1):
                rescale_intensity_batch(image, rescale_types, window_centers=window_centers,
                                        window_widths=window_widths)
            label = label.astype(int)
            label_all.append(label)
            Ls=np.unique(label)

            # weights = class_weights[label].reshape(len(image), pixels_per_image, 1)
            # label = to_categorical(label, num_labels).reshape(len(image), pixels_per_image, num_labels) * weights
            weights = class_weights[label]
            # label = to_categorical(label, num_labels) * weights
            label = to_categorical(label, num_labels)
            # sz=label.shape
            # label[:,:,:,sz[3]-1] = image[:,:,:,0]
            # if len(Ls)==1:
            #     # print('using image for training')
            #     label[:,:,:,0] = image[:,:,:,0]
            # print('unique label:',np.unique(label[:,1]))
            # if len(np.unique(label[:,1]))==1:
            #     print('using image for training')
            #     label[:,0] = image
            loss = net.test_on_batch(image, label)
            val_loss_epoch.append(loss)
            seg = get_segmentation(image)

            seg_all.append(seg)
            n_batches += 1
            seg0[i,:,:]=seg
            i=i+1

            if print_freq is not None:
                sys.stdout.write('Validation n_batches: %d\r' % n_batches)
                sys.stdout.flush()

            if n_batches == valid_batches_per_epoch:
                break

        seg_all = np.concatenate(seg_all)
        label_all = np.concatenate(label_all)

        print('seg_all elem: ',np.unique(seg_all))
        print('label_all elem: ',np.unique(label_all))
        dice_epoch = dice_coef(label_all, seg_all, labels=range(num_labels), use_lumped_dice=use_lumped_dice)
        portion_wrong = portion_wrong_image(label_all, seg_all)

        val_loss_mean = np.mean(val_loss_epoch)
        val_loss.append(val_loss_mean)
        dice_epoch_mean = dice_epoch[~np.isnan(dice_epoch)].mean()
        dice.append(dice_epoch_mean)

        if print_freq is not None and \
                (print_freq == 0.0 or epoch % int(n_epochs * print_freq) == 0 or epoch == n_epochs - 1):
            print
            print('val loss: %f' % val_loss_mean)
            print('val Dice score: ', dice_epoch)
            print('Dice mean: ', dice_epoch_mean)
            print('Portion wrong image: ', portion_wrong)
        with open(output_path + '_stdout.txt', 'a') as f:
            print('val loss: %f' % val_loss_mean, end="", file=f)
            print('val Dice score: ', dice_epoch, end="", file=f)
            print('Dice mean: ', dice_epoch_mean, end="", file=f)
            print('Portion wrong image: ', portion_wrong, end="", file=f)
            print('\n', end="", file=f)
        with open(output_path + '_res.txt', 'a') as f:
            print(' %f ' % val_loss_mean, ' ', dice_epoch_mean, end="", file=f)

            # print >> f, 'val loss: %f' % val_loss_mean
            # print >> f, 'val Dice score: ', dice_epoch
            # print >> f, 'Dice mean: ', dice_epoch_mean
            # print >> f, 'Portion wrong image: ', portion_wrong


        # The image generator has some memory issues
        gc.collect()

        test_loss_epoch = []
        seg_all = []
        label_all = []
        n_batches = 0
        i=0
        seg1=np.zeros((sz[0],sz[1],sz[2]),np.float32)
        for image, label in input_arrays.get_test_flow(batch_size):
            if rescale_types is not None and np.random.binomial(1, rescale_probability, 1):
                rescale_intensity_batch(image, rescale_types, window_centers=window_centers,
                                        window_widths=window_widths)
            label = label.astype(int)
            label_all.append(label)
            Ls=np.unique(label)

            # weights = class_weights[label].reshape(len(image), pixels_per_image, 1)
            # label = to_categorical(label, num_labels).reshape(len(image), pixels_per_image, num_labels) * weights
            weights = class_weights[label]
            # label = to_categorical(label, num_labels) * weights
            label = to_categorical(label, num_labels)
            # sz=label.shape
            # label[:,:,:,sz[3]-1] = image[:,:,:,0]
            # if len(Ls)==1:
            #     # print('using image for training')
            #     label[:,:,:,0] = image[:,:,:,0]
            # if len(np.unique(label[:,1]))==1:
            #     label[:,0] = image

            loss = net.test_on_batch(image, label)
            test_loss_epoch.append(loss)
            seg = get_segmentation(image)

            seg_all.append(seg)
            n_batches += 1
            seg1[i,:,:]=seg
            i=i+1
            if print_freq is not None:
                sys.stdout.write('Test n_batches: %d\r' % n_batches)
                sys.stdout.flush()

            if n_batches == valid_batches_per_epoch:
                break

        seg_all = np.concatenate(seg_all)
        label_all = np.concatenate(label_all)

        print("Seg_all shape: " + str(seg_all.shape))
        print(label_all.shape)

        dice_epoch = dice_coef(label_all, seg_all, labels=range(num_labels), use_lumped_dice=use_lumped_dice)
        portion_wrong = portion_wrong_image(label_all, seg_all)

        test_loss_mean = np.mean(test_loss_epoch)
        test_loss.append(test_loss_mean)
        dice_epoch_mean0 = dice_epoch[~np.isnan(dice_epoch)].mean()
        dice.append(dice_epoch_mean0)

        if print_freq is not None and \
                (print_freq == 0.0 or epoch % int(n_epochs * print_freq) == 0 or epoch == n_epochs - 1):
            print
            print('test loss: %f' % test_loss_mean)
            print('test Dice score: ', dice_epoch)
            print('Dice mean: ', dice_epoch_mean0)
            print('Portion wrong image: ', portion_wrong)
        with open(output_path + '_stdout.txt', 'a') as f:
            print('test loss: %f' % test_loss_mean, end="", file=f)
            print('test Dice score: ', dice_epoch, end="", file=f)
            print('Dice mean: ', dice_epoch_mean0, end="", file=f)
            print('Portion wrong image: ', portion_wrong, end="", file=f)
        with open(output_path + '_res.txt', 'a') as f:
            print(' %f ' % test_loss_mean, ' ', dice_epoch_mean0, end="", file=f)

            # print >> f, 'val loss: %f' % val_loss_mean
            # print >> f, 'val Dice score: ', dice_epoch
            # print >> f, 'Dice mean: ', dice_epoch_mean
            # print >> f, 'Portion wrong image: ', portion_wrong

        # if (epoch > int(n_epochs * model_choose_portion) or epoch == n_epochs-1) and (dice_epoch_mean > max_dice):
        if (dice_epoch_mean > max_dice):
            max_dice = dice_epoch_mean
            best_epoch = epoch
            best_model_weights = net.get_weights()
            best_model_updated=1
            itkimage = sitk.GetImageFromArray(seg0)
            # itkimage.SetSpacing(spacing)
            # itkimage.SetOrigin(origin)
            sitk.WriteImage(itkimage, output_model_path + '_best_valid_seg.nii.gz', True)
            print('save seg to '+output_model_path +'_best_valid_seg.nii.gz')
            itkimage = sitk.GetImageFromArray(seg1)
            # itkimage.SetSpacing(spacing)
            # itkimage.SetOrigin(origin)
            sitk.WriteImage(itkimage, output_model_path + '_best_test_seg.nii.gz', True)
            print('save seg to ' + output_model_path + '_best_test_seg.nii.gz')
        # if (epoch > int(n_epochs * model_choose_portion) or epoch == n_epochs-1) and (dice_epoch_mean > max_dice):
        #     max_dice = dice_epoch_mean
        #     best_epoch = epoch
        #     best_model_weights = net.get_weights()

        # The image generator has some memory issues
        gc.collect()

        print('best_model_updated: ')
        print(best_model_updated)
        # net.set_weights(best_model_weights)
        if best_model_updated>0 and output_model_path is not None:
            if not os.path.exists(output_model_path):
                os.makedirs(output_model_path)
            net.save(output_model_path + '_model.h5')
            print('saved model to '+output_model_path + '_model.h5')

    # fig, ax = plt.subplots()
    # fig.set_size_inches(10, 5)
    # ax.plot(np.arange(0, n_epochs, 1), train_loss, 'r', label='Train loss')
    # ax.plot(np.arange(0, n_epochs, 1), val_loss, 'b-.', label='Val Loss')
    # ax.plot(np.arange(0, n_epochs, 1), dice, 'm--', label='Validation dice')
    # plt.xlabel('Epochs')
    # # Now add the legend with some customizations.
    # legend = ax.legend(loc='upper center', shadow=True)
    #
    # ax.xaxis.label.set_fontsize(20)
    # ax.tick_params(labelsize=20)
    #
    # # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    # frame = legend.get_frame()
    # frame.set_facecolor('0.90')
    #
    # # Set the font size
    # for label in legend.get_texts():
    #     label.set_fontsize('large')
    #
    # for label in legend.get_lines():
    #     label.set_linewidth(1.5)  # the legend line width
    # plt.show()
    #
    # with PdfPages(output_path + '/plot.pdf') as pp:
    #     pp.savefig(fig)
    #
    end_time = time.time()

    print('Time used: %f seconds.' % (end_time - start_time))
    print('Best epoch: ', best_epoch)
    with open(output_path + '_stdout.txt', 'a') as f:
        print('\nTime used: %f seconds.' % (end_time - start_time), end="", file=f)
        print('Best epoch: ', best_epoch, end="", file=f)
        print('\n', end="", file=f)

        # print >> f, '\nTime used: %f seconds.' % (end_time - start_time)
        # print >> f, 'Best epoch: ', best_epoch

    # #
    # # Testing with the trained model
    # #
    #
    # segmentation_testing(
    #     get_segmentation=get_segmentation,
    #     input_arrays=input_arrays,
    #     output_path=output_path,
    #     output_image_path=output_image_path,
    #     rescale_types=rescale_types,
    #     rescale_probability=rescale_probability,
    #     window_centers=window_centers,
    #     window_widths=window_widths,
    #     batch_size=batch_size,
    #     save_image_batches=save_image_batches,
    #     cmap=cmap,
    #     use_lumped_dice=use_lumped_dice,
    #     unique_labels=unique_labels
    # )
