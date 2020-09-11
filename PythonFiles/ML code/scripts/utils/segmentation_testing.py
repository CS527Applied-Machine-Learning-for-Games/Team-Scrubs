import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from itertools import zip_longest
import cv2

from utils.dice import dice_coef, portion_wrong_image
from utils.kerasutils import get_image, rescale_intensity_batch
from utils.imageutils import map_label, label_overlay, colormap_31
from utils.input_data import InputSegmentationArrays

"""This module provides testing of a trained segmentation model. 
"""

__author__ = 'Ken C. L. Wong'


def segmentation_testing(
        get_segmentation,
        input_arrays,
        output_path=None,
        output_image_path=None,
        rescale_types=None,
        rescale_probability=1.0,
        window_centers=None,
        window_widths=None,
        batch_size=1,
        save_image_batches=1,
        save_overlay=True,
        image_alpha=0.5,
        label_alpha=0.5,
        cmap=colormap_31(),
        use_lumped_dice=True,
        unique_labels=None
):
    """Testing of a trained segmentation model. If ground truth is given (label_valid), the dice coefficients are
    computed. Segmentation results

    :param get_segmentation: a function that gets segmentation.
    :param InputSegmentationArrays input_arrays: input arrays of all training, validation, and testing images and
    labels.
    :param output_path: output directory.
    :param output_image_path: directory for saving testing results.
    :param rescale_types: if not None, contains a list of image intensity rescaling types to be performed.
    :param rescale_probability: probability to perform rescaling on an image batch.
    :param window_centers: if is dict, stores the window centers of different labels. Otherwise, the window center (
    scalar or list) used by all rescaling.
    :param window_widths: if is dict, stores the window widths of different labels. Otherwise, the window width (
    scalar or list) used by all rescaling.
    :param batch_size: batch size.
    :param save_image_batches: number of batches per testing image save. Valid only when output_image_path is not None.
    :param save_overlay: if True, save the original image, and the image overlays of the prediction and the ground
    truth (if given).
    :param image_alpha: transparency for image overlay.
    :param label_alpha: transparency for image overlay.
    :param cmap: colormap for image overlay.
    :param use_lumped_dice: if True, computes the Dice coefficients using all pixels from all images.
    :param unique_labels: list of unique label arrays for remapping different modalities for visualization.
    0 (background) excluded.
    """

    if output_path is None:
        output_path = output_image_path

    save_image_batches = int(save_image_batches)

    # Create output directories
    if output_image_path is not None and not os.path.exists(output_image_path):
        os.makedirs(output_image_path)
    if output_path is not None and not os.path.exists(output_path):
        os.makedirs(output_path)

    # print
    print('Testing trained model ...')
    start_time = time.time()

    n_test_batches = input_arrays.get_test_num_batches(batch_size)

    flow = input_arrays.get_test_flow(batch_size)
    if input_arrays.label_test is None:
        flow = zip_longest(flow, '')

    n_batches = 0
    save_ctr = 0
    seg_all = []
    label_all = []
    for image, label in flow:

        if rescale_types is not None and np.random.binomial(1, rescale_probability):
            rescale_intensity_batch(image, rescale_types, window_centers=window_centers, window_widths=window_widths)
        seg = get_segmentation(image)
        seg_all.append(seg)

        if label is not None:
            label_all.append(label)

        if output_image_path is not None and n_batches % save_image_batches == 0:

            img = get_image(image, 0)
            pred = seg[0]

            # Change to continuous labels for better color-mapping
            if unique_labels is not None:
                pred = pred.copy()
                p = np.unique(pred)
                for u in unique_labels:
                    if np.sum(np.in1d(p, u)) >= len(p) / 2:
                        pred[pred != 0] -= (np.min(u) - 1)
                        break

            # Temporary fix for 3D images
            if img.ndim == 3:
                img = img[len(img) / 2]
                pred = pred[len(pred) / 2]

            if label is not None:
                plt.figure(figsize=(15, 5))
            else:
                plt.figure(figsize=(10, 5))

            ax = plt.subplot(1, 3, 1)
            ax.set_title('Original Image')
            ax.imshow(img, cmap=cm.Greys_r, interpolation='none')
            ax.axis('off')

            ax = plt.subplot(1, 3, 2)
            ax.set_title('Deep Net Output')
            ax.imshow(cv2.cvtColor(map_label(pred, cmap=cmap), cv2.COLOR_RGB2BGR), interpolation='none')
            ax.axis('off')

            lab = None
            if label is not None:
                lab = get_image(label, 0)

                # Change to continuous labels for better color-mapping
                if unique_labels is not None:
                    lab = lab.copy()
                    for u in unique_labels:
                        if lab.max() in u:
                            lab[lab != 0] -= (np.min(u) - 1)
                            break

                if lab.ndim == 3:
                    lab = lab[len(lab) / 2]
                ax = plt.subplot(1, 3, 3)
                ax.set_title('Ground Truth')
                ax.imshow(cv2.cvtColor(map_label(lab, cmap=cmap), cv2.COLOR_RGB2BGR), interpolation='none')
                ax.axis('off')

            plt.savefig(output_image_path + '/%05d.png' % save_ctr)

            if save_overlay:
                cv2.imwrite(output_image_path + '/%05d_img.png' % save_ctr, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                overlay_pred, _ = label_overlay(img, pred, image_alpha=image_alpha, label_alpha=label_alpha)
                cv2.imwrite(output_image_path + '/%05d_pred.png' % save_ctr, overlay_pred,
                            [cv2.IMWRITE_PNG_COMPRESSION, 9])
                if label is not None:
                    overlay_gnd, _ = label_overlay(img, lab, image_alpha=image_alpha, label_alpha=label_alpha)
                    cv2.imwrite(output_image_path + '/%05d_gnd.png' % save_ctr, overlay_gnd,
                                [cv2.IMWRITE_PNG_COMPRESSION, 9])

            save_ctr += 1
            plt.close()

        n_batches += 1
        if n_batches == n_test_batches:
            break

    end_time = time.time()

    print('Segmentation time used: ', (end_time - start_time))

    if label_all:

        seg_all = np.concatenate(seg_all)
        label_all = np.concatenate(label_all)

        start_time = time.time()

        dice = dice_coef(label_all, seg_all, use_lumped_dice=use_lumped_dice)
        dice_mean = dice[~np.isnan(dice)][1:].mean()  # Mean without background
        portion_wrong = portion_wrong_image(label_all, seg_all)

        end_time = time.time()
        print('Dice time used: ', (end_time - start_time))

        print('Dice score: ', dice)
        print('Dice mean no background: ', dice_mean)
        print('Portion wrong image: ', portion_wrong)
        if output_path is not None:
            with open(output_path + '/dice_test.txt', 'w') as f:
                print >> f, 'Dice score: ', dice
                print >> f, 'Dice mean no background: ', dice_mean
                print >> f, 'Portion wrong image: ', portion_wrong

    print('Done')
