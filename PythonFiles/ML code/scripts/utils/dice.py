import numpy as np

# def dice_coef(y_pred, y_true, labels, pixels_per_image=None):
#     """Computes the Dice coefficients of specified labels.
#
#     :param list/numpy.array y_pred: prediction. Can be from multiple images.
#     :param list/numpy.array y_true: ground truth. Same size as y_pred.
#     :param list/numpy.array labels: labels for which the Dice coefficients are computed.
#     :param int pixels_per_image: number of pixels per image. If None, y_pred and y_true are considered from a single
#     image. This is less restrictive but more robust to noise (e.g. labels with only a few pixels).
#     :return numpy.array: the average Dice coefficients w.r.t. images.
#     """
#
#     # Standardize input formats
#     y_true = np.array(y_true).ravel()
#     y_pred = np.array(y_pred).ravel()
#     assert y_true.shape == y_pred.shape
#
#     # Compute number of images and error checking
#     total_pixels = len(y_true)
#     if pixels_per_image is None:
#         pixels_per_image = total_pixels
#     pixels_per_image = int(pixels_per_image)
#     num_images = total_pixels / pixels_per_image
#     if total_pixels % pixels_per_image != 0:
#         raise RuntimeError('pixels_per_image does not match.')
#
#     # Compute Dice coefficients
#     dice_all = []
#     for i in range(num_images):
#         if num_images == 1:
#             y_true_img = y_true
#             y_pred_img = y_pred
#         else:
#             idx = i*pixels_per_image
#             y_true_img = y_true[idx:idx+pixels_per_image]
#             y_pred_img = y_pred[idx:idx+pixels_per_image]
#         dice = []
#         for label in labels:
#             y_true_bin = (y_true_img == label)
#             y_pred_bin = (y_pred_img == label)
#             intersection = np.count_nonzero(y_true_bin & y_pred_bin)
#             y_true_count = np.count_nonzero(y_true_bin)
#             if y_true_count:
#                 dice.append((2.*intersection) / (y_true_count + np.count_nonzero(y_pred_bin)))
#             else:
#                 dice.append(-1)  # label does not exist in y_true
#         dice_all.append(dice)
#
#     # row: labels; col: images
#     dice_all = np.array(dice_all).T
#
#     # Loop through each label
#     output = []
#     for dice in dice_all:
#         idx = dice != -1
#         if idx.sum() != 0:
#             output.append(dice[idx].mean())
#         else:
#             output.append(np.nan)
#     return np.array(output)


def dice_coef(y_true, y_pred, labels=None, use_lumped_dice=True):
    """Computes the Dice coefficients of specified labels.

    :param list/numpy.array y_true: ground truth. Can be bhw or bdhw with or without last channel = 1.
    :param list/numpy.array y_pred: prediction. Can be bhw or bdhw with or without last channel = 1.
    :param list/numpy.array labels: labels for which the Dice coefficients are computed.
    :param bool use_lumped_dice: if True, computes the Dice coefficients using all pixels from all images.
    This is less restrictive but more robust to noise (e.g. labels with only a few pixels).
    :return numpy.array: the Dice coefficients of the labels averaged from images.
    """

    # Standardize input formats
    y_true = np.array(y_true, dtype=np.float)
    y_pred = np.array(y_pred, dtype=np.float)

    # Reshape according to use_lumped_dice
    y_true = y_true.reshape(1, -1) if use_lumped_dice else y_true.reshape(len(y_true), -1)
    y_pred = y_pred.reshape(1, -1) if use_lumped_dice else y_pred.reshape(len(y_pred), -1)
    assert y_true.shape == y_pred.shape

    if labels is None:
        labels = np.unique(y_true)

    # Compute Dice coefficients
    dice_all = []
    for y_true_img, y_pred_img in zip(y_true, y_pred):  # Loop through images
        dice = []
        for label in labels:
            y_true_bin = (y_true_img == label)
            y_pred_bin = (y_pred_img == label)
            intersection = np.count_nonzero(y_true_bin & y_pred_bin)
            y_true_count = np.count_nonzero(y_true_bin)
            if y_true_count:
                dice.append((2.*intersection) / (y_true_count + np.count_nonzero(y_pred_bin)))
            else:
                dice.append(-1)  # label does not exist in y_true
        dice_all.append(dice)

    # row: labels; col: images
    dice_all = np.array(dice_all).T

    # Loop through each label
    output = []
    for dice in dice_all:
        idx = dice != -1
        if idx.sum() != 0:
            output.append(dice[idx].mean())
        else:
            output.append(np.nan)

    return np.array(output)


def portion_wrong_image(y_true, y_pred):
    """In each image, computes the portion of pixels with labels not in the ground truth.
    Returns the average from all images.

    :param list/numpy.array y_true: ground truth. Can be bhw or bdhw with or without last channel = 1.
    :param list/numpy.array y_pred: prediction. Can be bhw or bdhw with or without last channel = 1.
    :return numpy.array: the average portion of pixels with labels not in the ground truth.
    """

    # Standardize input formats
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Reshape to ravel each image, (number of images, number of pixels)
    y_true = y_true.reshape(len(y_true), -1)
    y_pred = y_pred.reshape(len(y_pred), -1)
    assert y_true.shape == y_pred.shape

    # Computes the portion of pixels with labels not in the ground truth.
    unique_true = [np.unique(y) for y in y_true]  # Unique labels in each y_true image
    wrong_pred = np.array([~np.in1d(y, u) for u, y in zip(unique_true, y_pred)]).mean()

    return wrong_pred
