import sys
import numpy as np
from keras import backend as K

from utils.imageutils import rescale_intensity

"""Different Keras-specific utils."""

__author__ = 'Ken C. L. Wong'


def get_image(image, idx, channel=0):
    """Gets an image from a ND tensor according to the image_data_format. """
    if K.image_data_format() == 'channels_last':
        return image[idx, ..., channel]
    else:
        return image[idx, channel]


def set_image(image, idx, img):
    """Sets an image to a ND tensor according to the image_data_format.

    :param image: ND tensor
    :param idx: index to image
    :param img: replacing image
    """
    if K.image_data_format() == 'channels_last':
        image[idx, ..., 0] = img
    else:
        image[idx, 0] = img


def get_channel_axis():
    """Gets the channel axis."""
    if K.image_data_format() == 'channels_first':
        return 1
    else:
        return -1


def rescale_intensity_batch(image, rescale_types, label=None, rescale_labels=None,
                            window_centers=None, window_widths=None):
    """Rescales image intensity

    :param image: ND array of image batch. Modified at output.
    :param rescale_types: A list containing the types of operation to perform (see IntensityRescaleType).
    :param label: 2D array of image labels associated with image (shape = (len(image), 1)). Used if not all images
    are rescaled.
    :param rescale_labels: A list of labels that the corresponding images are rescaled. Used if not all images are
    rescaled.
    :param window_centers: If is dict, stores the window centers of different labels. Otherwise, the window center (
    scalar or list) used by all rescaling.
    :param window_widths: If is dict, stores the window widths of different labels. Otherwise, the window width (
    scalar or list) used by all rescaling.
    """

    # Error checking
    if label is not None:
        assert label.shape == (len(image), 1)

    # Loop through each image
    for i in range(len(image)):

        # Skip if the corresponding label is not in the rescale_labels list
        if label is not None and rescale_labels is not None and label[i, 0] not in rescale_labels:
            continue

        # Get window center and width
        if window_centers is None or window_widths is None:
            center = None
            width = None
        else:
            if not isinstance(window_centers, dict):
                center = window_centers
            else:
                center = window_centers[label[i, 0]]
            if not isinstance(window_widths, dict):
                width = window_widths
            else:
                width = window_widths[label[i, 0]]

        img = get_image(image, i)
        set_image(image, i, rescale_intensity(img, rescale_types, center, width))


def correct_data_format(data):
    """Corrects data format according to K.image_data_format().

    :param numpy.array data: a ND array of image or label data.
    :return: corrected data. No copy is made.
    """
    if K.image_data_format() == 'channels_last' and np.argmin(data.shape[1:]) == 0:
        axes = range(1, data.ndim)
        axes = [0] + axes[1:] + axes[:1]
        data = data.transpose(axes)
    elif K.image_data_format() == 'channels_first' and np.argmin(data.shape[1:]) == data.ndim - 2:
        axes = range(1, data.ndim)
        axes = [0] + axes[-1:] + axes[:-1]
        data = data.transpose(axes)
    return data


def save_model_summary(model, path):
    """Saves model summary to a text file.

    :param model: the model.
    :param path: text file.
    """
    with open(path, 'w') as f:
        current_stdout = sys.stdout
        sys.stdout = f
        print(model.summary())
        sys.stdout = current_stdout
