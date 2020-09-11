import numpy as np 
import cv2
from skimage.exposure import rescale_intensity as rescale_int
import matplotlib.cm as cm
import SimpleITK as sitk

"""Different image utils."""

__author__ = 'Ken C. L. Wong'


def get_pad_crop_bound(size):
    """Gets the padding or cropping bounds using the max of the input (padding) or output (cropping) size."""
    max_size = np.max(size)
    lb = []
    ub = []
    for sz in size:
        diff = int(max_size - sz)
        lb.append(diff / 2)
        ub.append(diff - diff/2)
    return lb, ub


def pad_resize(image, output_size, hw_only=False, interpolator=sitk.sitkNearestNeighbor, make_copy=True):
    """Resizes an image and pads it to a square or cubic image if required.

    :param image: numpy array (hw, dhw, hwc, or dhwc) or SimpleITK Image (xy or xyz).
    :param output_size: output size, (height, width) or (depth, height, width).
    :param hw_only: if True, only zero-pad the height and width. 3D image only.
    :param interpolator: SimpleITK interpolator. E.g. sitk.sitkNearestNeighbor, sitk.sitkLinear.
    :param make_copy: if True, a copy of the image is returned if there is no modification.
    :return: Resized image.
    """
    if len(output_size) == 2:
        hw_only = False

    original_image = image

    # Remember image type for numpy array
    image_type = None
    if not isinstance(image, sitk.Image):
        image_type = image.dtype

    # Get input size in xy or xyz
    if isinstance(image, sitk.Image):
        input_size = image.GetSize()
    else:
        input_size = image.shape[::-1]
        if image.ndim == len(output_size) + 1:
            input_size = input_size[1:]

    # Get output size in xy or xyz
    output_size = output_size[::-1]

    # Pad square or cube if necessary
    # Same size for all directions (2D, 3D)
    if np.unique(output_size).size == 1 and np.unique(input_size).size != 1 and not hw_only:
        lb, ub = get_pad_crop_bound(input_size)
        image = pad_or_crop(sitk.ConstantPad, image, lb, ub)
    # Same size for height and width only (3D)
    elif np.unique(output_size[:2]).size == 1 and np.unique(input_size[:2]).size != 1:
        lb, ub = get_pad_crop_bound(input_size[:2])
        image = pad_or_crop(sitk.ConstantPad, image, lb + [0], ub + [0])

    # Resize
    if not all(input_size[i] == output_size[i] for i in range(len(output_size))):
        image = resize(image=image, output_size=output_size[::-1], interpolator=interpolator)

    # Restore type
    if image_type is not None:
        image = get_array(image)
        image = np.asarray(image, dtype=image_type)

    # Make a copy if no change
    if original_image is image and make_copy:
        if isinstance(image, sitk.Image):
            image = sitk.Cast(image, image.GetPixelID())
        else:
            image = image.copy()

    return image


def pad_or_crop(ops, image, lb, ub):
    """Pads or crops an image.

    :param ops: operation, sitk.ConstantPad or sitk.Crop.
    :param image: input image. Can be numpy array or SimpleITK Image.
    :param lb: padding lower bound.
    :param ub: padding upper bound.
    :return: padded image (SimpleITK Image).
    """
    # Single-channel
    if (isinstance(image, sitk.Image) and image.GetNumberOfComponentsPerPixel() == 1) or image.ndim == len(lb):
        image = get_sitk_image(image)
        image = ops(image, lb, ub)
    # Multi-channel
    else:
        image = get_array(image)
        image = np.moveaxis(image, -1, 0)
        image_channels = [ops(get_sitk_image(img), lb, ub) for img in image]
        image = sitk.Compose(image_channels)

    return image


def reverse_pad_resize(image, output_size, hw_only=False, interpolator=sitk.sitkNearestNeighbor):
    """Reverses the process of pad_resize and returns the original sized image. This function is useful for resizing
    the CNN output mask to fit the original image.

    :param image: numpy array (hw, dhw, hwc or dhwc) or SimpleITK Image (xy or xyz).
    :param output_size: output size, (height, width) or (depth, height, width), usually the shape of the original
    image before pad_resize.
    :param hw_only: if True, only crop the height and width. Needs to be consistent with pad_resize.
    :param interpolator: SimpleITK interpolator. E.g. sitk.sitkNearestNeighbor, sitk.sitkLinear.
    :return the resized image with output_size.
    """
    if len(output_size) == 2:
        hw_only = False

    original_image = image

    # Remember image type for numpy array
    image_type = None
    if not isinstance(image, sitk.Image):
        image_type = image.dtype

    # Get input size in xy or xyz
    if isinstance(image, sitk.Image):
        input_size = image.GetSize()
    else:
        input_size = image.shape[::-1]
        if image.ndim == len(output_size) + 1:
            input_size = input_size[1:]

    # Get output size in xy or xyz
    output_size = output_size[::-1]

    # Resize and crop
    # Same size for all directions (2D, 3D)
    if np.unique(input_size).size == 1 and np.unique(output_size).size != 1 and not hw_only:
        size = np.ones(len(output_size)) * np.max(output_size)
        image = resize(image=image, output_size=size, interpolator=interpolator)
        lb, ub = get_pad_crop_bound(output_size)
        image = pad_or_crop(sitk.Crop, image, lb, ub)
    # Same size for height and width only (3D)
    elif np.unique(input_size[:2]).size == 1 and np.unique(output_size[:2]).size != 1:
        xy = output_size[:2]
        size = [output_size[2]] + list(np.ones(len(xy)) * np.max(xy))  # dhw
        image = resize(image=image, output_size=size, interpolator=interpolator)
        lb, ub = get_pad_crop_bound(xy)
        image = pad_or_crop(sitk.Crop, image, lb + [0], ub + [0])
    # Resize only
    else:
        image = resize(image=image, output_size=output_size[::-1], interpolator=interpolator)

    # Restore type
    if image_type is not None:
        image = get_array(image)
        image = np.asarray(image, dtype=image_type)

    # Make a copy if no change
    if original_image is image:
        if isinstance(image, sitk.Image):
            image = sitk.Cast(image, image.GetPixelID())
        else:
            image = image.copy()

    return image


def resize_by_spacing(image, input_spacing=None, interpolator=sitk.sitkNearestNeighbor):
    """Resizes an image to isotropic spacing.

    The smallest spacing is used. This is useful as the image may be abnormally deformed when spacing information is
    discarded.

    :param image: numpy array (hw or dhw) or SimpleITK Image (xy or xyz).
    :param input_spacing: input image spacing, (height, width) or (depth, height, width).
    :param interpolator: SimpleITK interpolator. E.g. sitk.sitkNearestNeighbor, sitk.sitkLinear.
    :return: Resized image.
    """
    if input_spacing is None:
        if isinstance(image, sitk.Image):
            input_spacing = image.GetSpacing()
        else:
            raise Exception('Input spacing must be provided for non-SimpleITK images.')

    # The smallest spacing is used
    output_spacing = np.ones(len(input_spacing)) * np.min(input_spacing)

    return resize(image=image, output_spacing=output_spacing, interpolator=interpolator)


def resize(image, output_size=None, output_spacing=None, interpolator=sitk.sitkNearestNeighbor):
    """Resizes an image by the given output size and/or output spacing.

    :param image: numpy array (hw or dhw) or SimpleITK Image (xy or xyz).
    :param output_size: output size, (height, width) or (depth, height, width).
    :param output_spacing: output spacing, (height, width) or (depth, height, width).
    :param interpolator: SimpleITK interpolator. E.g. sitk.sitkNearestNeighbor, sitk.sitkLinear.
    :return: Resized image.
    """
    if output_size is None and output_spacing is None:
        raise Exception('Both output_size and output_spacing are None.')

    image_type = None
    if not isinstance(image, sitk.Image):

        # Check if vector image
        if output_size is not None:
            target_dim = len(output_size)
        else:
            target_dim = len(output_spacing)
        isVector = False
        if image.ndim == target_dim + 1:
            isVector = True

        image_type = image.dtype  # Remember the original type which may be changed during operations
        image = get_sitk_image(image, isVector=isVector)

    input_spacing = np.asarray(image.GetSpacing())
    input_size = np.asarray(image.GetSize())
    physical_size = input_spacing * input_size

    # Change to SimpleITK format, xy or xyz
    if output_size is not None:
        output_size = np.asarray(output_size)[::-1]
    if output_spacing is not None:
        output_spacing = np.asarray(output_spacing)[::-1]

    # Compute missing arguments assuming same physical size
    if output_spacing is None:
        output_spacing = physical_size / output_size
    elif output_size is None:
        output_size = physical_size / output_spacing

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetSize(np.asarray(output_size, np.int))
    resample.SetOutputSpacing(output_spacing)
    resample.SetOutputOrigin(image.GetOrigin())
    image = resample.Execute(image)

    if image_type is not None:
        image = get_array(image)
        image = np.asarray(image, dtype=image_type)

    return image


def get_sitk_image(image, isVector=False):
    """Converts to a SimpleITK Image if necessary.

    :param image: numpy array (hw or dhw) or SimpleITK Image (xy or xyz).
    :return: SimpleITK Image.
    """
    if not isinstance(image, sitk.Image):
        image = sitk.GetImageFromArray(image, isVector=isVector)  # Transpose is taken care by SimpleITK
    return image


def get_array(image):
    """Converts to a numpy array if necessary.

    :param image: numpy array (hw or dhw) or SimpleITK Image (xy or xyz).
    :return: numpy array.
    """
    if isinstance(image, sitk.Image):
        image = sitk.GetArrayFromImage(image)  # Transpose is taken care by SimpleITK
    return image


def modify_size_channel(image, output_size, channels, interpolator=sitk.sitkNearestNeighbor):
    """Modifies image size and channel.

    :param image: numpy array, channels_last, hwc or dhwc.
    :param output_size: output size, (height, width) or (depth, height, width).
    :param channels: output channels (1 or 3).
    :param interpolator: SimpleITK interpolator. E.g. sitk.sitkNearestNeighbor, sitk.sitkLinear.
    :return: modified image.
    """

    if image.ndim not in [3, 4]:
        raise Exception('Input image must be 2D or 3D with channels.')

    # Resize all channels
    input_channels = image.shape[-1]
    output_image = []
    for i in range(input_channels):
        output_image.append(resize(image=image[..., i], output_size=output_size, interpolator=interpolator))
    image = np.array(output_image)  # channels_first

    # Modify channels if needed
    if channels == 1 and input_channels == 3:
        image = image.mean(axis=0, keepdims=True)
    elif channels == 3 and input_channels == 1:
        image = image.repeat(channels, axis=0)

    # Change to channels_last
    axes = range(image.ndim)
    axes = axes[1:] + axes[:1]
    image = image.transpose(axes)

    return image


def modify_size_channel_batch(image, output_size, channels, interpolator=sitk.sitkNearestNeighbor):
    """Modifies image size and channels of an image batch.

    :param image: numpy array, channels_last, bhwc or bdhwc.
    :param output_size: output size, (height, width) or (depth, height, width).
    :param channels: output channels (1 or 3).
    :param interpolator: SimpleITK interpolator. E.g. sitk.sitkNearestNeighbor, sitk.sitkLinear.
    :return: modified image batch.
    """
    assert image.ndim in [4, 5]
    output_image = []
    for img in image:
        output_image.append(
            modify_size_channel(image=img, output_size=output_size, channels=channels, interpolator=interpolator))
    image = np.array(output_image)

    return image


def bound_by_labels(labels, scale=1.0, pad_square=True):
    """
    Gets a bounding box from a label image.

    :param labels: grey-level label image.
    :param scale: isotropic scaling of the bounding box.
    :param pad_square: True if padding the shorter side of the bounding box.

    :return: the bounding box with shape (2, 2). The first index is for the dimension (h, w), and the second index is
    for the lower and upper bounds (lb, ub). Cropping with the bounding box can be performed as: cropped = image[
    bound[0][0]:bound[0][1], bound[1][0]:bound[1][1]]
    """

    # Get bounding box
    _, contours, _ = cv2.findContours(labels.astype(np.uint8), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    points = np.concatenate(contours)
    bound = cv2.boundingRect(points)  # [x, y, w, h]

    # Convert format
    sz = np.array([bound[3], bound[2]])
    bound = np.array([[bound[1], bound[1]+bound[3]],
                      [bound[0], bound[0]+bound[2]]])

    # Rescale
    if scale != 1.0:
        for i in range(2):
            diff = (scale-1) * 0.5 * sz[i]
            bound[i][0] -= diff  # lower bound
            bound[i][1] += diff  # upper bound
    bound = bound.astype(np.int)

    # Pad square
    sz = bound[:, 1] - bound[:, 0]
    if pad_square and sz[0] != sz[1]:
        diff = sz.max() - sz.min()
        idx = sz.argmin()
        bound[idx][0] -= diff/2
        bound[idx][1] += diff/2 + diff % 2

    # Correct index
    for i in range(2):
        bound[i][0] = bound[i][0] if bound[i][0] >= 0 else 0
        bound[i][1] = bound[i][1] if bound[i][1] <= labels.shape[i] else labels.shape[i]

    return bound


def windowing(image, window_center, window_width):
    """Performs windowing on an image.

    :param numpy.array image: grey-level image.
    :param int/list/array window_center: for a scalar, it is used as the window center. For a list or array,
    its length must be a multiple of two to represent pairs of possible ranges. A random number generated between a
    pair is used as the window center. The pair used is the same as that of window_width if it is also a list or array.
    :param int/list/array window_width: for a scalar, it is used as the window width. For a list or array,
    its length must be a multiple of two to represent pairs of possible ranges. A random number generated between a
    pair is used as the window width. The pair used is the same as that of window_center if it is also a list or array.
    :return windowed image

    window_center and window_width must have the same length if both are not scalar.
    """

    idx = None  # Index to both window_center and window_width

    if np.isscalar(window_center):
        center = window_center
    else:
        if len(window_center) % 2 != 0:
            raise Exception('window_center must be a scalar or a list with even number of elements.')
        if idx is None:
            idx = np.random.choice(len(window_center)/2) * 2
        center = np.random.uniform(window_center[idx], window_center[idx+1])

    if np.isscalar(window_width):
        width = window_width
    else:
        if len(window_width) % 2 != 0:
            raise Exception('window_width must be a scalar or a list with even number of elements.')
        if idx is None:
            idx = np.random.choice(len(window_width)/2) * 2
        width = np.random.uniform(window_width[idx], window_width[idx+1])

    input_min = center - 0.5*width
    input_max = center + 0.5*width
    return rescale_int(image, in_range=(input_min, input_max), out_range=(0, 255))


class IntensityRescaleType(object):
    """Defining the enum for image intensity rescaling."""
    WINDOW = 0
    CLAHE = 1


def rescale_intensity(image, rescale_types, window_center=None, window_width=None):
    """ Based on the given rescale_list, performs image intensity rescaling such as windosing and histogram equalization.

    :param image: numpy array (hw or hwd).
    :param rescale_types: a list of IntensityRescaleEnum indicating which methods can be performed.
    :param int/list/array window_center: for a scalar, it is used as the window center. For a list or array,
    its length must be a multiple of two to represent pairs of possible ranges. A random number generated between a
    pair is used as the window center. The pair used is the same as that of window_width if it is also a list or array.
    :param int/list/array window_width: for a scalar, it is used as the window width. For a list or array,
    its length must be a multiple of two to represent pairs of possible ranges. A random number generated between a
    pair is used as the window width. The pair used is the same as that of window_center if it is also a list or array.
    :return: intensity-rescaled image (0-255).
    """

    if not rescale_types:
        raise ValueError('rescale_types is empty')

    idx = np.random.choice(len(rescale_types))
    if rescale_types[idx] == IntensityRescaleType.WINDOW:
        if window_center is None or window_width is None:
            return cv2.normalize(image, dst=np.array([]), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                 dtype=cv2.CV_8UC1)
        else:
            return windowing(image, window_center, window_width)
    elif rescale_types[idx] == IntensityRescaleType.CLAHE:
        return clahe(image, clip_limit=3.0)
    else:
        raise Exception('Unexpected type.')


def clahe(image, clip_limit=3.0, tile_size=None):
    """CLAHE for 2D and 3D images. Slice-by-slice for 3D images.

    :param image: numpy array (hw or dhw).
    :param clip_limit: clip limit for CLAHE.
    :param tile_size: tile size for CLAHE.
    :return: image after CLAHE.
    """

    # Rescale to 0-255
    image = cv2.normalize(image, dst=np.array([]), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # Perform CLAHE
    equalizer = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    if image.ndim == 2:  # hw
        image = equalizer.apply(image)
    elif image.ndim == 3:  # dhw
        output_image = []
        for img in image:  # Slice-by-slice
            output_image.append(equalizer.apply(img))
        image = np.array(output_image)
    else:
        raise Exception('Only support 2D and 3D images.')

    return image


def colormap_31():
    """Gets a colormap that has 31 independent colors from label 1 to 31."""
    cmap = np.concatenate([cm.tab20b_r(range(20)), cm.Set3_r(range(11))])
    return matplotlib_to_opencv_colormap(cmap)


def matplotlib_to_opencv_colormap(cmap, black_bg=True, unique_color=True):
    cmap = cmap * 255
    cmap = cmap[:, :3].astype(np.uint8)
    if unique_color:
        _, idx = np.unique(cmap, axis=0, return_index=True)
        cmap = cmap[np.sort(idx)]
    if black_bg:
        cmap = np.concatenate(([[0, 0, 0]], cmap))
    if len(cmap) < 256:
        cmap = np.concatenate((cmap, np.ones((256 - len(cmap), 3)) * 255))
    cmap = np.fliplr(cmap)
    cmap = np.array([cmap])
    return cmap


def map_label(label, cmap=colormap_31()):
    """Maps label to a given colormap.

    :param label: label.
    :param cmap: numpy.array cmap: colormap for mapping the label.
    :return: label with mapped color.
    """

    label = cv2.cvtColor(label.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return cv2.LUT(label, cmap).astype(np.uint8)


def label_overlay(image, label, image_alpha=0.5, label_alpha=0.5, cmap=colormap_31()):
    """Overlays label on image.

    :param numpy.array image: image.
    :param numpy.array label: label.
    :param float image_alpha: controls transparency of image.
    :param float label_alpha: controls transparency of label.
    :param numpy.array cmap: colormap for mapping the label.
    :return: image overlapped with label.
    :return: label with mapped color.
    """

    image = cv2.normalize(image, dst=np.array([]), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    label_new = map_label(label, cmap)

    return cv2.addWeighted(image, image_alpha, label_new, label_alpha, 0), label_new


def slice_padding(image, num_slices, padding='zero'):
    """Slice padding for multi-instance learning.

    :param numpy.array image: input image.
    :param int num_slices: number of target slices.
    :param str padding: padding method. 'zero', 'repeat', or 'linear'.
    :return: padded image.
    :rtype: numpy.array
    """

    padding_allowed = {'zero', 'repeat', 'linear'}
    if padding not in padding_allowed:
        raise ValueError('The `padding` argument must be one of "zero", "repeat". Received: ' + str(padding))

    num_slices = int(num_slices)  # Ensure integer
    image_padded = image

    len_image = len(image)

    if padding == 'repeat':

        mid_slice = int(len_image * 0.5)

        # Padding. Padding slices are from the middle part of the image. Looping is used as the extra slices required
        # can be larger than the entire image, though this should seldom happen.
        while num_slices != len(image_padded):
            extra_slices = num_slices - len(image_padded)

            start_idx = mid_slice - extra_slices/2
            start_idx = 0 if start_idx < 0 else start_idx

            end_idx = mid_slice + extra_slices/2 + extra_slices % 2
            end_idx = len_image if end_idx > len_image else end_idx

            pad = image[range(start_idx, end_idx)]
            image_padded = np.concatenate([image_padded, pad])

    elif padding == 'zero':

        pad = np.zeros((num_slices - len_image,) + image_padded.shape[1:], dtype=image_padded.dtype)
        image_padded = np.concatenate([image_padded, pad])

    elif padding == 'linear':

        image_shape = image_padded.shape
        axis = np.argmin(image_shape)  # Channel axis
        assert image_shape[axis] == 1, 'Slice interpolation only works for gray-level images.'

        # Remove channel --> z, y, x.
        image_padded = image_padded[:, 0, ...] if axis == 1 else image_padded[..., 0]

        # Interpolation using SimpleITK
        image_sitk = sitk.GetImageFromArray(image_padded)  # Transpose is taken care by SimpleITK
        spacing = np.asarray(image_sitk.GetSpacing())
        spacing[2] *= len_image / float(num_slices)
        size = np.asarray(image_sitk.GetSize())
        size[2] = num_slices
        resample = sitk.ResampleImageFilter()
        resample.SetSize(size)
        resample.SetOutputSpacing(spacing)
        resample.SetOutputOrigin(image_sitk.GetOrigin())
        image_sitk = resample.Execute(image_sitk)

        # Restore format
        image_padded = sitk.GetArrayFromImage(image_sitk)  # Transpose is taken care by SimpleITK
        image_padded = np.expand_dims(image_padded, axis=axis)

    return image_padded
