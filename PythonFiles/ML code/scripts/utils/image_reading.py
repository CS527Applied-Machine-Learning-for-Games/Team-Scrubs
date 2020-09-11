import os
import numpy as np
import warnings
import SimpleITK as sitk
import cv2

from utils.imageutils import pad_resize, resize_by_spacing

__author__ = 'Ken C. L. Wong'


def read_dicom_series(dicom_folder, correct_ct_val=0):
    """Read a DICOM series from a folder as a volume.

    :param dicom_folder: a folder containing files of a volume.
    :param correct_ct_val: if not 0, the given value is subtracted from the image intensity if necessary.
    :return: a SimpleITK image.
    """

    # Read a series as a volume
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_folder)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    image = reader.Execute()

    # Correct CT intensity value if required
    if correct_ct_val != 0:

        # Check conflict
        first_slice = sitk.ReadImage(file_names[0])
        if first_slice.HasMetaDataKey('0008|0060') and first_slice.GetMetaData('0008|0060') != 'CT':
            raise Exception('correct_ct_val != 0 for non-CT image.')

        # Correct CT intensity value
        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(image)
        if minmax.GetMinimum() >= 0:
            image -= correct_ct_val

    return image


def read_image(image_path, gray_scale=True, correct_ct_val=0, check_direction=True):
    """Reads an image by its file extension.

    :param image_path: an image file or a DICOM series folder.
    :param gray_scale: if True, a gray-scale image is returned.
    :param correct_ct_val: if not 0, the given value is subtracted from the image intensity if necessary.
    :param check_direction: if True, assert direction matrix == identity matrix for SimpleITK images.
    :return: a 2D (height, width) or 3D (depth, height, width) numpy array.
    """

    # DICOM series
    if os.path.isdir(image_path):
        image = read_dicom_series(image_path, correct_ct_val=correct_ct_val)
    # Single file
    elif any(image_path.endswith(ext) for ext in ['.nii', '.nii.gz', '.dcm']):

        image = sitk.ReadImage(image_path)

        # Correct CT intensity value if required
        if correct_ct_val != 0:

            # Check conflict
            if image.HasMetaDataKey('0008|0060') and image.GetMetaData('0008|0060') != 'CT':
                raise Exception('correct_ct_val != 0 for non-CT image.')

            # Correct CT intensity value
            minmax = sitk.MinimumMaximumImageFilter()
            minmax.Execute(image)
            if minmax.GetMinimum() >= 0:
                image -= correct_ct_val
    elif any(image_path.endswith(ext) for ext in ['.npy', '.npz']):
        image = np.load(image_path)
        if image_path.endswith('.npz'):
            image = image[image.files[0]]
    else:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # image = sitk.ReadImage(image_path)
        # image = sitk.GetArrayFromImage(image)
        if gray_scale and image.ndim == 3:
            image = image.mean(axis=-1).astype(image.dtype)
        # elif not gray_scale and image.ndim == 2:
        #     image = np.expand_dims(image, -1).repeat(3, axis=-1)

    if isinstance(image, sitk.Image):

        # Check direction
        if check_direction:
            direction = image.GetDirection()
            if tuple(np.eye(image.GetDimension()).flatten()) != direction:
                raise Exception('Non-identity direction: ', direction)

        # Change to 2D if necessary
        if image.GetDimension() == 3 and image.GetDepth() == 1:
            image = sitk.Extract(image, image.GetSize()[:2] + (0,), np.zeros(3, np.int))

        # Resize to isotropic spacing
        if np.unique(image.GetSpacing()).size != 1:
            image = resize_by_spacing(image)

        # Change to numpy array
        image = sitk.GetArrayFromImage(image)

    return image


def load_images(image_folder=None, image_list=None, output_size=(256, 256), hw_only=False, precision=np.int16,
                correct_ct_val=0, check_direction=True):
    """Loads images and returns a 4D or 5D numpy array in channels_last format.
    This function automatic determines the image types to be read.

    :param image_folder: a folder containing the images. DICOM series folders are not handled.
    :param image_list: a list of absolute image paths. DICOM series folders are handled.
    :param output_size: output image size, (height, width) or (depth, height, width).
    :param hw_only: if True, only zero-pad the height and width.
    :param precision: desired precision.
    :param correct_ct_val: if not 0, the given value is subtracted from the image intensity if necessary.
    :param check_direction: if True, assert direction matrix == identity matrix for SimpleITK images.
    :return: a 4D (bhwc) or 5D (bdhwc) numpy array in channels_last format.
    """

    # Image extensions supported
    ext_list = ['.nii', '.nii.gz', '.dcm', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']

    # Get image_list
    if image_list is None:
        if image_folder is None:
            raise RuntimeError('Both image_folder and image_list are None.')
        image_list = [img for img in os.listdir(image_folder) if any(img.endswith(ext) for ext in ext_list)]
        image_list = [os.path.join(image_folder, img) for img in image_list]

    image_list = sorted(image_list)

    # Read images and put them in a numpy array
    data = []
    for img in image_list:

        image = read_image(img, correct_ct_val=correct_ct_val, check_direction=check_direction)

        if image is None:
            warnings.warn('Cannot read image %s' % img)
            continue

        image = pad_resize(image, output_size, hw_only=hw_only)
        image = np.expand_dims(image, axis=image.ndim)

        data.append(image)

    data = np.array(data, dtype=precision)

    return data
