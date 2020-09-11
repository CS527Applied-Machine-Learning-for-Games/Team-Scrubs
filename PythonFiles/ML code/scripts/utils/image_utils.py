import os
import sys
import cv2
import numpy as np
import SimpleITK as sitk
from PIL import Image
# import pydicom

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
    :param gray_scale: if True, a gray-scale image is returned by computing the average of all channels.
    If a real number, a gray-scale image is returned by extracting the corresponding channel.
    :param correct_ct_val: if not 0, the given value is subtracted from the image intensity if necessary.
    :param check_direction: if True, assert direction matrix == identity matrix for SimpleITK images.
    :return: a 2D (height, width) or 3D (depth, height, width) numpy array.
    """

    # DICOM series
    if os.path.isdir(image_path):
        image = read_dicom_series(image_path, correct_ct_val=correct_ct_val)
        return sitk_to_array(image, check_direction, gray_scale)

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

        return sitk_to_array(image, check_direction, gray_scale)

    elif any(image_path.endswith(ext) for ext in ['.npy', '.npz']):
        image = np.load(image_path)
        if image_path.endswith('.npz'):
            image = image[image.files[0]]
        return image

    else:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image.ndim == 3:
            if isinstance(gray_scale, bool):
                if gray_scale:
                    image = image.mean(axis=-1).astype(image.dtype)
            elif np.isreal(gray_scale):
                image = image[..., gray_scale]
        return image
def sitk_to_array(image, check_direction, gray_scale):

    # Check direction
    if check_direction:
        direction = image.GetDirection()
        if tuple(np.eye(image.GetDimension()).flatten()) != direction:
            raise Exception('Non-identity direction: ', direction)

    print(image.GetDimension())
    print(image.GetDepth())
    print(image.GetSize())
    # Change to 2D if necessary
    if image.GetDimension() == 3 and image.GetDepth() == 1:
        image = sitk.Extract(image, image.GetSize()[:2] + (0,), np.zeros(3, np.int))

    # # Resize to isotropic spacing
    # if np.unique(image.GetSpacing()).size != 1:
    #     image = resize_by_spacing(image)

    # Change to gray-scale if desired
    if image.GetNumberOfComponentsPerPixel() != 1:
        image = sitk.GetArrayFromImage(image)  # nD to (n+1)D, pixel vectors become the last dimension
        if isinstance(gray_scale, bool):
            if gray_scale:
                image = image.mean(axis=-1).astype(image.dtype)
        elif np.isreal(gray_scale):
            image = image[..., gray_scale]
    else:
        image = sitk.GetArrayFromImage(image)

    return image

# def read_image(image_name):
#     '''
#     Load the image passed
#
#     Args:
#         image_name: string containing location and image name
#     Returns:
#         image: numpy array with the image loaded
#     '''
#     print (os.path.splitext(os.path.basename(image_name))[1])
#
#     if (os.path.splitext(os.path.basename(image_name))[1]=='.dcm'):
#         # d = dicom.read_file(image_name)
#         d = pydicom.dcmread(image_name)
#         image = np.asarray(d.pixel_array, dtype=np.uint8)
#     else:
#         image = np.asarray(Image.open(image_name), dtype=np.uint8)
#
#
#     if(np.ndim(image) > 2):
#         image = image[:, :, 0]
#
#     return image
