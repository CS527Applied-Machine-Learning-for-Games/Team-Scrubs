import numpy as np
import os
import cv2
from skimage.measure import block_reduce

def _histeq(images):
    ''' Histogram equalize the images 
    '''
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    
    if images.ndim == 2:
        histeq_images = clahe.apply(images)
    else:
        histeq_images = np.zeros_like(images)
        for idx in range(len(images)):
            histeq_images[idx] = clahe.apply(images[idx])

    return histeq_images

def preprocess(images, image_size=512, segmentation=False):
    ''' Appropriately preprocess the images. If image, run histeq and resize.
        Else, only resize
    '''

    resize_factor = int(np.shape(images)[1] * 1./image_size)

    if images.ndim == 2:
        if segmentation == False:
            histeq_images = _histeq(images)
            small_images = block_reduce(histeq_images, (resize_factor, resize_factor), np.max)
        else:
            small_images = block_reduce(images, (resize_factor, resize_factor), np.max)
            # replace all the non-zero entries with 1 in the segmentation image
            # this means we arer treating all the segmentations in the same way
            # and not differentiating the different types of lines for the 
            # UNet training
            small_images = 1*(small_images > 0)

    elif images.ndim == 3:
        if segmentation == False:
            histeq_images = _histeq(images)
            small_images = block_reduce(histeq_images, (1, resize_factor, resize_factor), np.max)
        else:
            small_images = block_reduce(images, (1, resize_factor, resize_factor), np.max)
            # replace all the non-zero entries with 1 in the segmentation image
            # this means we arer treating all the segmentations in the same way
            # and not differentiating the different types of lines for the 
            # UNet training
            small_images = 1*(small_images > 0)
    return small_images


