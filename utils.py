#!/usr/bin/env python

import math
import cv2
from parameters import params
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import tensorflow as tf


def get_filename(path):
    return path.split('/')[-1]

def dim(patch,stride,length):
    rest = length - patch
    return 1 + math.floor(rest/stride)


def preprocess_images(images):
    import tensorflow as tf
    from parameters import params
    # Crop in y direction
    images = images[:,params['crop_y_top']:(160-params['crop_y_bottom']),:,:]
    images = tf.image.rgb_to_grayscale(images)
    images = tf.image.resize_images(images, [66,200])
    images = tf.map_fn(lambda image: tf.image.per_image_standardization(image), images)
    return images
