import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os,sys
from PIL import Image
from skimage import color
from skimage import feature
import re

# Helper functions

def load_image(infilename):
    """
    Load an image
    :param infilename:
    :return:
    """
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    """
    Convert float image to uint8
    :param img:
    :return: image in uint8
    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def concatenate_images(img, gt_img):
    """
    Concatenate an image with its groundtruth
    :param img: considered image
    :param gt_img: groundtruth image
    :return:
    """
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    """
    Crop an image into patches
    :param im: considered image
    :param w: width of the patch
    :param h: height of the patch
    :return: list of created patches
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


# Compute features for each image patch


def value_to_class_custom(patch):
    """
    Determine if the patch is considered as road or not by looking at each pixels
    :param patch: patch considered
    :return: 1 or -1 if the patch can be considered as road or not
    """
    size = patch.shape[0] * patch.shape[1]
    threshold = 1
    if (np.count_nonzero(patch) / size) < threshold:
        return -1
    return 1


def label_to_img(imgwidth, imgheight, w, h, labels):
    """
    Convert array of labels to an image
    :param imgwidth: width of the image
    :param imgheight: height of the image
    :param w: width of the patch
    :param h: height of the patch
    :param labels: arrays of labels (-1 and +1)
    :return: resulting image
    """
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            tmp_lab = labels[idx]
            
            to_print = 0
            if tmp_lab == 1:
                to_print = 1
                
            im[j:j+w, i:i+h] = to_print
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    """
    Overlay an image on top of another
    :param img: background image
    :param predicted_img: overlay
    :return: mix of images
    """
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

#
def patch_to_label(patch):
    """
    Assign a label (1 or 0) to a patch for submission
    :param patch: patch
    :return: label
    """
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.mean(patch)

    if df > foreground_threshold:
        return 1
    else:
        return 0
    
def natural_sort(l):
    """
    Sort a list of folder
    :param l: list of folders
    :return: sorted list of folder
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)