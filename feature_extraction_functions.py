import helpers as hp
import numpy as np
from skimage import color
from skimage import transform
from skimage import feature

# ==================================================================== #
# Features extraction
# ==================================================================== #

def extract_features(img):
    """
    Extract 6-dimensional features consisting of average RGB color as well as variance
    :param img: image considered
    :return: features of the image
    """
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

def extract_features_2d(img):
    """
    Extract 2-dimensional features consisting of average gray color as well as variance
    :param img: image considered
    :return: features of the image
    """
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

#
def extract_img_features(filename, patch_size):
    """
    Extract features for a given image per patches
    :param filename: filename of the image
    :param patch_size: size of the patch
    :return: features of the image for each patch
    """
    img = hp.load_image(filename)
    img_patches = hp.img_crop(img, patch_size, patch_size)
    X = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))])
    return X


def get_canny_edge_features(img):
    """
    Obtain features with a Canny Edge Detector
    :param img: considered image
    :return: canny edge features
    """
    img = hp.img_float_to_uint8(img)
    greylvl = color.rgb2gray(img)
    sigma = 2
    canny_edges = feature.canny(greylvl, sigma)

    return canny_edges

# ==================================================================== #
# Getting features from color images in rgb and hsv space.
# ==================================================================== #

def extract_hsv2(img):
    """
    Extract hsv features for a given image
    :param img: image considered
    :return: hsv features
    """
    tmp = color.rgb2hsv(img)
    return extract_features(tmp)

def extract_hsv(img):
    """
    Extract HSV features consisting of average HSV color
    :param img: image considered
    :return: HSV mean features
    """
    tmp = color.rgb2hsv(img)
    
    r_mean = np.mean(tmp[:, :, 0])
    g_mean = np.mean(tmp[:, :, 1])
    b_mean = np.mean(tmp[:, :, 2])
    
    feat = np.append(r_mean, g_mean)
    feat = np.append(feat, b_mean)

    return feat

def extract_rgb(img):
    """
    Extract RGB features consisting of average RGB color
    :param img: image considered
    :return: RGB mean features
    """
    r_mean = np.mean(img[:, :, 0])
    g_mean = np.mean(img[:, :, 1])
    b_mean = np.mean(img[:, :, 2])
    
    feat = np.append(r_mean, g_mean)
    feat = np.append(feat, b_mean)

    return feat


# ==================================================================== #
# Data augmentation
# ==================================================================== #

def rotate_imgs(images, angle):
    """
    Rotate an image for a specific angle
    :param images: image considered
    :param angle: angle of rotation
    :return: rotated image
    """
    rt_imgs = []
    for i in range(len(images)):
        rt_imgs.append(transform.rotate(images[i], angle, resize=False, mode='reflect'))
    return rt_imgs

def resize_imgs(images, size):
    """
    Resize a list of images
    :param images: list of targeted images
    :param size: targeted size
    :return: resized images
    """
    rt_imgs = []
    for i in range(len(images)):
        rt_imgs.append(transform.resize(images[i], (size, size), mode='reflect'))
    return rt_imgs

def data_augmentation(images):
    """
    This method increase the number of initial images by adding rotated version of the formers.
    This allows to increase the number of images used for training
    :param images: original images
    :return: original + rotated images
    """
    tmp = images
    images = np.concatenate((images, rotate_imgs(tmp, 30)), axis=0)
    images = np.concatenate((images, rotate_imgs(tmp, 45)), axis=0)
    images = np.concatenate((images, rotate_imgs(tmp, 60)), axis=0)
    return images
