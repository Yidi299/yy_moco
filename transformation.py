#!/usr/bin/env python3
#coding:utf-8
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageOps, ImageEnhance


def resize_short(img, target_size):
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))
    img = cv2.resize(img, (resized_width, resized_height), interpolation = cv2.INTER_LINEAR)
    return img


def crop_image(img, target_size, center=True):
    width, height = img.shape[1], img.shape[0]
    size = target_size
    if center == True:
        w_start = (width - size) // 2
        h_start = (height - size) // 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end]
    return img


def random_crop(img, size, scale=[0.08,1.0], ratio=[3./4.,4./3.],
                flip_p=0, rot90_p=0, tmeta=None, keep_wh_ratio=False):
    if isinstance(img, list):
        assert len(img) > 1
        img1 = img[0]
        for img2 in img[1:]:
            assert img1.shape == img2.shape
    else:
        img = [img]
    aspect_ratio = np.sqrt(np.random.uniform(*ratio))
    if np.random.rand() < 0.5:
        aspect_ratio = 1.0 / aspect_ratio
    ws = 1. * aspect_ratio
    hs = 1. / aspect_ratio
    
    imgsize = img_w, img_h = img[0].shape[1], img[0].shape[0]
    rand_scale = np.random.uniform(*scale)
    if keep_wh_ratio:
        rand_scale = np.sqrt(rand_scale)
        target_w, target_h = img_w * rand_scale, img_h * rand_scale
    else:
        target_w = target_h = np.sqrt(img_w * img_h * rand_scale)
    w = min(img_w, max(1, int(0.5 + target_w * ws)))
    h = min(img_h, max(1, int(0.5 + target_h * hs)))
    
    i = np.random.randint(0, imgsize[0] - w + 1)
    j = np.random.randint(0, imgsize[1] - h + 1)
    if tmeta is not None:
        assert tmeta in list(range(9))
        # 1  6  2
        # 5  0  7
        # 4  8  3
        if   tmeta == 0:
            pass
        elif tmeta == 1:
            i, j = 0, 0
        elif tmeta == 2:
            i, j = imgsize[0] - w, 0
        elif tmeta == 3:
            i, j = imgsize[0] - w, imgsize[1] - h
        elif tmeta == 4:
            i, j = 0, imgsize[1] - h
        elif tmeta == 5:
            i, j = 0, (imgsize[1] - h)//2
        elif tmeta == 6:
            i, j = (imgsize[0] - w)//2, 0
        elif tmeta == 7:
            i, j = imgsize[0] - w, (imgsize[1] - h)//2
        elif tmeta == 8:
            i, j = (imgsize[0] - w)//2, imgsize[1] - h
    img = [img1[j:j+h, i:i+w] for img1 in img]
    img = [cv2.resize(img1, (size, size), interpolation = cv2.INTER_LINEAR)
           for img1 in img]
    
    if flip_p > 0:
        if np.random.rand() < flip_p:
            img = [np.fliplr(img1) for img1 in img]
    if rot90_p > 0:
        p = np.random.rand()
        if p < 1-rot90_p:                 # +0
            pass
        elif p < 1-rot90_p+rot90_p*0.333: # +90
            img = [np.rot90(img1) for img1 in img]
        elif p < 1-rot90_p+rot90_p*0.666: # +180
            img = [np.rot90(img1, 2) for img1 in img]
        elif p < 1-rot90_p+rot90_p*0.999: # +270
            img = [np.rot90(img1, 3) for img1 in img]
    
    if len(img) == 1:
        img = img[0]
    return img


def distort_color(img, color_pca_p=0):
    def random_brightness(img, lower=0.5, upper=1.5):
        img = np.clip(img, 0.0, 1.0)
        e = np.random.uniform(lower, upper)
        # zero = np.zeros([1] * len(img.shape), dtype=img.dtype)
        return img * e # + zero * (1.0 - e)

    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        gray = np.mean(img[0]) * 0.299 + np.mean(img[1]) * 0.587 + np.mean(img[2]) * 0.114
        mean = np.ones([1] * len(img.shape), dtype=img.dtype) * gray
        return img * e + mean * (1.0 - e)

    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        gray = img[0] * 0.299 + img[1] * 0.587 + img[2] * 0.114
        gray = np.expand_dims(gray, axis=0)
        return img * e + gray * (1.0 - e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)

    if img.dtype == 'uint8' and img.shape[-1] == 3:
        img = img.astype('float32').transpose((2, 0, 1)) * (1.0 / 255)

    assert img.shape[0] == 3 and img.dtype == 'float32'

    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)

    if color_pca_p > 0:
        eigvec = np.array([ [-0.5675,  0.7192,  0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948,  0.4203] ], dtype='float32')
        alpha = (np.random.randn(3) * color_pca_p).astype('float32')
        eigval = np.array([0.2175, 0.0188, 0.0045], dtype='float32')
        rgb = np.sum(eigvec * alpha * eigval, axis=1)
        img += rgb.reshape([3, 1, 1])

    #img = np.clip(img, 0.0, 1.0)

    return img


def to_gray(img):
    gray = img[0] * 0.299 + img[1] * 0.587 + img[2] * 0.114
    return np.stack([gray, gray, gray])


def adjust_hue(img, hue_factor_p=0.1):
    hue_factor = np.random.uniform(-hue_factor_p, hue_factor_p)
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))
    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img
    h, s, v = img.convert('HSV').split()
    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')
    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def blur(img, factor_p=[.1, 2.]):
    factor = np.random.uniform(*factor_p)
    img = img.filter(ImageFilter.GaussianBlur(radius=factor))
    return img


def rotate_image(img, p=0.5):
    if np.random.rand() < 1-p:
        return img
    # very slow
    angle = np.random.randint(-15, 16)
    if angle > 3 or angle < -3:
        img = Image.fromarray(img)
        img = img.rotate(angle, expand=1)
        img = np.array(img, dtype='uint8')
    return img


img_mean = np.array([0.485, 0.456, 0.406], dtype='float32').reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225], dtype='float32').reshape((3, 1, 1))

def std_image(img):
    img -= img_mean
    img *= (1.0 / img_std)
    return img

