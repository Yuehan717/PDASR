import random

import numpy as np
import skimage.color as sc

import torch

def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret

def np_RGB2YCrCb(img):
    # Implement the conversion of RGB 2 YCrCb channels for numpy arrays, keep the float value
    # The formulation refers to Open CV
    # print(img.dtype)
    if img.dtype == np.uint8:
        delta = 128
    elif img.dtype == np.uint16:
        delta = 32768
    elif img.dtype == np.float32:
        delta = 0.5
        
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    Y = 0.299*R + 0.587*G + 0.114*B
    Cr = (R - Y)*0.713 + delta
    Cb = (B - Y)*0.564 + delta
    out = np.stack((Y,Cr,Cb),axis=2)
    return out

def set_channel(*args, n_channels=3, train=True):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        
        c = img.shape[2]
        if n_channels == 1 and c == 3: ## RGB image but only process on Y channel
            # print()
            # print("Shape before color change in set_channel()", img.shape)
            img = np_RGB2YCrCb(img)
            # img = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
            # print(img[0,0,0])
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)
        
        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1))) # why? what's the requirement of img tensor
        # print("np_transpose shape {}".format(np_transpose.shape))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args]

