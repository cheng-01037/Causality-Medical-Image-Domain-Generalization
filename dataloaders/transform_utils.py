"""
Utilities for image transforms, part of the code base credits to Dr. Jo Schlemper
"""
from os.path import join
import torch
import numpy as np
import torchvision.transforms as deftfx
import dataloaders.image_transforms as myit
import copy
import math


my_augv = {
'flip'      : { 'v':False, 'h':False, 't': False, 'p':0.25 },
'affine'    : {
  'rotate':20,
  'shift':(15,15),
  'shear': 20,
  'scale':(0.5, 1.5),
},
'elastic'   : {'alpha':20,'sigma':5}, # medium
'reduce_2d': True,
'gamma_range': (0.2, 1.8),
'noise' : {
    'noise_std': 0.15,
    'clip_pm1': False
    },
'bright_contrast': {
    'contrast': (0.60, 1.5),
    'bright': (-10,  10)
    }
}

tr_aug = {
    'aug': my_augv
}

def get_geometric_transformer(aug, order=3):
    affine     = aug['aug'].get('affine', 0)
    alpha      = aug['aug'].get('elastic',{'alpha': 0})['alpha']
    sigma      = aug['aug'].get('elastic',{'sigma': 0})['sigma']
    flip       = aug['aug'].get('flip', {'v': True, 'h': True, 't': True, 'p':0.125})

    tfx = []
    if 'flip' in aug['aug']:
        tfx.append(myit.RandomFlip3D(**flip))

    if 'affine' in aug['aug']:
        tfx.append(myit.RandomAffine(affine.get('rotate'),
                                     affine.get('shift'),
                                     affine.get('shear'),
                                     affine.get('scale'),
                                     affine.get('scale_iso',True),
                                     order=order))

    if 'elastic' in aug['aug']:
        tfx.append(myit.ElasticTransform(alpha, sigma))

    input_transform = deftfx.Compose(tfx)
    return input_transform

def get_intensity_transformer(aug):

    def gamma_tansform(img):
        gamma_range = aug['aug']['gamma_range']
        if isinstance(gamma_range, tuple):
            gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
            cmin = img.min()
            irange = (img.max() - cmin + 1e-5)

            img = img - cmin + 1e-5
            img = irange * np.power(img * 1.0 / irange,  gamma)
            img = img + cmin

        elif gamma_range == False:
            pass
        else:
            raise ValueError("Cannot identify gamma transform range {}".format(gamma_range))
        return img

    def brightness_contrast(img):
        '''
        Chaitanya,K. et al. Semi-Supervised and Task-Driven data augmentation,864in: International Conference on Information Processing in Medical Imaging,865Springer. pp. 29–41.
        '''
        cmin, cmax = aug['aug']['bright_contrast']['contrast']
        bmin, bmax = aug['aug']['bright_contrast']['bright']
        c = np.random.rand() * (cmax - cmin) + cmin
        b = np.random.rand() * (bmax - bmin) + bmin
        img_mean = img.mean()
        img = (img - img_mean) * c + img_mean + b
        return img

    def zm_gaussian_noise(img):
        """
        zero-mean gaussian noise
        """
        noise_sigma = aug['aug']['noise']['noise_std']
        noise_vol = np.random.randn(*img.shape) * noise_sigma
        img = img + noise_vol

        if aug['aug']['noise']['clip_pm1']: # if clip to plus-minus 1
            img = np.clip(img, -1.0, 1.0)
        return img

    def compile_transform(img):
        # bright contrast
        if 'bright_contrast' in aug['aug'].keys():
            img = brightness_contrast(img)

        # gamma
        if 'gamma_range' in aug['aug'].keys():
            img = gamma_tansform(img)

        # additive noise
        if 'noise' in aug['aug'].keys():
            img = zm_gaussian_noise(img)

        return img

    return compile_transform


def transform_with_label(aug, add_pseudolabel = False):
    """
    Doing image geometric transform
    Proposed image to have the following configurations
    [H x W x C + CL]
    Where CL is the number of channels for the label. It is NOT a one-hot thing
    """

    geometric_tfx = get_geometric_transformer(aug)
    intensity_tfx = get_intensity_transformer(aug)

    def transform(comp, c_label, c_img, nclass, is_train, use_onehot = False):
        """
        Args
            comp:               a numpy array with shape [H x W x C + c_label]
            c_label:            number of channels for a compact label. Note that the current version only supports 1 slice (H x W x 1)
            nc_onehot:          -1 for not using one-hot representation of mask. otherwise, specify number of classes in the label
            is_train:           whether this is the training set or not. If not, do not perform the geometric transform
        """
        comp = copy.deepcopy(comp)
        if (use_onehot is True) and (c_label != 1):
            raise NotImplementedError("Only allow compact label, also the label can only be 2d")
        assert c_img + 1 == comp.shape[-1], "only allow single slice 2D label"

        if is_train is True:
            _label = comp[..., c_img ]
            # compact to onehot
            _h_label = np.float32(np.arange( nclass ) == (_label[..., None]) )
            comp = np.concatenate( [comp[...,  :c_img ], _h_label], -1 )
            comp = geometric_tfx(comp)
            # round one_hot labels to 0 or 1
            t_label_h = comp[..., c_img : ]
            t_label_h = np.rint(t_label_h)
            t_img = comp[..., 0 : c_img ]

        # intensity transform
        t_img = intensity_tfx(t_img)

        if use_onehot is True:
            t_label = t_label_h
        else:
            t_label = np.expand_dims(np.argmax(t_label_h, axis = -1), -1)
        return t_img, t_label

    return transform


