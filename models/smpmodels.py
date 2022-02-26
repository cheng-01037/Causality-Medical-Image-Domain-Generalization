"""
Off-the-shelf segmentation model from https://github.com/qubvel/segmentation_models.pytorch
"""

import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from pdb import set_trace
import sys
sys.path.append('your path to the segmentation_models_pytorch package')
from segmentation_models_pytorch import Unet

class EfficientUNet(torch.nn.Module):
    def __init__(self, nclass, classifier, return_feature = False, backbone_name = 'efficientnet-b2', in_channels = 3):
        super(EfficientUNet, self).__init__()
        self.model = Unet(encoder_name = backbone_name,
                encoder_weights = None,
                in_channels = in_channels,
                classes = nclass,
                activation = None
                )
        self.return_feature = return_feature

    def forward(self, x, volatile_return_feature = False):
        self.enc_features = self.model.encoder(x)
        self.decoder_output = self.model.decoder(*self.enc_features)

        masks = self.model.segmentation_head(self.decoder_output)
        if self.return_feature or volatile_return_feature:
            return masks, self.decoder_output
        else:
            return masks, []

def efficient_unet(nclass, in_channel, gpu_ids = [], return_feature = False, **kwargs):
    return EfficientUNet(nclass=nclass, classifier = None, return_feature = return_feature, **kwargs).cuda(gpu_ids[0])


