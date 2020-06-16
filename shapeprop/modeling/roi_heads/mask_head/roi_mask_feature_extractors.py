# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import torch
from torch import nn
from torch.nn import functional as F

from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from shapeprop.modeling import registry
from shapeprop.modeling.poolers import Pooler
from shapeprop.modeling.make_layers import make_conv3x3
from shapeprop.modeling.utils import cat


registry.ROI_MASK_FEATURE_EXTRACTORS.register(
    "ResNet50Conv5ROIFeatureExtractor", ResNet50Conv5ROIFeatureExtractor
)


@registry.ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNFPNFeatureExtractor")
class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()
        self.cfg = cfg.clone()
        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        next_feature = input_size + (input_size if self.cfg.MODEL.SHAPEPROP_ON else 0)
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

        if self.cfg.MODEL.SHAPEPROP_ON:
            self.encoder = nn.Sequential(
                make_conv3x3(1, input_size, dilation=dilation, stride=1, use_gn=use_gn),
                nn.ReLU(True),
                make_conv3x3(input_size, input_size, dilation=dilation, stride=1, use_gn=use_gn),
                nn.ReLU(True),
                make_conv3x3(input_size, input_size, dilation=dilation, stride=1, use_gn=use_gn),
                nn.ReLU(True),
                make_conv3x3(input_size, input_size, dilation=dilation, stride=1, use_gn=use_gn)
            )

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        
        if self.cfg.MODEL.SHAPEPROP_ON:
            eps = 1e-5
            shape_activation = cat([v.get_field('shape_activation') for v in proposals]).unsqueeze(1)
            # normalize
            shape_activation = F.relu(shape_activation)
            norm_factors, _ = shape_activation.view(shape_activation.shape[0], 1, -1).max(2)
            shape_activation /= (norm_factors.unsqueeze(2).unsqueeze(3) + eps)
            shape_activation = self.encoder(shape_activation)
            # fuse shape_activation into input
            shape_activation = F.interpolate(shape_activation, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = cat([x, shape_activation], 1)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x


def make_roi_mask_feature_extractor(cfg, in_channels):
    func = registry.ROI_MASK_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
