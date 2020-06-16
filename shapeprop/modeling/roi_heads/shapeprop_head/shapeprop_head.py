import torch
import torch.nn.functional as F
from torch import nn

from shapeprop.modeling.poolers import Pooler
from shapeprop.modeling.make_layers import make_conv1x1, make_conv3x3, group_norm
from shapeprop.modeling.utils import cat

from .loss import make_propagating_loss_evaluator 


class ShapePropFeatureExtractor(nn.Module):

    def __init__(self, cfg, in_channels):
        super(ShapePropFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_SHAPEPROP_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_SHAPEPROP_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_SHAPEPROP_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_SHAPEPROP_HEAD.USE_GN
        layers = cfg.MODEL.ROI_SHAPEPROP_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_SHAPEPROP_HEAD.DILATION

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "shapeprop_feature{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x

class ShapePropPredictor(nn.Module):

    def __init__(self, cfg, in_channels):
        super(ShapePropPredictor, self).__init__()
        use_gn = cfg.MODEL.ROI_SHAPEPROP_HEAD.USE_GN
        self.cls = make_conv1x1(
            in_channels, 
            cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            use_gn)

    def forward(self, x):
        logits = self.cls(x)
        return logits

class ShapePropWeightRegressor(nn.Module):

    def __init__(self, cfg, in_channels):
        super(ShapePropWeightRegressor, self).__init__()
        use_gn = cfg.MODEL.ROI_SHAPEPROP_HEAD.USE_GN
        self.reg = make_conv1x1(
            in_channels, 
            (1 if cfg.MODEL.ROI_SHAPEPROP_HEAD.CHANNEL_AGNOSTIC else cfg.MODEL.ROI_SHAPEPROP_HEAD.LATENT_DIM) * 9,
            use_gn)

    def forward(self, x):
        weights = self.reg(x)
        return torch.sigmoid(weights)

class MessagePassing(nn.Module):

    def __init__(self, k=3, max_step=-1, sym_norm=False):
        super(MessagePassing, self).__init__()
        self.k = k
        self.size = k * k
        self.max_step = max_step
        self.sym_norm = sym_norm

    def forward(self, input, weight):
        eps = 1e-5
        n, c, h, w = input.size()
        wc = weight.shape[1] // self.size
        weight = weight.view(n, wc, self.size, h * w)
        if self.sym_norm:
            # symmetric normalization D^(-1/2)AD^(-1/2)
            D = torch.pow(torch.sum(weight, dim=2) + eps, -1/2).view(n, wc, h, w)
            D = F.unfold(D, kernel_size=self.k, padding=self.padding).view(n, wc, self.window, h * w) * D.view(n, wc, 1, h * w)
            norm_weight = D * weight
        else:
            # random walk normalization D^(-1)A
            norm_weight = weight / (torch.sum(weight, dim=2).unsqueeze(2) + eps)
        x = input
        for i in range(max(h, w) if self.max_step < 0 else self.max_step):
            x = F.unfold(x, kernel_size=self.k, padding=1).view(n, c, self.size, h * w)
            x = (x * norm_weight).sum(2).view(n, c, h, w)
        return x

class ShapePropEncoder(nn.Module):

    def __init__(self, cfg, in_channels):
        super(ShapePropEncoder, self).__init__()
        use_gn = cfg.MODEL.ROI_SHAPEPROP_HEAD.USE_GN
        latent_dim = cfg.MODEL.ROI_SHAPEPROP_HEAD.LATENT_DIM
        dilation = cfg.MODEL.ROI_SHAPEPROP_HEAD.DILATION
        self.encoder = nn.Sequential(
            make_conv3x3(in_channels, latent_dim, dilation=dilation, stride=1, use_gn=use_gn),
            nn.ReLU(True),
            make_conv3x3(latent_dim, latent_dim, dilation=dilation, stride=1, use_gn=use_gn),
            nn.ReLU(True),
            make_conv3x3(latent_dim, latent_dim, dilation=dilation, stride=1, use_gn=use_gn)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

class ShapePropDecoder(nn.Module):

    def __init__(self, cfg, out_channels):
        super(ShapePropDecoder, self).__init__()
        use_gn = cfg.MODEL.ROI_SHAPEPROP_HEAD.USE_GN
        latent_dim = cfg.MODEL.ROI_SHAPEPROP_HEAD.LATENT_DIM
        dilation = cfg.MODEL.ROI_SHAPEPROP_HEAD.DILATION
        self.decoder = nn.Sequential(
            make_conv3x3(latent_dim, latent_dim, dilation=dilation, stride=1, use_gn=use_gn),
            nn.ReLU(True),
            make_conv3x3(latent_dim, latent_dim, dilation=dilation, stride=1, use_gn=use_gn),
            nn.ReLU(True),
            make_conv3x3(latent_dim, out_channels, dilation=dilation, stride=1, use_gn=use_gn)
        )

    def forward(self, embedding):
        x = self.decoder(embedding)
        return x

class ShapeProp(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(ShapeProp, self).__init__()
        self.cfg = cfg.clone()
        # activating saliency
        self.feature_extractor_activating = ShapePropFeatureExtractor(self.cfg, in_channels)
        self.predictor = ShapePropPredictor(self.cfg, self.feature_extractor_activating.out_channels)
        # propagating saliency
        self.feature_extractor_propagating = (self.feature_extractor_activating
            if self.cfg.MODEL.ROI_SHAPEPROP_HEAD.SHARE_FEATURE
            else ShapePropFeatureExtractor(self.cfg, in_channels))
        self.propagation_weight_regressor = ShapePropWeightRegressor(self.cfg, self.feature_extractor_propagating.out_channels)
        self.encoder = ShapePropEncoder(self.cfg, 1)
        self.message_passing = MessagePassing(sym_norm=self.cfg.MODEL.ROI_SHAPEPROP_HEAD.USE_SYMMETRIC_NORM)
        self.decoder = ShapePropDecoder(self.cfg, 1)
        self.propagating_loss_evaluator = make_propagating_loss_evaluator(cfg)

    def activating_saliency(self, features, proposals, targets=None):
        x = self.feature_extractor_activating(features, proposals)
        saliency = self.predictor(x)
        labels = []
        for saliency_per_image, proposals_per_image in zip(saliency.split([len(v) for v in proposals], 0), proposals):
            labels_per_image = proposals_per_image.get_field('labels')
            labels.append(labels_per_image)
            saliency_per_image = torch.stack([v[l] for v, l in zip(saliency_per_image, labels_per_image)])
            proposals_per_image.add_field('saliency', saliency_per_image)
        # inference mode
        if not self.training:
            return x, proposals, saliency, None
        # compute loss
        labels = cat(labels, dim=0) 
        num_batch, num_channel, h, w = saliency.shape
        class_logits = saliency.view(num_batch, num_channel, h * w).mean(2)
        loss_activating = F.cross_entropy(class_logits, labels)
        return x, proposals, saliency, loss_activating

    def propagating_saliency(self, features, proposals, targets=None):
        if self.cfg.MODEL.ROI_SHAPEPROP_HEAD.SHARE_FEATURE:
            x = features
        else:
            x = self.feature_extractor_propagating(features, proposals)
        weights = self.propagation_weight_regressor(x)
        saliency = cat([v.get_field('saliency') for v in proposals]).unsqueeze(1)
        embedding = self.encoder(saliency)
        embedding = self.message_passing(embedding, weights)
        shape_activation = self.decoder(embedding).squeeze(1)
        for proposal_per_image, shape_activation_per_image in zip(proposals, shape_activation.split([len(v) for v in proposals], 0)):
            proposal_per_image.add_field('shape_activation', shape_activation_per_image)
        # inference mode
        if not self.training:
            return x, proposals, shape_activation, None
        # compute loss
        loss_propagating = self.propagating_loss_evaluator(proposals, shape_activation, targets)
        return x, proposals, shape_activation, loss_propagating

    def forward(self, features, proposals, targets=None):
        x, proposals, saliency, loss_activating = self.activating_saliency(
            features, proposals, targets)
        x, proposals, shape_activation, loss_propagating = self.propagating_saliency(
            x if self.cfg.MODEL.ROI_SHAPEPROP_HEAD.SHARE_FEATURE else features, proposals, targets)

        if not self.training:
            return proposals, {}
        return proposals, dict(loss_activating=loss_activating, loss_propagating=loss_propagating)

def build_shapeprop_head(cfg, in_channels):
    return ShapeProp(cfg, in_channels)
