import torch
from torch.nn import functional as F

from shapeprop.layers import smooth_l1_loss
from shapeprop.modeling.matcher import Matcher
from shapeprop.structures.boxlist_ops import boxlist_iou
from shapeprop.modeling.utils import cat


def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )

    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.get_mask_tensor()
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)

class PropLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields(["labels", "masks", "valid_masks"])
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
        valid_inds = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            valid_masks = (labels_per_image > 0) & matched_targets.get_field("valid_masks").to(dtype=torch.uint8)
            positive_inds = torch.nonzero(valid_masks).squeeze(1)

            segmentation_masks = matched_targets.get_field("masks")
            segmentation_masks = segmentation_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            masks_per_image = project_masks_on_boxes(
                segmentation_masks, positive_proposals, self.discretization_size
            )

            labels.append(labels_per_image)
            masks.append(masks_per_image)
            valid_inds.append(valid_masks)

        return labels, masks, valid_inds

    def __call__(self, proposals, prop_logits, targets):
        _, prop_targets, valid_inds = self.prepare_targets(proposals, targets)
        
        prop_targets = cat(prop_targets, dim=0)
        valid_inds = cat(valid_inds, dim=0)
        
        # focus on instances that have mask annotation
        positive_inds = torch.nonzero(valid_inds).squeeze(1)
        if prop_targets.numel() == 0:
            return prop_logits.sum() * 0
        else:
            prop_logits = prop_logits[positive_inds]

        prop_loss = F.binary_cross_entropy_with_logits(
            prop_logits, prop_targets
        )
        return prop_loss


def make_propagating_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = PropLossComputation(
        matcher, cfg.MODEL.ROI_SHAPEPROP_HEAD.POOLER_RESOLUTION
    )

    return loss_evaluator
