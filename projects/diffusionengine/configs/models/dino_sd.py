import copy
import torch.nn as nn

from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine

from projects.diffusionengine.modeling import (
    DINOSD,
    DINOTransformerEncoder,
    DINOTransformerDecoder,
    DINOTransformer,
    DINOCriterion,
)

model = L(DINOSD)(
    model_id_or_path="pt_models/stable-diffusion-2-base",
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    neck=L(ChannelMapper)(
        input_shapes={
            "div1": ShapeSpec(channels=960),
            "div2": ShapeSpec(channels=1600),
            "div4": ShapeSpec(channels=1920),
            "div8": ShapeSpec(channels=3840)
        },
        in_features=["div1", "div2", "div4", "div8"],
        out_channels=256,
        num_outs=5,
        kernel_size=1,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
    ),
    transformer=L(DINOTransformer)(
        encoder=L(DINOTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            post_norm=False,
            num_feature_levels="${..num_feature_levels}",
            use_checkpoint=False
        ),
        decoder=L(DINOTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels="${..num_feature_levels}",
            use_checkpoint=False,
        ),
        num_feature_levels=5,
        two_stage_num_proposals="${..num_queries}",
    ),
    embed_dim=256,
    num_classes=80,
    num_queries=900,
    aux_loss=True,
    criterion=L(DINOCriterion)(
        num_classes="${..num_classes}",
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 1,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_class_dn": 1,
            "loss_bbox_dn": 5.0,
            "loss_giou_dn": 2.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
        two_stage_binary_cls=False,
    ),
    dn_number=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    pixel_mean=[0., 0., 0.],
    pixel_std=[255., 255., 255.],
    vis_period=0,
    input_format="RGB",
    device="cuda",
)

# set aux loss weight dict
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict
