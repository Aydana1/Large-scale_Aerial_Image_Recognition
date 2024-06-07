# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ## Description
# 
# Faster RCNN (ResNet101), 70l iterations
# ***

# %%
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.solver.build import get_default_optimizer_params
from detectron2.data.build import build_batch_data_loader, get_detection_dataset_dicts
from detectron2.data.datasets import register_coco_instances
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import TrainingSampler
from detectron2.data import transforms as T
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
# Backbone
from resnet import ResNet, BasicStem, BasicBlock, BottleneckBlock, DeformBottleneckBlock
# Proposal Generator
from rpn import RPN, StandardRPNHead
from anchor_generator import DefaultAnchorGenerator
# Backbone
from fpn import FPN, LastLevelMaxPool
from dlabv3_fpn import DeepLabV3plus_FPN
# ROI Heads
from fast_rcnn import FastRCNNOutputLayers
from box_head import FastRCNNConvFCHead
from roi_heads import StandardROIHeads
from poolers import ROIPooler

from rcnn import GeneralizedRCNN as RCNN
from trainer import DefaultTrainer, DefaultPredictor

import matplotlib.pyplot as plt

import os
import numpy as np
import torch

seed = 217
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

torch.multiprocessing.set_sharing_strategy('file_system')
print(torch.__version__, torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# %%
cfg = get_cfg()
cfg.merge_from_file("config.yaml")
# cfg.MODEL.WEIGHTS = 'model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
cfg.MODEL.WEIGHTS = ''
print(cfg)

# %% [markdown]
# ## Define the backbone
# %% [markdown]
# First, Define the FPN bottom-up part (which is a ResNet)

# %%

input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
norm = cfg.MODEL.RESNETS.NORM
stem = BasicStem(
    in_channels     = input_shape.channels,
    out_channels    = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
    norm            = norm,
)

freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
depth               = cfg.MODEL.RESNETS.DEPTH
num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
bottleneck_channels = num_groups * width_per_group
in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS

# assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

num_blocks_per_stage = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}[depth]

if depth in [18, 34]:
    assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
    assert not any(
        deform_on_per_stage
    ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
    assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
    assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

stages = []

for idx, stage_idx in enumerate(range(2, 6)):
    # res5_dilation is used this way as a convention in R-FCN & Deformable Conv paper
    dilation = res5_dilation if stage_idx == 5 else 1
    first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
    stage_kargs = {
        "num_blocks": num_blocks_per_stage[idx],
        "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "norm": norm,
    }
    # Use BasicBlock for R18 and R34.
    if depth in [18, 34]:
        stage_kargs["block_class"] = BasicBlock
    else:
        stage_kargs["bottleneck_channels"] = bottleneck_channels
        stage_kargs["stride_in_1x1"] = stride_in_1x1
        stage_kargs["dilation"] = dilation
        stage_kargs["num_groups"] = num_groups
        if deform_on_per_stage[idx]:
            stage_kargs["block_class"] = DeformBottleneckBlock
            stage_kargs["deform_modulated"] = deform_modulated
            stage_kargs["deform_num_groups"] = deform_num_groups
        else:
            stage_kargs["block_class"] = BottleneckBlock
    blocks = ResNet.make_stage(**stage_kargs)
    in_channels = out_channels
    out_channels *= 2
    bottleneck_channels *= 2
    stages.append(blocks)

# fpn_bottom_up (type Backbone > nn.Module)
fpn_bottom_up = ResNet(stem, stages, out_features=out_features, freeze_at=freeze_at)
# fpn_bottom_up

# %% [markdown]
# Define the backbone

# %%
# backbone = FPN(
#     bottom_up       = fpn_bottom_up,
#     in_features     = cfg.MODEL.FPN.IN_FEATURES,
#     out_channels    = cfg.MODEL.FPN.OUT_CHANNELS,
#     norm            = cfg.MODEL.FPN.NORM,
#     fuse_type       = cfg.MODEL.FPN.FUSE_TYPE,
#     top_block       = LastLevelMaxPool(),
# )
# # backbone


# %%
backbone = DeepLabV3plus_FPN(
    out_channels    = cfg.MODEL.FPN.OUT_CHANNELS, # default 256
    out_features    = ['p2', 'p3', 'p4', 'p5', 'p6'],
    strides         = {'p2': 4, 'p3': 8, 'p4': 16, 'p5': 32, 'p6': 64},
)
print(backbone)

# %% [markdown]
# ## Define the Region Proposal Network (RPN)
# %% [markdown]
# Define the Anchor generator

# %%
input_shape = backbone.output_shape()
input_shape = [input_shape[f] for f in cfg.MODEL.RPN.IN_FEATURES]

anchor_generator = DefaultAnchorGenerator(
    sizes           = cfg.MODEL.ANCHOR_GENERATOR.SIZES, 
    aspect_ratios   = cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,
    strides         = [x.stride for x in input_shape],
    offset          = cfg.MODEL.ANCHOR_GENERATOR.OFFSET,
)
# anchor_generator

# %% [markdown]
# Define the RPN head

# %%
input_shape = backbone.output_shape()
input_shape = [input_shape[f] for f in cfg.MODEL.RPN.IN_FEATURES]
in_channels = [s.channels for s in input_shape]
assert len(set(in_channels)) == 1, "Each level must have the same channel!"
in_channels = in_channels[0]

assert (
    len(set(anchor_generator.num_anchors)) == 1
), "Each level must have the same number of anchors per spatial position"

rpn_head = StandardRPNHead(
    in_channels = in_channels,
    num_anchors = anchor_generator.num_anchors[0],
    box_dim     = anchor_generator.box_dim,
    conv_dims   = cfg.MODEL.RPN.CONV_DIMS,
)
# rpn_head

# %% [markdown]
# Define the Region Proposal Generator (RPN)

# %%
proposal_generator = RPN(
    in_features             = cfg.MODEL.RPN.IN_FEATURES,
    head                    = rpn_head,
    anchor_generator        = anchor_generator,
    anchor_matcher          = Matcher(
                                cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
                                ),
    box2box_transform       = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS),
    batch_size_per_image    = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
    positive_fraction       = cfg.MODEL.RPN.POSITIVE_FRACTION,
    pre_nms_topk            = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST),
    post_nms_topk           = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST),
    nms_thresh              = cfg.MODEL.RPN.NMS_THRESH,
    min_box_size            = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
    anchor_boundary_thresh  = cfg.MODEL.RPN.BOUNDARY_THRESH,
    loss_weight             = {
                                "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,
                                "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
                                },
    box_reg_loss_type       = cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE,
    smooth_l1_beta          = cfg.MODEL.RPN.SMOOTH_L1_BETA,
)
# proposal_generator

# %% [markdown]
# Define the Region of Interest Head

# %%
input_shape = backbone.output_shape()
input_shape = [input_shape[f] for f in cfg.MODEL.RPN.IN_FEATURES]
in_channels = [s.channels for s in input_shape]
assert len(set(in_channels)) == 1, "Each level must have the same channel!"
in_channels = in_channels[0]

box_head = FastRCNNConvFCHead(
    input_shape = ShapeSpec(
                    channels    = in_channels, 
                    height      = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION, 
                    width       = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
                    ),
    conv_dims   = [cfg.MODEL.ROI_BOX_HEAD.CONV_DIM] * cfg.MODEL.ROI_BOX_HEAD.NUM_CONV,
    fc_dims     = [cfg.MODEL.ROI_BOX_HEAD.FC_DIM] * cfg.MODEL.ROI_BOX_HEAD.NUM_FC,
    conv_norm   = cfg.MODEL.ROI_BOX_HEAD.NORM,
)

box_predictor = FastRCNNOutputLayers(
        input_shape             = box_head.output_shape,
        box2box_transform       = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
        num_classes             = cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        test_score_thresh       = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
        test_nms_thresh         = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
        test_topk_per_image     = cfg.TEST.DETECTIONS_PER_IMAGE,
        cls_agnostic_bbox_reg   = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
        smooth_l1_beta          = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
        box_reg_loss_type       = cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
        loss_weight             = {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
)
# box_predictor


# %%
input_shape = backbone.output_shape()

roi_pooler = ROIPooler(
    output_size     = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
    scales          = tuple(1.0 / input_shape[k].stride for k in cfg.MODEL.ROI_HEADS.IN_FEATURES),
    sampling_ratio  = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
    pooler_type     = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE,
)

roi_heads = StandardROIHeads(
    box_in_features         = cfg.MODEL.ROI_HEADS.IN_FEATURES,
    box_pooler              = roi_pooler,
    box_head                = box_head,
    box_predictor           = box_predictor,
    num_classes             = cfg.MODEL.ROI_HEADS.NUM_CLASSES,
    batch_size_per_image    = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
    positive_fraction       = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
    proposal_matcher        = Matcher(
                                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                                allow_low_quality_matches=False,
                                ),
    proposal_append_gt      = True,
    mask_in_features        = None, #optional
    mask_pooler             = None, #optional
    mask_head               = None, #optional
    keypoint_in_features    = None, #optional
    keypoint_pooler         = None, #optional
    keypoint_head           = None, #optional
    train_on_pred_boxes     = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
)
# roi_heads

# %% [markdown]
# Define the model

# %%
faster_rcnn_101 = RCNN(
    backbone            = backbone,
    proposal_generator  = proposal_generator,
    roi_heads           = roi_heads,
    pixel_mean          = cfg.MODEL.PIXEL_MEAN,
    pixel_std           = cfg.MODEL.PIXEL_STD,
    input_format        = None,
    vis_period          = 0,
)
faster_rcnn_101 = faster_rcnn_101.to(device)
# faster_rcnn_101 # model

# %% [markdown]
# # Training

# %%
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

register_coco_instances("iSAID_train", {}, 
                        "/apps/local/shared/CV703/datasets/iSAID/iSAID_patches/train/instancesonly_filtered_train.json",
                        "/apps/local/shared/CV703/datasets/iSAID/iSAID_patches/train/images/")
register_coco_instances("iSAID_val", {}, 
                        "/apps/local/shared/CV703/datasets/iSAID/iSAID_patches/val/instancesonly_filtered_val.json",
                        "/apps/local/shared/CV703/datasets/iSAID/iSAID_patches/val/images/")

# %% [markdown]
# Build optimizer

# %%
params = get_default_optimizer_params(
    faster_rcnn_101, # model
    base_lr             = cfg.SOLVER.BASE_LR,
    weight_decay_norm   = cfg.SOLVER.WEIGHT_DECAY_NORM,
    bias_lr_factor      = cfg.SOLVER.BIAS_LR_FACTOR,
    weight_decay_bias   = cfg.SOLVER.WEIGHT_DECAY_BIAS,
) 
# ^ this can be expanded further

optimizer = torch.optim.SGD(
    params,
    lr              = cfg.SOLVER.BASE_LR,
    momentum        = cfg.SOLVER.MOMENTUM,
    nesterov        = cfg.SOLVER.NESTEROV,
    weight_decay    = cfg.SOLVER.WEIGHT_DECAY,
)
# optimizer

# %% [markdown]
# Train loader

# %%
is_train = True
dataset = get_detection_dataset_dicts(
    cfg.DATASETS.TRAIN,
    filter_empty    = cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
    min_keypoints   = cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                    if cfg.MODEL.KEYPOINT_ON
                    else 0,
    proposal_files  = cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
)

sampler = TrainingSampler(len(dataset))

augmentations = [
    T.ResizeShortestEdge(
        cfg.INPUT.MIN_SIZE_TRAIN, 
        cfg.INPUT.MAX_SIZE_TRAIN, 
        cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    ),
    T.RandomFlip(
        horizontal  = cfg.INPUT.RANDOM_FLIP == "horizontal",
        vertical    = cfg.INPUT.RANDOM_FLIP == "vertical",
    ),
]
if cfg.INPUT.CROP.ENABLED and is_train:
    augmentations.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    recompute_boxes = cfg.MODEL.MASK_ON
else:
    recompute_boxes = False

mapper = DatasetMapper(
    is_train                    = is_train,
    augmentations               = augmentations,
    image_format                = cfg.INPUT.FORMAT,
    use_instance_mask           = cfg.MODEL.MASK_ON,
    use_keypoint                = cfg.MODEL.KEYPOINT_ON,
    instance_mask_format        = cfg.INPUT.MASK_FORMAT,
    keypoint_hflip_indices      = None,
    precomputed_proposal_topk   = None,
    recompute_boxes             = recompute_boxes,
)

if isinstance(dataset, list):
    dataset = DatasetFromList(dataset, copy=False)
dataset = MapDataset(dataset, mapper)

data_loader = build_batch_data_loader(
    dataset,
    sampler,
    total_batch_size        = cfg.SOLVER.IMS_PER_BATCH,
    aspect_ratio_grouping   = cfg.DATALOADER.ASPECT_RATIO_GROUPING,
    num_workers             = cfg.DATALOADER.NUM_WORKERS,
    collate_fn              = None,
)
data_loader


# %%
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg, model=faster_rcnn_101, optimizer=optimizer, data_loader=data_loader) 
trainer.resume_or_load(resume=False)
trainer.train()


# %%
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set a custom testing threshold
predictor = DefaultPredictor(cfg, model=faster_rcnn_101)


# %%

evaluator = COCOEvaluator("iSAID_val")
val_loader = build_detection_test_loader(cfg, "iSAID_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`


