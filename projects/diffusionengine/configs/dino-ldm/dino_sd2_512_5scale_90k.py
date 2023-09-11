from detectron2.layers import ShapeSpec
from detrex.config import get_config
from ..models.dino_sd import model

# get default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

global_batch_size = 64

# modify training config
train.max_iter = 90000
train.eval_period = 2000
train.log_period = 20
train.seed = 42

train.checkpointer.period = 2000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 2e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0. if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = global_batch_size


model.model_id_or_path = 'pt_models/stable-diffusion-2-base'
model.neck.input_shapes = {
    "div1": ShapeSpec(channels=960),
    "div2": ShapeSpec(channels=1600),
    "div4": ShapeSpec(channels=1920),
    "div8": ShapeSpec(channels=3840),
}
model.neck.in_features = ["div1", "div2", "div4", "div8"]
model.neck.num_outs = 5
model.transformer.num_feature_levels = 5

# modify training config
train.output_dir = "pt_models/dino_sd2-0_5scale_bsz64_90k"
model.vis_period = 50
