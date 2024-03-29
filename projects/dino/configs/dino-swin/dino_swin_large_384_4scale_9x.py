from detrex.config import get_config
from ..models.dino_swin_large_384 import model

# get default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_3x
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "pt_models/swin_large_patch4_window12_384_22kto1k.pth"

train.max_iter = 270000
train.eval_period = 5000
train.log_period = 20
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.seed = 0
train.device = "cuda"
train.amp.enabled = True
model.device = train.device

# modify optimizer config
optimizer.lr = 2e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 48

train.output_dir = "outputs/DINO-SwinL-4scale-9x"
