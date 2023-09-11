from detrex.config import get_config
from .dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    model,
)

optimizer.lr = 2e-4

# get default config
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_3x

# modify model config
model.position_embedding.temperature = 20
model.position_embedding.offset = 0.0

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"

# max training iterations
train.max_iter = 270000

# modify dataloader config
# not filter empty annotations during training
dataloader.train.dataset.filter_empty = False
dataloader.train.total_batch_size = 32
dataloader.train.num_workers = 16

train.output_dir = "outputs/DINO-R50-6x"