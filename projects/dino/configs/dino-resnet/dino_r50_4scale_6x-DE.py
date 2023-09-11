from .dino_r50_4scale_6x import (
    train,
    dataloader,
    optimizer,
    model,
    lr_multiplier
)

dataloader.train.dataset.names = ["coco_2017_train", "coco-de"]

train.output_dir = "outputs/DINO-R50-6x-DE"
