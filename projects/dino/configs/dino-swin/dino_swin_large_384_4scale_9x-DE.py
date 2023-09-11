from .dino_swin_large_384_4scale_9x import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

dataloader.train.dataset.names = ["coco_2017_train", "coco-de"]

# modify training config
train.output_dir = "outputs/DINO-SwinL-4scale-9x-DE"
