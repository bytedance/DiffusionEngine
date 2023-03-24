# DiffusionEngine
<p align="center">
<img src=misc/samples/head.png />
</p>

## Environment
Follow [this](https://github.com/CompVis/latent-diffusion/#requirements) to create and activate the `ldm` environment.
Then
```shell
bash init_diffeng.sh
```

## Datasets
Datasets should be placed in `./data`
We provide [COCO-DE](), [VOC-DE]() 

## Pretrained Models
Download the checkpoints and placed in `pretrained_models/`
- [sd-v2-512-base-ema.ckpt](https://huggingface.co/stabilityai/stable-diffusion-2-base/tree/main/512-base-ema.ckpt)
- [diffeng-dino-ldm.pth]()

## Try DiffusionEngine with Gradio App
```shell
python diffusionEngine_gradio.py
```

## Train your own DiffusionEngine
```shell
python tools/dist_train.py configs/dino/dino_4scale_ldm_8x2_12e_coco.py 8 --work-dir path_to_save_model_and_log
```