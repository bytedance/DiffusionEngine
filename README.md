# DiffusionEngine
[[Project Page](https://mettyz.github.io/DiffusionEngine/)]
<p align="center">
<img src=misc/samples/head.jpg />
</p>

## Environment
```shell
conda create -n DE python=3.10
conda activate DE

pip install torch torchvision
python -m pip install -e detectron2
pip install -e .
```

## Datasets
DE datasets are assumed to be placed in `./engine_output/`
We provide [COCO-DE](), [VOC-DE]() 

## Pretrained Models
Download the checkpoints and placed in `./pt_models/`
- [stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base)
- [diffeng_model_best](dino_sd2-0_5scale_bsz64_90k_model_best.pth)

## Try DiffusionEngine with Gradio App
```shell
python diffusionEngine_gradio.py
```

## Train your own DiffusionEngine
```shell
python projects/diffusionengine/train_net.py \
    --config-file projects/diffusionengine/configs/dino-ldm/dino_sd2_512_5scale_90k.py \
    --num-gpus ${GPUS_PER_NODE} --machine-rank ${RANK} --num-machines ${NNODES} \
    --dist-url=tcp://${MASTER_ADDR}:${MASTER_PORT}
```

## Dataset Scaling-up with DiffusionEngine
```shell
python projects/diffusionengine/train_net.py \
    --config-file projects/diffusionengine/configs/dino-ldm/dino_sd2_512_5scale_90k.py \
    --num-gpus ${GPUS_PER_NODE} --machine-rank ${RANK} --num-machines ${NNODES} \
    --dist-url=tcp://${MASTER_ADDR}:${MASTER_PORT} \
    -de \
    train.init_checkpoint=pt_models/dino_sd2-0_5scale_bsz64_90k_model_best.pth \
    train.engine_output_dir=${OUTPUT_DIR} \
    train.seed=${SEED}
```

## Dataset PostProcess & Regsiter
Add the engine output dataset dir in `./detectron2/detectron2/data/datasets/register_coco_de.py`.

## License

This project is released under the [Apache 2.0 license](LICENSE).
