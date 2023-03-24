import gradio as gr

from mmdet.datasets.pipelines import Compose
import numpy as np
import torch
from mmcv import Config, tensor2imgs
from mmcv.runner import (load_checkpoint, wrap_fp16_model)
from mmdet.models import build_detector
from mmdet.utils import (build_dp, compat_cfg,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
from tools import COCO80, PALETTE
from mmcv.parallel import DataContainer as DC

ckpt = "pretrained_models/diffeng-dino-ldm.pth"
cfg = Config.fromfile("diffeng_gen.py")
# replace the ${key} with the value of cfg.key
cfg = replace_cfg_vals(cfg)
# update data root according to MMDET_DATASETS
update_data_root(cfg)
cfg = compat_cfg(cfg)
# set multi-process settings
setup_multi_processes(cfg)
# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True
if 'pretrained' in cfg.model:
    cfg.model.pretrained = None
elif 'init_cfg' in cfg.model.backbone:
    cfg.model.backbone.init_cfg = None

if cfg.model.get('neck'):
    if isinstance(cfg.model.neck, list):
        for neck_cfg in cfg.model.neck:
            if neck_cfg.get('rfp_backbone'):
                if neck_cfg.rfp_backbone.get('pretrained'):
                    neck_cfg.rfp_backbone.pretrained = None
    elif cfg.model.neck.get('rfp_backbone'):
        if cfg.model.neck.rfp_backbone.get('pretrained'):
            cfg.model.neck.rfp_backbone.pretrained = None
cfg.gpu_ids = [0]
cfg.device = "cuda"
distributed = False
cfg.model.train_cfg = None
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
fp16_cfg = cfg.get('fp16', None)
if fp16_cfg is None and cfg.get('device', None) == 'npu':
    fp16_cfg = dict(loss_scale='dynamic')
if fp16_cfg is not None:
    wrap_fp16_model(model)
load_checkpoint(model, ckpt, map_location='cpu')
model.CLASSES = COCO80
model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
model.module.backbone.model.cuda()


@torch.no_grad()
def process(input_image, style, prompt, p_prompt, n_prompt, num_samples,
            scale, seed, eta, encode_ratio):
    if input_image is None:
        input_image = ((np.random.randn(512, 512, 3) + 1) * 127.5).astype(np.uint8)
        encode_ratio = 1.
    img_norm_cfg = dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=False)
    trans = Compose([dict(type='Resize', img_scale=(768, 512), keep_ratio=True),
                     dict(type='Normalize', **img_norm_cfg),
                     dict(type='Pad', size_divisor=256),
                     dict(type='DefaultFormatBundle'),
                     dict(type='Collect', keys=['img'],
                          meta_keys=('ori_shape', 'img_shape', 'pad_shape',
                                     'scale_factor', 'img_norm_cfg'))
                     ])
    data = trans({'img': input_image, 'img_shape': input_image.shape, 'ori_shape': input_image.shape})
    style_postfix = {'clipart': 'clipart, computerized graphic art',
                     'watercolor': 'painting produced with watercolors',
                     'photo': 'photograph'}

    if style in style_postfix:
        caption = f'{prompt}. {style_postfix[style]}.'
    else:
        caption = f'{style} of {prompt}. {style}.'

    data['img_metas'] = DC([[data['img_metas'].data]], cpu_only=True)
    img_meta = data['img_metas'].data[0][0]
    result, img_tensor, _ = model(return_loss=False, rescale=False, return_img_ann=False,
                                  img=data['img'].data.unsqueeze(0).cuda(), img_metas=data['img_metas'],
                                  i2i=True,
                                  encode_ratio=float(encode_ratio), n_iters=1,
                                  custom_scale=scale, seed=seed, eta=eta,
                                  n_samples=num_samples, prompt=caption,
                                  p_prompt=p_prompt, n_prompt=n_prompt)
    img_norm_cfg = img_meta['img_norm_cfg']
    gen_images = tensor2imgs(img_tensor, **img_norm_cfg)
    h, w, _ = img_meta['img_shape']

    image_buffer = []
    for iid, (im, bbox_result) in enumerate(zip(gen_images, result)):
        im = im[:h, :w, :]

        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        scores = bboxes[:, -1]

        inds = scores > 0.3
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        bbox_result = [bboxes[labels == i] for i in range(len(bbox_result))]

        img_wbox = model.module.show_result(
            im[:, :, ::-1], bbox_result,
            bbox_color=PALETTE, text_color=PALETTE, mask_color=PALETTE,
            score_thr=0., thickness=2, font_size=16)
        image_buffer.append(im)
        image_buffer.append(img_wbox[:, :, ::-1])

    return image_buffer


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("## DiffusionEngine")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            enc_ratio = gr.Slider(label="Encode Ratio", minimum=0.1, maximum=1.0, value=0.6, step=0.05)
            style = gr.Textbox(label="Domain", value="photo")
            scale = gr.Slider(label="Guidance Scale", minimum=1., maximum=15.0, value=7.5, step=0.5)
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="#Samples", minimum=1, maximum=12, value=4, step=1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Slider(label="eta", minimum=0., maximum=1.0, value=0.1, step=0.1)
                p_prompt = gr.Textbox(label="Added Prompt",
                                      value='crafted, elegant, meticulous, magnificent, maximum details, extremely hyper aesthetic, intricately detailed.')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, ' \
                                            'out of frame, extra limbs, disfigured, deformed, body out of frame, ' \
                                            'bad anatomy, watermark, signature, cut off, low contrast, underexposed, ' \
                                            'overexposed, bad art, beginner, amateur, distorted face.')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2,
                                                                                                   height='auto')

    ips = [input_image, style, prompt, p_prompt, n_prompt, num_samples, scale, seed, eta, enc_ratio]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])

    gr.Markdown("## Examples")
    gr.Examples(
        examples=[
            ['misc/samples/000000556666.png',
             'photo', 'people walking down an Asia alley.',
             'crafted, elegant, meticulous, magnificent, maximum details, extremely hyper aesthetic, intricately detailed.',
             'ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, ''out of frame, extra limbs, disfigured, deformed, body out of frame, ''bad anatomy, watermark, signature, cut off, low contrast, underexposed, ''overexposed, bad art, beginner, amateur, distorted face.',
             4, 7.5, 42, 0.1, 0.6],
            ['misc/samples/000000133679.png',
             'photo', 'A room with a hardwood floor and various types of chairs and furniture in the room.',
             'crafted, elegant, meticulous, magnificent, maximum details, extremely hyper aesthetic, intricately detailed.',
             'ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, ''out of frame, extra limbs, disfigured, deformed, body out of frame, ''bad anatomy, watermark, signature, cut off, low contrast, underexposed, ''overexposed, bad art, beginner, amateur, distorted face.',
             4, 7.5, 42, 0.1, 0.6],
        ],
        inputs=ips,
        outputs=[result_gallery],
        fn=process,
        cache_examples=True,
    )

demo.queue().launch()
