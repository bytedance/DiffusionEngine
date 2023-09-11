import gradio as gr

import numpy as np
import torch

from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate

ckpt = "pt_models/dino_sd2-0_5scale_bsz64_90k/model_best.pth"
cfg = LazyConfig.load("projects/diffusionengine/configs/dino-ldm/dino_sd2_512_5scale_90k.py")
model = instantiate(cfg.model)
model.eval().to("cuda")
checkpointer = DetectionCheckpointer(model)
checkpointer.resume_or_load(ckpt)


@torch.no_grad()
def process(input_image, style, prompt, p_prompt, n_prompt, num_samples, scale, seed, eta, enc_ratio):
    kwargs = {
        'positive_prompt': p_prompt,
        'negative_prompt': n_prompt,
        'num_images_per_prompt': num_samples,
        'guidance_scale': scale,
        'seed': seed,
        'eta': eta,
        'strength': enc_ratio
    }
    if input_image is None:
        input_image = ((np.random.randn(512, 512, 3) + 1) * 127.5).astype(np.uint8)
        kwargs['strength'] = 1.

    style_map = {
        'photo': 'realistic photo',
        'clipart': 'clipart anime',
        'watercolor': 'chinese painting, watercolor painting'
    }
    if style in style_map:
        prompt = f'{style_map[style]} of {prompt}'
    else:
        prompt = f'{style} of {prompt}'

    inputs = [{'image': torch.tensor(input_image).permute(2, 0, 1), 'prompt': prompt}]
    image_buffer = model(inputs, data_engine=True, gradio=True, **kwargs)
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
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery",
                                        columns=2, height='auto')

    ips = [input_image, style, prompt, p_prompt, n_prompt, num_samples, scale, seed, eta, enc_ratio]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])

    gr.Markdown("## Examples")
    gr.Examples(
        examples=[
            ['misc/samples/000000556666.png',
             'clipart', 'people walking down an Asia alley.',
             '',
             '',
             4, 7.5, 42, 0.1, 0.8],
            ['misc/samples/000000133679.png',
             'photo', 'A room with a hardwood floor and various types of chairs and furniture in the room.',
             'crafted, elegant, meticulous, magnificent, maximum details, extremely hyper aesthetic, intricately detailed.',
             'ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face.',
             4, 8, 42, 0., 0.55],
        ],
        inputs=ips,
        outputs=[result_gallery],
        fn=process,
        cache_examples=True,
    )

demo.queue().launch()
