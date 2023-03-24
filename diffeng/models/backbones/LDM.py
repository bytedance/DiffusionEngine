import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from mmcv.runner import BaseModule, get_dist_info
from mmdet.models.builder import BACKBONES
import torch.nn.functional as F
from mmdet.core.visualization import imshow_det_bboxes
import math
from tqdm import tqdm
from copy import deepcopy

from tools import COCO80, PALETTE

import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import random
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
import cv2


def norm(mp):
    mini, maxi = mp.min(), mp.max()
    return (mp - mini) / (maxi - mini)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    if not isinstance(pl_sd, OrderedDict):
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys: pass")
    #         print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys: pass")
    #         print(u)
    model.eval()
    return model


@BACKBONES.register_module()
class LDM(BaseModule):
    def __init__(self,
                 ldm_cfg,
                 ldm_ckpt,
                 out_stage,
                 out_layers,
                 out_step=None,  # default last step
                 ddim_steps: int = 30,
                 seed: int = 42,
                 scale: float = 7.5,
                 method: str = "multistep",
                 order: int = 2,
                 precision="autocast",
                 init_cfg=None,
                 ):
        super().__init__(init_cfg)
        config = OmegaConf.load(f"{ldm_cfg}")
        model = load_model_from_config(config, f"{ldm_ckpt}")
        self.sampler = DPMSolverSampler(model)
        self.model = model

        self.ddim_steps = ddim_steps

        self.encode_steps = order
        self.encode_ratio = self.encode_steps / self.ddim_steps

        self.scale = scale
        self.method = method
        self.order = order
        self.precision = precision

        self.out_stage = out_stage
        self.out_layers = out_layers

        self.neg_prompt = 'ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, ' \
                          'out of frame, extra limbs, disfigured, deformed, body out of frame, ' \
                          'bad anatomy, watermark, signature, cut off, low contrast, underexposed, ' \
                          'overexposed, bad art, beginner, amateur, distorted face.'
        self.pos_prompt = 'crafted, elegant, meticulous, magnificent, maximum details, extremely hyper aesthetic, intricately detailed.'

        if out_step is None:
            last_step_time = self.sampler.ratio_to_time(self.encode_ratio / self.encode_steps)
            out_step = int(self.sampler.time_continuous_to_discrete(last_step_time))

        choices = []
        for l in out_layers:
            choices.append(f'{out_step}_{l}')
        self.choices = choices

        # If seed is None, randomly select seed from 0 to 2^32-1
        if seed is None:
            seed = random.randrange(2 ** 32 - 1)
        seed_everything(seed)
        self.seed = seed

        self._freeze_stages()

    def _freeze_stages(self):
        for m in self.model.modules():
            # if isinstance(m, _BatchNorm):
            #     print('have BN')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def i2i(self, ref_image, ref_filename, n_samples=1, n_iters=1, encode_ratio=None, hw=None,
            custom_scale=None, custom_steps=None, prompt: str = '', p_prompt=None, n_prompt=None, eta=None,
            return_img_ann=False, seed=None):
        if custom_steps:
            ddim_steps = custom_steps
            tmp_encode_ratio = self.encode_steps / custom_steps
            last_step_time = self.sampler.ratio_to_time(tmp_encode_ratio / self.encode_steps)
            out_step = int(self.sampler.time_continuous_to_discrete(last_step_time))
            choices = []
            for chc in self.choices:
                choices.append(f'{out_step}_{chc.split("_")[-1]}')
        else:
            ddim_steps = self.ddim_steps
            choices = self.choices

        if seed is None:
            seed = time.time_ns() % 4294967295
            seed_everything(seed)

        if isinstance(encode_ratio, float):
            enc_steps = int(ddim_steps * encode_ratio)
            lb = ub = enc_steps
        elif isinstance(encode_ratio, tuple):
            lbR, ubR = encode_ratio
            lb, ub = int(ddim_steps * lbR), int(ddim_steps * ubR)
        else:
            lb, ub = int(ddim_steps * 0.3), int(ddim_steps * 1.0)

        precision_scope = autocast if self.precision == "autocast" else nullcontext
        device = self.model.device

        img_anns = []
        features = {k: [] for k in choices}
        out_samples = []
        self.model.cond_stage_model = self.model.cond_stage_model.to(device)
        with torch.no_grad():
            with precision_scope(self.model.device.type):
                with self.model.ema_scope():
                    init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(ref_image))
                    init_latent = init_latent.repeat(n_samples, 1, 1, 1)

                    for iter_id in range(n_iters):
                        if eta is None:
                            eta = random.randint(1, 4) / 10

                        enc_steps = random.randint(lb, ub)
                        enc_ratio = enc_steps / ddim_steps

                        noised_sample = self.sampler.stochastic_encode(init_latent, enc_ratio)
                        shape = noised_sample.shape[1:]

                        n_prompt = n_prompt if n_prompt is not None else self.neg_prompt
                        p_prompt = p_prompt if p_prompt is not None else self.pos_prompt

                        uc = None
                        if self.scale != 1.0:
                            uc = self.model.get_learned_conditioning([n_prompt] * n_samples)

                        if prompt.endswith('.'): prompt = prompt[:-1]
                        cond = self.model.get_learned_conditioning([". ".join([prompt, p_prompt])] * n_samples)

                        fb = {f'{self.out_stage}_feat': {}}
                        samples, _ = self.sampler.sample(
                            S=enc_steps,
                            batch_size=n_samples,
                            shape=shape,
                            conditioning=cond,
                            unconditional_guidance_scale=custom_scale if custom_scale else self.scale,
                            unconditional_conditioning=uc,
                            method=self.method,
                            order=self.order,
                            eta=eta,
                            lower_order_final=False,
                            t_start=self.sampler.ratio_to_time(enc_ratio),
                            x_T=noised_sample,
                            correcting_xt_fn=None,
                            extract_features=True, buffer=fb
                        )

                        x_samples = self.model.decode_first_stage(samples)
                        x_samples = torch.clamp(x_samples, min=-1.0, max=1.0)
                        out_samples.append(x_samples)

                        if return_img_ann:
                            for i in range(n_samples):
                                timestamp = f'{str(time.time_ns())}'
                                save_file_name = f'{ref_filename[:-4].replace("/", "-")}_{enc_steps}_{timestamp}.png'
                                # save_file_name = f'{timestamp}.png'
                                img_ann = {'file_name': save_file_name, 'id': int(timestamp),
                                           'caption': prompt, 'ref': ref_filename,
                                           'sd_setting': {'sampler': 'dpm-solver++', 'prompt': prompt,
                                                          'neg_prompt': n_prompt, 'pos_prompt': p_prompt,
                                                          'steps': ddim_steps, 'eta': eta, 'encode_steps': enc_steps,
                                                          'scale': custom_scale if custom_scale else self.scale,
                                                          'seed': seed},
                                           }
                                img_anns.append(img_ann)

                        for k, v in fb[f'{self.out_stage}_feat'].items():
                            if k in choices:
                                features[k].append(v.detach().float())

        out_features = []
        for k in choices:
            out_features.append(torch.cat(features[k], dim=0))

        if self.out_stage == 'up':
            out_features = out_features[::-1]

        return out_features, torch.cat(out_samples, dim=0), img_anns

    def train(self, mode=True):
        """keep freezed."""
        super().train(mode)
        self._freeze_stages()

    def forward(self, init_image, **kwargs):
        precision_scope = autocast if self.precision == "autocast" else nullcontext
        device = self.model.device

        if 'caption' not in kwargs:
            caption = [""] * len(init_image)
        else:
            caption = kwargs["caption"]
            assert len(caption) == len(init_image)
        self.model.cond_stage_model = self.model.cond_stage_model.to(device)
        with torch.no_grad():
            with precision_scope(self.model.device.type):
                with self.model.ema_scope():
                    uc = None
                    if self.scale != 1.0:
                        uc = self.model.get_learned_conditioning([""])
                    init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))

                    cond = self.model.get_learned_conditioning(caption)
                    uc = uc.repeat(len(init_latent), 1, 1)

                    noised_sample = self.sampler.stochastic_encode(init_latent, self.encode_ratio)

                    fb = {f'{self.out_stage}_feat': {}}
                    recover_latent, _ = self.sampler.sample(
                        self.encode_steps,
                        noised_sample.shape[0],
                        noised_sample.shape[1:],
                        conditioning=cond,
                        unconditional_guidance_scale=self.scale,
                        unconditional_conditioning=uc,
                        method=self.method,
                        order=self.order,
                        lower_order_final=False,
                        t_start=self.sampler.ratio_to_time(self.encode_ratio),
                        x_T=noised_sample,
                        correcting_xt_fn=None,
                        extract_features=True, buffer=fb
                    )

        features = []
        for k, v in fb[f'{self.out_stage}_feat'].items():
            if k in self.choices:
                features.append(v.detach().float())

        if self.out_stage == 'up':
            features = features[::-1]
        assert len(self.choices) == len(features)
        return features
