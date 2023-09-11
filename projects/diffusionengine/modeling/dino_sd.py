# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import random
import time
import math
import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances

from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb
import detectron2.utils.comm as comm

from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
from .unet_2d_condition import UNet2DConditionModel
from diffusers import DDIMScheduler

coco_id2name = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
                9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
                24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
                34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
                46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
                53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
                60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
                78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
                86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
coco_continuous = list(coco_id2name.values())


class DINOSD(nn.Module):
    """ Code is modified from the `official github repo <https://github.com/IDEA-Research/DINO>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
        pixel_std (List[float]): Pixel std value for image normalization.
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            selected at postprocess for evaluation. Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(
        self,
        model_id_or_path: str,
        position_embedding: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        num_queries: int,
        criterion: nn.Module,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        aux_loss: bool = True,
        select_box_nums_for_evaluation: int = 300,
        device="cuda",
        dn_number: int = 100,
        label_noise_ratio: float = 0.2,
        box_noise_scale: float = 1.0,
        input_format: Optional[str] = "RGB",
        vis_period: int = 0,
    ):
        super().__init__()
        # define backbone and position embedding module
        scheduler = DDIMScheduler.from_pretrained(model_id_or_path, subfolder='scheduler')
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16, scheduler=scheduler)
        pipe.scheduler.set_timesteps(50)
        pipe.unet = UNet2DConditionModel.from_pretrained(model_id_or_path, torch_dtype=torch.float16, subfolder='unet')
        pipe._progress_bar_config = {'disable': True}

        self.backbone = pipe.to(device)
        self.position_embedding = position_embedding
        self.size_divisibility = pipe.image_processor.config.vae_scale_factor

        # define neck module
        self.neck = neck

        # number of dynamic anchor boxes and embedding dimension
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # define transformer module
        self.transformer = transformer

        # define classification head and box head
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion

        # denoising
        self.label_enc = nn.Embedding(num_classes, embed_dim)
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # initialize weights
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers + 1
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        # two-stage
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # hack implementation for two-stage
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        # the period for visualizing training samples
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.style2prompt = {'photo': ("elegant, meticulous, magnificent, maximum details, extremely hyper aesthetic, highly detailed.",
                                       "naked, deformed, bad anatomy, out of focus, disfigured, bad image, poorly drawn face, mutation, mutated, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, mutated hands and fingers, watermark, oversaturated, distorted hands."),
                             'clipart': ("", "")
                            }
        self.generated_json_dict = {
            'images': [], 'annotations': []
        }

    def forward(self, batched_inputs, data_engine=False, save_pth=None, gradio=False, **kwargs):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images, prompts = self.preprocess_image(batched_inputs)

        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            if batch_size > 1:
                img_masks = images.tensor.new_ones(batch_size, H, W)
                for img_id in range(batch_size):
                    img_h, img_w = batched_inputs[img_id]["instances"].image_size
                    img_masks[img_id, :img_h, :img_w] = 0
            else:
                img_masks = images.tensor.new_zeros(batch_size, H, W)

            if data_engine:
                batch_size *= kwargs['num_images_per_prompt']
                img_masks = img_masks.repeat_interleave(kwargs['num_images_per_prompt'], dim=0)

        # original feature
        if data_engine:
            assert (not self.training) and (save_pth or gradio)
            if gradio:
                positive_prompt, negative_prompt = kwargs.pop('positive_prompt', ''), kwargs.pop('negative_prompt', '')
                prompts = [', '.join([p.lower(), positive_prompt]) for p in prompts]
                strength = kwargs.pop('strength', 0.5)
                cur_seed = kwargs.pop('seed', 0)
                eta = kwargs.pop('eta', 0.1)
            else:
                style = batched_inputs[0]['style']
                positive_prompt, negative_prompt = self.style2prompt[style]
                prompts = [', '.join([p.lower(), positive_prompt]) for p in prompts]
                n_infer = 50
                if style == 'photo':
                    strength = random.randint(int(0.4 * n_infer), int(0.8 * n_infer)) / n_infer
                else:
                    strength = random.randint(int(0.7 * n_infer), int(0.9 * n_infer)) / n_infer
                print(prompts)
                cur_seed = time.time_ns()
                eta = random.randint(1, 4)/10
            features, gen_images = self.feature_extractor(image=images.tensor, prompt=prompts, gen_image=True,
                                                          negative_prompt=len(prompts) * [negative_prompt],
                                                          # diversity
                                                          strength=strength,
                                                          generator=torch.Generator(self.device).manual_seed(cur_seed),
                                                          eta=eta,
                                                          **kwargs)
        else:
            features = self.feature_extractor(image=images.tensor, prompt=prompts)[0]  # output feature dict

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in DINO
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        # denoising preprocessing
        # prepare label query embedding
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                targets,
                dn_number=self.dn_number,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.embed_dim,
                label_enc=self.label_enc,
            )
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)

        # feed into transformer
        (
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,  # [0..1]
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            attn_masks=[attn_mask, None],
        )
        # hack implementation for distributed training
        inter_states[0] += self.label_enc.weight[0, 0] * 0.0

        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # denoising postprocessing
        if dn_meta is not None:
            outputs_class, outputs_coord = self.dn_post_process(
                outputs_class, outputs_coord, dn_meta
            )

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        # prepare two stage output
        interm_coord = enc_reference
        interm_class = self.transformer.decoder.class_embed[-1](enc_state)
        output["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord}

        if self.training:
            # visualize training samples
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    box_cls = output["pred_logits"]
                    box_pred = output["pred_boxes"]
                    results = self.inference(box_cls, box_pred, images.image_sizes)
                    self.visualize_training(batched_inputs, results)

            # compute loss
            loss_dict = self.criterion(output, targets, dn_meta)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            if data_engine:
                all_image_sizes = torch.tensor(images.image_sizes).repeat_interleave(kwargs['num_images_per_prompt'],
                                                                                     dim=0).tolist()
                new_batched_inputs = []
                for i, gen_img in enumerate(gen_images):
                    new_input = copy.deepcopy(batched_inputs[i // kwargs['num_images_per_prompt']])
                    h, w = all_image_sizes[i]
                    gen_img = gen_img.crop((0, 0, w, h))
                    new_input['gen_image'] = np.asarray(gen_img).transpose(2, 0, 1)
                    new_batched_inputs.append(new_input)
                batched_inputs = new_batched_inputs
            else:
                all_image_sizes = images.image_sizes
            results = self.inference(box_cls, box_pred, all_image_sizes)

            if data_engine:
                img_buffer = self.data_engine(batched_inputs, results, save_pth, gradio=gradio)
                comm.synchronize()
                return img_buffer

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, all_image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            comm.synchronize()
            return processed_results

    @torch.no_grad()
    def feature_extractor(self, image, prompt, gen_image=False,
                          num_images_per_prompt=1, strength=0., eta=0., **kwargs):
        """ image: List of FloatTensor in [0, 1] """
        if gen_image:
            gen_images, diffusion_features = self.backbone(prompt=prompt, image=image,
                                                           strength=strength, eta=eta,
                                                           num_images_per_prompt=num_images_per_prompt,
                                                           return_dict=False,
                                                           feat_only=False,
                                                           feat_names=self.neck.in_features,
                                                           **kwargs)
        else:
            gen_images = None
            diffusion_features = self.backbone(prompt=prompt, image=image,
                                               strength=0., num_images_per_prompt=1,
                                               feat_only=True, feat_names=self.neck.in_features)

        return diffusion_features, gen_images

    def data_engine(self, batched_inputs, processed_results, save_pth, thr=0.3, gradio=False):
        from detectron2.utils.visualizer import Visualizer
        import detectron2.utils.comm as comm
        import os.path as osp
        if gradio:
            img_buffer = []
        if save_pth:
            os.makedirs(osp.join(save_pth, 'images'), exist_ok=True)
        for input, results_per_image in zip(batched_inputs, processed_results):
            select = results_per_image.scores > thr
            select_insts = results_per_image[select].to('cpu')
            if len(select_insts) == 0:
                print('No Valid Box, Skip')
                continue

            # detect real image
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)

            gen_img = input["gen_image"]
            gen_img = convert_image_to_rgb(gen_img.transpose(1, 2, 0), self.input_format)

            height, width = gen_img.shape[:2]

            gen_img_id = time.time_ns()
            fn = f'{gen_img_id}.jpg'

            if gradio:
                img_buffer.append(Image.fromarray(gen_img))
            if save_pth:
                Image.fromarray(gen_img).save(osp.join(save_pth, 'images', fn))
            boxes = select_insts.pred_boxes.tensor.detach().cpu().numpy()  # XYXY_ABS
            labels = select_insts.pred_classes.detach().cpu().numpy()
            labels = [coco_continuous[l] for l in labels]
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)

            results = instances_to_coco_json(select_insts, img_id=gen_img_id)
            for res, area in zip(results, areas):
                res['area'] = area.item()

            """
            >>> data.keys()
            dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
            >>> data['annotations'][0].keys()
            dict_keys(['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'])
            >>> data['images'][0].keys()
            dict_keys(['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id'])
            """
            self.generated_json_dict['annotations'].extend(results)
            self.generated_json_dict['images'].append({
                'file_name': fn, 'id': gen_img_id,
                'height': height, 'width': width
            })

            v_pred = Visualizer(gen_img, None)
            v_pred = v_pred.overlay_instances(boxes=boxes, labels=labels)
            pred_img = v_pred.get_image()

            if gradio:
                img_buffer.append(Image.fromarray(pred_img))

            cat_h, cat_w = max(img.shape[0], pred_img.shape[0]), max(img.shape[1], pred_img.shape[1])
            concat_image = np.zeros((cat_h, 2 * cat_w, 3), dtype=np.uint8)
            concat_image[:img.shape[0], :img.shape[1], :] = img
            concat_image[:pred_img.shape[0], cat_w:cat_w + pred_img.shape[1], :] = pred_img
            if save_pth:
                Image.fromarray(concat_image).save(osp.join(save_pth, f'test_{comm.get_rank()}.jpg'))

        if gradio:
            return img_buffer

    def visualize_training(self, batched_inputs, results):
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_box = 20

        for input, results_per_image in zip(batched_inputs, results):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=results_per_image.pred_boxes[:max_vis_box].tensor.detach().cpu().numpy()
            )
            pred_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, pred_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted boxes"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def prepare_for_cdn(
        self,
        targets,
        dn_number,
        label_noise_ratio,
        box_noise_scale,
        num_queries,
        num_classes,
        hidden_dim,
        label_enc,
    ):
        """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding
            in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
        if dn_number <= 0:
            return None, None, None, None
            # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t["labels"])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None, None

        dn_number = dn_number // (int(max(known_num) * 2))

        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat(
            [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(
                -1
            )  # half of bbox prob
            new_label = torch.randint_like(
                chosen_indice, 0, num_classes
            )  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_padding = int(max(known_num))

        pad_size = int(single_padding * 2 * dn_number)
        positive_idx = (
            torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        )
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = (
                torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            )
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to("cuda")
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num]
            )  # [1,2, 1,2,3]
            map_known_indice = torch.cat(
                [map_known_indice + single_padding * i for i in range(2 * dn_number)]
            ).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                single_padding * 2 * i: single_padding * 2 * (i + 1),
                single_padding * 2 * (i + 1): pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[
                single_padding * 2 * i: single_padding * 2 * (i + 1), : single_padding * i * 2
                ] = True
            else:
                attn_mask[
                single_padding * 2 * i: single_padding * 2 * (i + 1),
                single_padding * 2 * (i + 1): pad_size,
                ] = True
                attn_mask[
                single_padding * 2 * i: single_padding * 2 * (i + 1), : single_padding * 2 * i
                ] = True

        dn_meta = {
            "single_padding": single_padding * 2,
            "dn_num": dn_number,
        }

        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas):
        if dn_metas and dn_metas["single_padding"] > 0:
            padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]

            out = {"pred_logits": output_known_class[-1], "pred_boxes": output_known_coord[-1]}
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_coord)
            dn_metas["output_known_lbs_bboxes"] = out
        return outputs_class, outputs_coord

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        prompts = [x["prompt"] if "prompt" in x else "" for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images, prompts

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # box_cls.shape: 1, 300, 80
        # box_pred.shape: 1, 300, 4
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
                zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets
