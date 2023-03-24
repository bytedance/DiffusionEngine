# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
from torch.nn.functional import normalize
from mmcv.runner import auto_fp16
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.detectors import DINO
import numpy as np


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


@DETECTORS.register_module()
class DINOGEN(DINO):
    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            raise NotImplementedError()

        if return_loss:
            raise NotImplementedError()
        else:
            if 'i2i' in kwargs and kwargs['i2i'] is True:
                kwargs.pop('i2i')
                assert ('prompt' in kwargs) and ('encode_ratio' in kwargs)
                return self.i2i_test(img, img_metas, **kwargs)
            else:
                raise NotImplementedError()

    def i2i_test(self, img, img_metas, encode_ratio, n_samples, n_iters,
                 custom_scale=None, custom_steps=None, eta=None, return_img_ann=True,
                 prompt='', rescale=False, seed=None, **kwargs):
        assert img_metas is None or len(img_metas) == 1

        img_meta = img_metas[0]
        if 'caption' in img_meta:
            prompt = img_meta['caption']

        batch_input_shape = tuple(img.size()[-2:])
        img_meta['batch_input_shape'] = batch_input_shape

        ref_filename = img_meta['ori_filename'] if 'ori_filename' in img_meta else None

        x, img, img_anns = self.backbone.i2i(ref_image=img, ref_filename=ref_filename,
                                             n_samples=n_samples, n_iters=n_iters, eta=eta,
                                             encode_ratio=encode_ratio, prompt=prompt,
                                             custom_scale=custom_scale, custom_steps=custom_steps,
                                             return_img_ann=return_img_ann, seed=seed, **kwargs)
        if self.with_neck:
            x = self.neck(x)

        bbox_results = []
        for i in range(len(img)):
            feats = tuple([fl[i].unsqueeze(0) for fl in x])
            results_list = self.bbox_head.simple_test(feats, [img_metas[0]], rescale=rescale)
            bbox_results.extend([
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ])
        return bbox_results, img.detach().cpu(), img_anns

