# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import itertools
import json
import logging
import os
import os.path as osp
import time
from collections import OrderedDict
from contextlib import ExitStack, contextmanager

import torch
import torch.distributed as dist
from torch import nn

import detectron2.utils.comm as comm
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.utils.comm import get_world_size, is_main_process


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results

class simple_iter_loader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._iterator = iter(self.data_loader)

    def reset(self):
        self._iterator = iter(self.data_loader)

    def __next__(self):
        try:
            next_item = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._iterable)
            next_item = next(self._iterator)
        return next_item



def data_engine_on_dataset(
    cfg, model, data_loader, save_pth, save_period, max_iter=10, **kwargs
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """

    json_fn = osp.join(save_pth, 'annotations_pre_process.json')

    data_loader = simple_iter_loader(data_loader)
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        step = -1
        while step < max_iter:
            step += 1

            inputs = next(data_loader)
            model(inputs, save_pth=save_pth, **kwargs)
            if torch.cuda.is_available():
                comm.synchronize()

                model_without_ddp = model.module if comm.get_world_size() > 1 else model

                if step % save_period == 0 or step == max_iter-1:
                    tic = time.perf_counter()
                    current_json_dict = {}
                    current_json_dict['images'] = list(itertools.chain(*comm.gather(model_without_ddp.generated_json_dict['images'], dst=0)))
                    current_json_dict['annotations'] = list(itertools.chain(*comm.gather(model_without_ddp.generated_json_dict['annotations'], dst=0)))

                    if comm.is_main_process():
                        save_json_dict = {'images': [], 'annotations': []}
                        if osp.exists(json_fn):
                            with open(json_fn, 'r') as f:
                                save_json_dict = json.load(f)
                        exist_img = len(save_json_dict['images'])
                        exist_ann = len(save_json_dict['annotations'])

                        for k, v in current_json_dict.items():
                            save_json_dict[k].extend(v)
                        with open(json_fn, 'w') as f:
                            f.write(json.dumps(save_json_dict))
                            f.flush()

                        print(f'Saving Json #Images = {exist_img}->{len(save_json_dict["images"])}, '
                              f'#Anns = {exist_ann}->{len(save_json_dict["annotations"])}')

                    current_json_dict = save_json_dict = {}
                    model_without_ddp.generated_json_dict = {
                        'images': [], 'annotations': []
                    }
                    comm.synchronize()
                    print(f'Saving Cost: {time.perf_counter() - tic:.4f}')


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
