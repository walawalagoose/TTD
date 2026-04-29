# -*- coding: utf-8 -*-
import copy
import functools
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer

from ttab.loads.datasets.cifar.data_aug_cifar import aug_cifar
from ttab.loads.datasets.imagenet.data_aug_imagenet import aug_imagenet
from ttab.loads.datasets.mnist.data_aug_mnist import aug_mnist
from ttab.loads.datasets.yearbook.data_aug_yearbook import aug_yearbook

from ttab.model_adaptation.clip_ori.custom_clip import ClipZeroShot
from ttab.model_adaptation.clip_ori.prompts_classes import get_classnames


class CLIP_ZS(BaseAdaptation):
    def __init__(self, meta_conf, model: nn.Module):
        clip_model = ClipZeroShot(
            meta_conf.device, get_classnames(meta_conf.base_data_name),
            arch="ViT-L/14")
        super(CLIP_ZS, self).__init__(meta_conf, clip_model)
        self.transform_helper = self._get_transform_helper()
        
        self.input_resolution = 224 
        
    def _prior_safety_check(self):
        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert self._meta_conf.n_train_steps > 0, "adaptation steps requires >= 1."
        
    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
        # it'ok to set all modules in eval mode, since prompt learner in tpt has no BN layers or Dropout.
        model.eval()
        
        # disable grad, to (re-)enable only what specified adaptation method updates
        model.requires_grad_(False)

        return model.to(self._meta_conf.device)
    
    def initialize(self, seed: int):
        self._model = self._initialize_model(model=copy.deepcopy(self._base_model))
        # params, names = self._initialize_trainable_parameters()
    
    def _post_safety_check(self):
        pass
    
    def _get_transform_helper(self):
        """get particular augmentation method for different datasets"""
        if self._meta_conf.base_data_name in ["cifar10", "cifar100"]:
            return aug_cifar
        elif self._meta_conf.base_data_name in [
            "imagenet",
            "officehome",
            "waterbirds",
            "pacs",
        ]:
            return aug_imagenet
        elif self._meta_conf.base_data_name in ["coloredmnist"]:
            return aug_mnist
        elif self._meta_conf.base_data_name in ["yearbook"]:
            return aug_yearbook

    def _preprocess_image(self, x):
        """Transform the input image to the format that CLIP expects."""
        if x.shape[2] == self.input_resolution:
                return x
        elif len(x.shape) == 3:
            x = x.unsqueeze(0)
            x = F.interpolate(x, size=(self.input_resolution, self.input_resolution), 
                            mode='bilinear', align_corners=False)
            return x.squeeze(0)
        elif len(x.shape) == 4:
            return F.interpolate(x, size=(self.input_resolution, self.input_resolution), 
                            mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}. Expected (B,C,H,W) or (C,H,W)")
    
    def one_adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        timer: Timer,
        random_seed: int = None,
    ):
        """adapt the model in one step."""
        with timer("forward"):
            # Zero-Shot Prediction.
            inputs = self._preprocess_image(batch._x)
            with fork_rng_with_seed(random_seed):
                y_hat = model(inputs)
        
        return {
            "yhat": y_hat,
        }

    def run_multiple_steps(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        model_selection_method: BaseSelection,
        nbsteps: int,
        timer: Timer,
        random_seed: int = None,
    ):
        for step in range(1, nbsteps + 1):
            adaptation_result = self.one_adapt_step(
                model,
                optimizer,
                batch,
                timer,
                random_seed=random_seed,
            )

            model_selection_method.save_state(
                {
                    "model": copy.deepcopy(model).state_dict(),
                    "step": step,
                    "lr": self._meta_conf.lr,
                    **adaptation_result,
                },
                current_batch=batch,
            )

    def adapt_and_eval(
        self,
        episodic: bool,
        metrics: Metrics,
        model_selection_method: BaseSelection,
        current_batch: Batch,
        previous_batches: List[Batch],
        logger: Logger,
        timer: Timer,
    ):
        """The key entry of test-time adaptation."""
        # some simple initialization.
        log = functools.partial(logger.log, display=self._meta_conf.debug)
        if episodic:
            log("\treset model to initial state during the test time.")
            self.reset()

        log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()

        # evaluate the per batch pre-adapted performance. Different with no adaptation.
        if self._meta_conf.record_preadapted_perf:
            with timer("evaluate_preadapted_performance"):
                self._model.eval()
                with torch.no_grad():
                    x = self._preprocess_image(current_batch._x)
                    yhat = self._model(x)
                metrics.eval_auxiliary_metric(
                    current_batch._y, yhat, metric_name="preadapted_accuracy_top1"
                )

        # adaptation.
        with timer("test_time_adaptation"):
            nbsteps = self._get_adaptation_steps(index=len(previous_batches))
            log(f"\tadapt the model for {nbsteps} steps with lr={self._meta_conf.lr}.")
            self.run_multiple_steps(
                model=self._model,
                optimizer=None,
                batch=current_batch,
                model_selection_method=model_selection_method,
                nbsteps=nbsteps,
                timer=timer,
                random_seed=self._meta_conf.seed,
            )

        # select the optimal checkpoint, and return the corresponding prediction.
        with timer("select_optimal_checkpoint"):
            optimal_state = model_selection_method.select_state()
            log(
                f"\tselect the optimal model ({optimal_state['step']}-th step and lr={optimal_state['lr']}) for the current mini-batch.",
            )

            self._model.load_state_dict(optimal_state["model"])
            model_selection_method.clean_up()

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, optimal_state["yhat"])
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    optimal_state["yhat"],
                    current_batch._y,
                    current_batch._g,
                    is_training=False,
                )

    @property
    def name(self):
        return "clip_zs"
