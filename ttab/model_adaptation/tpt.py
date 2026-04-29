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

from ttab.model_adaptation.clip_ori.custom_clip import ClipTestTimeTuning
from ttab.model_adaptation.clip_ori.prompts_classes import get_classnames

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

class TPT(BaseAdaptation):
    """
    Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models (NeurIPS 2022),
    github link: https://github.com/azshue/TPT,
    """

    def __init__(self, meta_conf, model: nn.Module):
        clip_model = ClipTestTimeTuning(
            meta_conf.device, get_classnames(meta_conf.base_data_name),
            arch="ViT-L/14",ctx_init="a photo of a")
        super(TPT, self).__init__(meta_conf, clip_model)
        self.transform_helper = self._get_transform_helper()
        
        self.input_resolution = 224
        
    def _prior_safety_check(self):
        assert (
            self._meta_conf.aug_size > 0
        ), "The number of augmentation operation requires >= 1."
        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert self._meta_conf.n_train_steps > 0, "adaptation steps requires >= 1."
        
    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
        # it'ok to set all modules in eval mode, since prompt learner in tpt has no BN layers or Dropout.
        model.train()
        # model.eval()
        
        # disable grad, to (re-)enable only what specified adaptation method updates
        model.requires_grad_(False)
        for param in model.prompt_learner.parameters():
            param.requires_grad = True
        return model.to(self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        """
        select target parameters for adaptation methods.
        Only adapted prompt (Prompt Learner in clip_model)
        """
        adapt_params = []
        adapt_param_names = []
        
        for name, param in self._model.prompt_learner.named_parameters():
            adapt_params.append(param)
            adapt_param_names.append(f"prompt_learner.{name}")

        return adapt_params, adapt_param_names
    
    
    def _post_safety_check(self):
        # only prompts are adapted in tpt, so there is no need to check the training mode.
        param_grads = [p.requires_grad for p in (self._model.parameters())]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)
        assert has_any_params, "adaptation needs some trainable params."
        assert not has_all_params, "not all params are trainable."
    
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
        # model.eval()
        with timer("forward"):
            # Prompt Tuning.
            
            # Modification for fair comparison: apply gradient accumulation when batch size > 1.
            NUM_ACCUMULATION_STEPS = len(batch._x)
            for i in range(NUM_ACCUMULATION_STEPS):
                ori_input = self._preprocess_image(batch._x[i].unsqueeze(0))
                aug_inputs = [
                    self._preprocess_image(
                        self.transform_helper(batch._x[i], data_name=self._meta_conf.base_data_name)
                        )
                    for _ in range(self._meta_conf.aug_size-1)
                ]
                aug_inputs = torch.stack(aug_inputs).to(self._meta_conf.device)
                inputs = torch.cat((ori_input,aug_inputs),dim=0)
                
                with fork_rng_with_seed(random_seed):
                    y_hat = model(inputs)
                    
                # Entropy selection
                y_hat, _ = select_confident_samples(y_hat, self._meta_conf.rou)
                
                loss, _ = adaptation_utils.marginal_entropy(y_hat) # mariginal entropy
                loss = loss / NUM_ACCUMULATION_STEPS
                
                # apply fisher regularization when enabled
                if self.fishers is not None:
                    ewc_loss = 0
                    for name, param in model.prompt_learner.named_parameters():
                        if name in self.fishers:
                            ewc_loss += (
                                self._meta_conf.fisher_alpha
                                * (
                                    self.fishers[name][0]
                                    * (param - self.fishers[name][1]) ** 2
                                ).sum()
                            )
                    loss += ewc_loss
                
                loss.backward()
            

        # with timer("backward"):
        with timer("update"):
            grads = dict(
                (name, param.grad.clone().detach())
                for name, param in model.named_parameters()
                if param.grad is not None
            )
            # update parameters using accumulated gradients.
            optimizer.step()
            optimizer.zero_grad()
            
        # evaluate after adaptation
        with torch.no_grad():
            model.eval()
            y_hat = model(
                self._preprocess_image(batch._x)
                )  # already in eval mode
        return {
            "optimizer": copy.deepcopy(optimizer).state_dict(),
            "loss": loss.item(),
            "grads": grads,
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
                self._model.train()
                metrics.eval_auxiliary_metric(
                    current_batch._y, yhat, metric_name="preadapted_accuracy_top1"
                )

        # adaptation.
        with timer("test_time_adaptation"):
            nbsteps = self._get_adaptation_steps(index=len(previous_batches))
            log(f"\tadapt the model for {nbsteps} steps with lr={self._meta_conf.lr}.")
            self.run_multiple_steps(
                model=self._model,
                optimizer=self._optimizer,
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

            if self._oracle_model_selection:
                # oracle model selection needs to save steps
                self.oracle_adaptation_steps.append(optimal_state["step"])
                # update optimizer.
                self._optimizer.load_state_dict(optimal_state["optimizer"])

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, optimal_state["yhat"])
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    optimal_state["yhat"],
                    current_batch._y,
                    current_batch._g,
                    is_training=False,
                )

        # stochastic restore part of model parameters if enabled.
        if self._meta_conf.stochastic_restore_model:
            self.stochastic_restore()

    @property
    def name(self):
        return "tpt"
