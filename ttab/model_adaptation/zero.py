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

from ttab.model_adaptation.clip_ori.custom_clip import ClipZeroShot4TTA
from ttab.model_adaptation.clip_ori.prompts_classes import get_classnames

def select_confident_samples_zero(logits: torch.Tensor, probs: torch.Tensor, top:float, return_idx: bool=False):
    batch_entropy = -(probs * probs.log()).sum(1)
    full_idx = torch.argsort(batch_entropy, descending=False)
    filt_idx = full_idx[:int(batch_entropy.size()[0] * top)]
    if not return_idx:
        return logits[filt_idx]
    return logits[filt_idx], filt_idx, full_idx

def break_sample_tie(ties, logit, device):
    ties = torch.tensor(ties, dtype=torch.int, device=device)
    logit[~ties] = -torch.inf
    scalar_pred = torch.argmax(logit, dim=-1)
    return scalar_pred

def greedy_break(ties, logits, device):
    ties_tensor = torch.tensor(ties, dtype=torch.int, device=device)
    preds = torch.argmax(logits, dim=1)
    for pred in preds:
        if pred in ties_tensor:
            return pred
    return break_sample_tie(ties, logit=logits[0], device=device)

class ZERO(BaseAdaptation):
    """
    Frustratingly Easy Test-Time Adaptation of Vision-Language Models (NeurIPS 2024),
    github link: https://github.com/FarinaMatteo/zero,
    """

    def __init__(self, meta_conf, model: nn.Module):
        clip_model = ClipZeroShot4TTA(
            meta_conf.device, get_classnames(meta_conf.base_data_name),
            arch="ViT-L/14")
        super(ZERO, self).__init__(meta_conf, clip_model)
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
        model.eval()
        
        # disable grad, to (re-)enable only what specified adaptation method updates
        model.requires_grad_(False)

        return model.to(self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        """
        select target parameters for adaptation methods.
        No trainable parameters in ZERO.
        """
        adapt_params = []
        adapt_param_names = []
        
        self._adapt_module_names = []
        for name_module, module in self._model.named_modules():
            if isinstance(
                module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
            ):  # only bn is used in the paper.
                self._adapt_module_names.append(name_module)
                for name_parameter, parameter in module.named_parameters():
                    if name_parameter in ["weight", "bias"]:
                        adapt_params.append(parameter)
                        adapt_param_names.append(f"{name_module}.{name_parameter}")

        return adapt_params, adapt_param_names
    
    def _post_safety_check(self):
        assert all([not p.requires_grad for p in self._model.parameters()]), "ZERO should not have trainable parameters."
    
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
        with timer("forward"):
            # Modification for fair comparison: apply gradient accumulation when batch size > 1.
            BS = len(batch._x)
            for i in range(BS):
                ori_input = self._preprocess_image(batch._x[i].unsqueeze(0))
                aug_inputs = [
                    self._preprocess_image(
                        self.transform_helper(batch._x[i], data_name=self._meta_conf.base_data_name)
                        )
                    for _ in range(self._meta_conf.aug_size-1)
                ]
                aug_inputs = torch.stack(aug_inputs).to(self._meta_conf.device)
                inputs = torch.cat((ori_input,aug_inputs),dim=0)
            
            # compute probabilities and confidence filter
            with fork_rng_with_seed(random_seed):
                with torch.no_grad():
                    logits, logits_ori, _ = model(inputs) # scaled and unscaled logits
                    probs = logits.softmax(1)
                    logits_filt, _, sorted_idx = select_confident_samples_zero(logits_ori, probs, top=self._meta_conf.rou, return_idx=True) # retain most confident views
                
            # zero-out the temperature, marginalize and predict
            zero_temp = torch.finfo(logits_filt.dtype).eps
            p_bar = (logits_filt / zero_temp).softmax(1).sum(0) # marginalize
            
            # check if we have to break ties in some way
            max_counts, scalar_pred = torch.max(p_bar, dim=-1)
            ties = [scalar_pred]
            for idx in range(len(p_bar)):
                if idx == scalar_pred: continue
                if p_bar[idx] == max_counts: ties.append(idx)

            # if so, break ties greedily
            if len(ties) > 1:
                k = int(inputs.size(0) * self._meta_conf.rou) 
                sorted_logits = logits_ori[sorted_idx]
                scalar_pred = greedy_break(ties, sorted_logits[k:], device=logits_ori.device)
                p_bar[scalar_pred]+=1

            # need to unsqueeze for compatibility with the 'accuracy' function
            y_hat = p_bar.unsqueeze(0)       
        
        return {
            "optimizer": copy.deepcopy(optimizer).state_dict(),
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
        return "zero"
