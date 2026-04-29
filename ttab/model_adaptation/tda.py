# -*- coding: utf-8 -*-
import copy
import functools
from typing import List
import operator

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

# from ttab.model_adaptation.clip_ori import clip
from ttab.model_adaptation.clip_ori.custom_clip import ClipZeroShot4TTA
from ttab.model_adaptation.clip_ori.prompts_classes import get_classnames

def get_entropy_tda(loss, n_classes):
    """get entropy normalized by number of classes"""
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    return loss / torch.log(torch.tensor(n_classes, dtype=torch.float)).item()

class TDA(BaseAdaptation):
    """
    Efficient Test-Time Adaptation of Vision-Language Models (CVPR 2024),
    github link: https://kdiaaa.github.io/tda,
    """

    def __init__(self, meta_conf, model: nn.Module):
        clip_model = ClipZeroShot4TTA(
            meta_conf.device, get_classnames(meta_conf.base_data_name),
            arch="ViT-L/14")
        super(TDA, self).__init__(meta_conf, clip_model)    
        self.input_resolution = 224
        
        # Default configurations for positive and negative cache
        self.pos_config = {
            'enabled': True,
            'shot_capacity': 3,
            'alpha': 2.0,
            'beta': 5.0
        }
        self.neg_config = {
            'enabled': True,
            'shot_capacity': 2,
            'alpha': 0.117,
            'beta': 1.0,
            'entropy_threshold': {'lower': 0.2, 'upper': 0.5},
            'mask_threshold': {'lower': 0.03, 'upper': 1.0}
        }
        # Override with user configurations if provided
        if self._meta_conf.pos_config:
            self.pos_config.update(self._meta_conf.pos_config)
        if self._meta_conf.neg_config:
            self.neg_config.update(self._meta_conf.neg_config)
            
        # Positive and negative cache
        self.pos_cache = {}
        self.neg_cache = {}
        
    def _prior_safety_check(self):
        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert self._meta_conf.n_train_steps > 0, "adaptation steps requires >= 1."
        
    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
        # it'ok to set all modules in eval mode, since prompt learner in tda has no BN layers or Dropout.
        model.eval()
        
        # disable grad, to (re-)enable only what specified adaptation method updates
        model.requires_grad_(False)

        return model.to(self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        """
        select target parameters for adaptation methods.
        actually no parameters!
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
        pass
    
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
        
    def update_cache(self, cache, pred, features_loss, shot_capacity, include_prob_map=False):
        """Update cache with new features and loss, maintaining the maximum shot capacity."""
        with torch.no_grad():
            item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
            if pred in cache:
                if len(cache[pred]) < shot_capacity:
                    cache[pred].append(item)
                elif features_loss[1] < cache[pred][-1][1]:
                    cache[pred][-1] = item
                cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
            else:
                cache[pred] = [item]

    def compute_cache_logits(self, image_features, cache, alpha, beta, neg_mask_thresholds=None):
        """Compute logits using positive/negative cache."""
        with torch.no_grad():
            cache_keys = []
            cache_values = []
            for class_index in sorted(cache.keys()):
                for item in cache[class_index]:
                    cache_keys.append(item[0])
                    if neg_mask_thresholds:
                        cache_values.append(item[2])
                    else:
                        cache_values.append(class_index)

            cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
            if neg_mask_thresholds:
                cache_values = torch.cat(cache_values, dim=0)
                cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).to(self._meta_conf.device).half()
            else:
                cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=self._model.n_classes)).to(self._meta_conf.device).half()

            affinity = image_features @ cache_keys
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            return alpha * cache_logits
    
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
            inputs = self._preprocess_image(batch._x)
            with fork_rng_with_seed(random_seed):
                clip_logits, _, image_features = model(inputs)
                
            if image_features.size(0) > 1:
                batch_entropy = adaptation_utils.softmax_entropy(clip_logits)
                selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
                output = clip_logits[selected_idx]
                image_features = image_features[selected_idx].mean(0).unsqueeze(0)
                clip_logits = output.mean(0).unsqueeze(0)

                prob_map = output.softmax(1).mean(0).unsqueeze(0)
                pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
                
                loss, _ = adaptation_utils.marginal_entropy(output)
            else:
                prob_map = clip_logits.softmax(1)
                pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0]) 
                
                loss = adaptation_utils.softmax_entropy(clip_logits)
            
            prop_entropy = get_entropy_tda(loss, self._model.n_classes)
            
            if self.pos_config['enabled']:
                self.update_cache(
                    self.pos_cache,
                    pred, [image_features, loss],
                    self.pos_config['shot_capacity'])
            
            if self.neg_config['enabled'] and self.neg_config['entropy_threshold']['lower'] < prop_entropy < self.neg_config['entropy_threshold']['upper']:
                self.update_cache(
                    self.neg_cache,
                    pred, [image_features, loss, prob_map],
                    self.neg_config['shot_capacity'], True)
                
            # Initialize final logits with CLIP logits
            final_logits = clip_logits.clone()
            
            # Compute adapted logits using positive cache
            if self.pos_config['enabled'] and self.pos_cache:
                pos_logits = self.compute_cache_logits(
                    image_features, 
                    self.pos_cache, 
                    self.pos_config['alpha'], 
                    self.pos_config['beta'],
                )
                final_logits += pos_logits
                
            # Compute adapted logits using negative cache
            if self.neg_config['enabled'] and self.neg_cache:
                neg_logits = self.compute_cache_logits(
                    image_features, 
                    self.neg_cache, 
                    self.neg_config['alpha'], 
                    self.neg_config['beta'],
                    (self.neg_config['mask_threshold']['lower'],
                     self.neg_config['mask_threshold']['upper'])
                )
                final_logits -= neg_logits
        
        return {
            "optimizer": copy.deepcopy(optimizer).state_dict(),
            "loss": loss.item(),
            "yhat": final_logits,
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
                    yhat = self._model(x)[0]
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
        return "tda"
