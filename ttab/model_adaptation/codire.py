# -*- coding: utf-8 -*-
import copy
import functools
from typing import List

import torch
import torch.nn as nn
import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer

from ttab.model_adaptation.clip_ori.custom_clip import ClipZeroShot
from ttab.model_adaptation.clip_ori.prompts_classes import get_classnames
import torch.nn.functional as F
# torch.autograd.detect_anomaly(True)

import numpy as np

import torch
import torch.nn.functional as F


class CoDiRe(BaseAdaptation):
    """
    Test-Time Distillation for Continual Model Adaptation,
    https://arxiv.org/abs/2506.02671,
    """
    
    def __init__(self, meta_conf, model: nn.Module):
        clip_arch = getattr(meta_conf, "clip_arch", "ViT-L/14")
        self.clip_model = ClipZeroShot(
            meta_conf.device, get_classnames(meta_conf.base_data_name),
            arch=clip_arch)
        super(CoDiRe, self).__init__(meta_conf, model)
        self.input_resolution = 224
        
        self.domain_detector = DomainDetectorGrad(
            shift_detection_threshold=self._meta_conf.reset_threshold,
            period=self._meta_conf.anchor_update_step,
            update_source=True,
        )

    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
        model.train()
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        
        # disable grad, to (re-)enable only what specified adaptation method updates
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # bn module always uses batch statistics, in both training and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
        
        model = model.to(self._meta_conf.device)
        return model

    def _initialize_trainable_parameters(self):
        """
        Collect the affine scale + shift parameters from norm layers.

        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        self._adapt_module_names = []
        adapt_params = []
        adapt_param_names = []

        for name_module, module in self._model.named_modules():
            if isinstance(
                module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
            ):  # only bn is used in the paper.
                self._adapt_module_names.append(name_module)
                for name_parameter, parameter in module.named_parameters():
                    if name_parameter in ["weight", "bias"]:
                        adapt_params.append(parameter)
                        adapt_param_names.append(f"{name_module}.{name_parameter}")

        assert (
            len(self._adapt_module_names) > 0
        ), "CoDiRe needs some adaptable model parameters."
        return adapt_params, adapt_param_names
    
    def reset(self, reset_ratio=1.0, selective_reset=False, reset_start_idx=0, reset_end_idx=None):
        """
            Selectively reset the layers.
        """
        if not hasattr(self, "model_state_dict"):
            print("Warning: Initial model state not found, initializing now")
            self.model_state_dict = copy.deepcopy(self._base_model).state_dict()
        
        if selective_reset:
            if reset_start_idx < 0:
                reset_start_idx = len(self._adapt_module_names) + reset_start_idx
                reset_start_idx = max(0, reset_start_idx)  
            if reset_end_idx is None:
                reset_end_idx = len(self._adapt_module_names)
            elif reset_end_idx < 0:
                reset_end_idx = len(self._adapt_module_names) + reset_end_idx
                reset_end_idx = max(0, reset_end_idx) 
            
            reset_start_idx = min(reset_start_idx, len(self._adapt_module_names))
            reset_end_idx = min(reset_end_idx, len(self._adapt_module_names))
            modules_to_reset = self._adapt_module_names[reset_start_idx:reset_end_idx]
            if reset_start_idx == 0 and reset_end_idx == len(self._adapt_module_names):
                location_desc = "All layers"
            else:
                location_desc = f"From Index {reset_start_idx} to {reset_end_idx}(not including {reset_end_idx})"
            
            print(f"Selectively resetting layers: {location_desc}, reset ratio={reset_ratio:.2f}")
        else:
            modules_to_reset = self._adapt_module_names
        
        total_params = 0
        reset_params = 0
        
        for name in self._adapt_module_names:
            for param_name in ['weight', 'bias']:
                full_param_name = f"{name}.{param_name}"
                if full_param_name in self.model_state_dict:
                    try:
                        param = self._model.get_parameter(full_param_name)
                        total_params += 1
                        if name in modules_to_reset:
                            with torch.no_grad():
                                if reset_ratio >= 1.0:
                                    param.copy_(self.model_state_dict[full_param_name])
                                else:
                                    mask = (torch.rand(param.shape) < reset_ratio).float().to(param.device)
                                    param.data = self.model_state_dict[full_param_name] * mask + param.data * (1.0 - mask)
                            reset_params += 1
                    except Exception as e:
                        print(f"Error resetting parameter {full_param_name}: {e}")
        
        # Reset optimizer state if the model is fully reset and base optimizer state is available
        if reset_ratio >= 1.0 and not selective_reset and hasattr(self, "_base_optimizer"):
            self._optimizer.load_state_dict(self._base_optimizer.state_dict())
            print("Optimizer state has been reset to the initial state.")
        
        if selective_reset:
            print(f"Finished selective reset: {reset_params}/{total_params} parameters reset, reset ratio={reset_ratio:.2f}")
        else:
            if reset_ratio >= 1.0:
                print(f"Model fully reset to initial state: {reset_params}/{total_params} parameters reset.")
            else:
                print(f"Model partially reset: {reset_params}/{total_params} parameters reset with reset ratio={reset_ratio:.2f}.")
        
        if hasattr(self, 'domain_detector'):
            self.domain_detector._reset(self._model)
            
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
            with fork_rng_with_seed(random_seed):
                y_hat_tta = model(batch._x)
                y_hat_vlm = self.clip_model(self._preprocess_image(batch._x))
            y_hat_tta = y_hat_tta - torch.logsumexp(y_hat_tta, dim=-1, keepdim=True)
            y_hat_vlm = y_hat_vlm - torch.logsumexp(y_hat_vlm, dim=-1, keepdim=True)
            
            # Interpolation based on MSP to construct the blended teacher
            max_prob_vlm = torch.softmax(y_hat_vlm, dim=1).max(dim=1)[0]
            max_prob_tta = torch.softmax(y_hat_tta, dim=1).max(dim=1)[0]
            confidences = torch.stack([max_prob_vlm, max_prob_tta], dim=1)
            weights = torch.softmax(confidences, dim=1)
            lambda_ori = weights[:, 0].detach()  # shape: (batch_size,)
            lambda_ori = lambda_ori.view(-1,1)
            y_hat = lambda_ori * y_hat_vlm + (1 - lambda_ori) * y_hat_tta
            
            # Rectification based on Optimized Transport (OT)
            with torch.no_grad():
                C = y_hat_tta.size(1)
                # Voting
                preds_tta, preds_vlm, preds_exp = torch.argmax(y_hat_tta, dim=1), torch.argmax(y_hat_vlm, dim=1), torch.argmax(y_hat, dim=1)
                majority = preds_exp.clone()
                eq_tv = preds_tta == preds_vlm
                majority[eq_tv] = preds_tta[eq_tv]
                eq_te = (~eq_tv) & (preds_tta == preds_exp)
                majority[eq_te] = preds_tta[eq_te]
                eq_ve = (~eq_tv) & (~eq_te) & (preds_vlm == preds_exp)
                majority[eq_ve] = preds_vlm[eq_ve]
                labels_count_vote = torch.bincount(majority, minlength=C).to(torch.int64).cpu().numpy()
                codes = compute_codes(
                    y_hat_tta, epsilon=0.8, num_iters=3, 
                    observed_marginal=True, labels_count=labels_count_vote)
            cal_y_hat_tta = codes
            
            # Distillation with confidence-based sample reweighting
            distill_weights = calculate_sample_weights(y_hat_vlm, y_hat_tta, y_hat, mode='confidence')
            distill_loss = weighted_alignment_loss(F.softmax(y_hat_tta,dim=1), F.softmax(y_hat,dim=1), distill_weights) 
            
            # Entropy reweighting with dynamic margin (following DeYO)
            import math
            K = y_hat_tta.shape[1]
            margin = math.log(K) * 0.40
            reweight_ent = 1.0
            entropys = adaptation_utils.softmax_entropy(y_hat_tta)
            filter_ids = torch.where(entropys < margin)
            entropys = entropys[filter_ids]
            coeff = reweight_ent * (1 / (torch.exp(((entropys.clone().detach()) - margin)))) 
            entropys = entropys.mul(coeff)
            
            # Final loss function
            loss = distill_loss + entropys.mean(0)
            loss += adaptation_utils.IID_loss(F.softmax(y_hat_tta, dim=1), cal_y_hat_tta).mean(0)

        with timer("backward"):
            loss.backward()
            grads = dict(
                (name, param.grad.clone().detach())
                for name, param in model.named_parameters()
                if param.grad is not None
            )
            optimizer.step()
            optimizer.zero_grad()
        
        # Check for domain switch and reset if needed
        if self.domain_detector._check_domain_switch(self._model):
            start_idx = -int(self._meta_conf.reset_ratio * len(self._adapt_module_names)) # Deep reset the last 20% layers
            self.reset(reset_ratio=1.0, selective_reset=True, reset_start_idx=start_idx)
        self.domain_detector._update_domain_info(self._model)
            
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
                    yhat = self._model(current_batch._x)
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
        return "codire"


class DomainDetectorGrad(nn.Module):
    """
        A domain switch detector based on the gradient direction changes of normalization parameters.
    """
    def __init__(self, shift_detection_threshold=0.1, update_source=True, period=20):
        super(DomainDetectorGrad, self).__init__()
        self.shift_detection_threshold = shift_detection_threshold
        self.source_params = None
        self.last_params = None
        
        self.step = 0
        self.period = period
        self.update_source = update_source
        self.shifts_history = []
        
    def _get_norm_params(self, model: nn.Module) -> torch.Tensor:
        params = []
        for module in model.modules():
            if isinstance(
                module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
            ):
                if hasattr(module, 'weight') and module.weight is not None:
                    params.append(module.weight.detach().flatten())
                if hasattr(module, 'bias') and module.bias is not None:
                    params.append(module.bias.detach().flatten())
        return torch.cat(params)
    
    @torch.no_grad()
    def _check_domain_switch(self, model: nn.Module):
        current_params = self._get_norm_params(model)
        
        # Initialize source domain parameters
        if self.source_params is None:
            self.source_params = current_params.clone()
            self.last_params = current_params.clone()
            return False
        
        # Calculate the angle between current update and historical direction
        delta = current_params - self.last_params # Current direction
        td = self.last_params - self.source_params # Historical direction
        
        if delta.norm() == 0 or td.norm() == 0:
            angle = 0.0
        else:
            angle = 1 - F.cosine_similarity(delta.unsqueeze(0), (delta + td).unsqueeze(0)).item()
            
        self.shifts_history.append(angle)
        self.last_params = current_params.clone()
        
        # If angle exceeds the threshold, a domain switch is detected
        return angle > self.shift_detection_threshold

    @torch.no_grad()
    def _update_domain_info(self, model: nn.Module):
        current_params = self._get_norm_params(model)

        self.step += 1
        if self.update_source and self.step % self.period == 0:
            self.source_params = current_params.clone()
        self.last_params = current_params.clone()
    
    @torch.no_grad()    
    def _reset(self, model: nn.Module):
        current_params = self._get_norm_params(model)
        self.last_params = current_params.clone()
        
        
def compute_codes(logits, epsilon=0.8, num_iters=3, observed_marginal=False, labels_count=None):
    # Compute similarities and soft codes for all crops together
    with torch.no_grad():
        # Estimate marginal distributions
        r, c = None, None
        if observed_marginal:
            label_dist = labels_count / np.sum(labels_count)
            r = torch.tensor(label_dist).to(logits.device)

        # Compute soft code pseudo-labels
        soft_code = distributed_sinkhorn(logits, epsilon=epsilon, num_iters=num_iters, r=r, c=c)

    return soft_code

def distributed_sinkhorn(similarities, epsilon=0.8, num_iters=3, r=None, c=None):
    x = similarities.float()
    inv_eps = 1.0 / max(epsilon, 1e-3)
    Q = torch.exp((x * inv_eps).clamp(min=-50.0, max=50.0)).t()  # K-by-B
    K, B = Q.shape

    if r is None:
        r = torch.ones(K, device=x.device, dtype=torch.float32) / K
    else:
        r = r.float()
        r = torch.clamp(r, min=1e-8)
        r = r / r.sum()

    if c is None:
        c = torch.ones(B, device=x.device, dtype=torch.float32) / B
    else:
        c = c.float()
        c = torch.clamp(c, min=1e-8)
        c = c / c.sum()

    eps_sum = 1e-12

    # make the matrix sums to 1
    sum_Q = torch.sum(Q) + eps_sum
    Q = Q / sum_Q

    for _ in range(num_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True) + eps_sum
        Q = Q / sum_of_rows
        Q = Q * r.unsqueeze(1)

        # normalize each column: total weight per sample must be 1/B
        sum_of_cols = torch.sum(Q, dim=0, keepdim=True) + eps_sum
        Q = Q / sum_of_cols
        Q = Q * c.unsqueeze(0)

    Q = Q * B  # the columns must sum to 1 so that Q is an assignment
    Q = Q.t().contiguous()

    return Q

def calculate_sample_weights(y_hat_vlm, y_hat_tta, y_hat, mode='confidence'):
    """
        Calculate sample-wise weights based on the agreement between VLM and TTA predictions.
    """
    B, K = y_hat_vlm.shape
    vlm_probs = F.softmax(y_hat_vlm, dim=1)  # [B, K]
    tta_probs = F.softmax(y_hat_tta, dim=1)  # [B, K]
    teacher_probs = F.softmax(y_hat, dim=1)  # [B, K]
    
    if mode == 'confidence':
        # Using the maximum probability from the teacher (interpolated) as weight
        weights = teacher_probs.max(dim=1)[0]
        return weights.detach()

    elif mode == 'identity':
        weights = torch.ones(B, device=y_hat_tta.device)
        return weights.detach()
    
    else:
        raise ValueError("mode should be 'identity' or 'confidence'")

def weighted_alignment_loss(adaptnet_probs, teacher_probs, weights, eps=1e-10):
    """
        Calculate a weighted alignment loss (e.g., KL divergence) between the TTA model's predictions and the teacher's predictions.
    """
    cross_entropy_per_sample = - (teacher_probs * (adaptnet_probs + eps).log()).sum(dim=1) # [B,]
    weighted_loss = weights * cross_entropy_per_sample # [B,]

    # Normalize by the sum of weights to avoid scaling issues
    if weights.sum() > 0:
        return weighted_loss.sum() / weights.sum()
    else:
        return torch.tensor(0.0, device=adaptnet_probs.device)