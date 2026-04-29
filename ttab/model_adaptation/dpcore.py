# -*- coding: utf-8 -*-
import copy
import functools
import os
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import ttab.loads.define_dataset as define_dataset
import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer

from ttab.model_adaptation.clip_ori.vpt import PromptViT


class DPCore(BaseAdaptation):
    """
    DPCore: Dynamic Prompt Coreset for Continual Test-Time Adaptation,
    http://arxiv.org/abs/2406.10737 (ICML'25),
    https://github.com/yunbeizhang/DPCore
    """

    def __init__(self, meta_conf, model: nn.Module):
        assert "vit" in meta_conf.model_name.lower(), "DPCore only supports ViT backbone."
        super(DPCore, self).__init__(meta_conf, model)

        self.temp_tau = self._meta_conf.temp_tau
        self.ema_alpha = self._meta_conf.ema_alpha
        self.thr_rho = self._meta_conf.thr_rho
        self.lamda = self._meta_conf.lamda
        self.E_ID = self._meta_conf.E_ID
        self.E_OOD = self._meta_conf.E_OOD
        self.inner_lr = self._meta_conf.inner_lr
        # Statistics of source domain: (std, mean) tuple of tensors
        # coreset: List[[mean, std, prompts_on_cpu]]

    def _check_model_interfaces(self, model: nn.Module):
        lacks = []
        if not hasattr(model, "prompts"):
            lacks.append("model.prompts")
        if not hasattr(model, "forward_raw_features"):
            lacks.append("model.forward_raw_features(x)")
        if not hasattr(model, "forward_features"):
            lacks.append("model.forward_features(x)")
        if model.vit is None or not hasattr(model.vit, "forward_head"):
            lacks.append("model.vit.forward_head(features)")
        if lacks:
            raise AttributeError(
                f"DPCore needs PromptViT style model, missing interfaces/attributes: {', '.join(lacks)}"
            )

    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
        if not isinstance(model, PromptViT):
            model = PromptViT(model, self._meta_conf.num_prompts)
        self._check_model_interfaces(model)
        model.train()
        model.requires_grad_(False)
        model.prompts.requires_grad_(True)
        return model.to(self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        """
            For ViT with prompts, only train the prompts.
        """    
        assert isinstance(self._model, PromptViT)
        self._adapt_module_names = ["prompts"]
        adapt_params = [self._model.prompts]
        adapt_param_names = ["prompts"]
                
        return adapt_params, adapt_param_names

    def _post_safety_check(self):
        assert self._model.training, "DPCore needs train mode."
        assert any(p.requires_grad for p in self._model.parameters()), "DPCore requires trainable params."

    def _load_train_info_from_meta(self):
        # Loading source domain statistics, from meta_conf.dpcore_train_info ((std, mean) tuple of tensors).
        if self._meta_conf.dpcore_train_info is not None:
            std, mean = self._meta_conf.dpcore_train_info
        else:
            stat_path = self._meta_conf.dpcore_src_stat_path
            assert stat_path is not None and os.path.exists(stat_path), "DPCore needs source domain statistics (std, mean) tensors provided in meta_conf.dpcore_train_info."
            loaded = torch.load(stat_path, map_location="cpu")
            std, mean = loaded["std"], loaded["mean"]
        self.train_info = (std.to(self._meta_conf.device), mean.to(self._meta_conf.device))
        return

    def initialize(self, seed: int):
        """Initialize the algorithm."""
        super(DPCore, self).initialize(seed)
        # DPCore settings.
        self._load_train_info_from_meta()
        assert self.train_info is not None and len(self.train_info) == 2
        self.coreset = []

    def reset(self):
        """recover model and optimizer to their initial states, then clear coreset"""
        self._model.load_state_dict(self.model_state_dict)
        self._optimizer.load_state_dict(self._base_optimizer.state_dict())
        # self._optimizer = copy.deepcopy(self._base_optimizer)
        self.coreset = []
    
    @torch.no_grad()
    def obtain_src_stat(
        self,
        scenario,
        num_samples: int = 5000,
        use_entropy_filter: bool = False,
        save_to: Optional[str] = None,
    ):
        src_dataset, src_loader = self.get_src_data(scenario, num_samples)
        feats = []
        n_collected = 0

        self._model.eval()
        for step, _, batch in src_loader.iterator(
            batch_size=self._meta_conf.batch_size,
            shuffle=True,
            repeat=False,
            ref_num_data=None,
            num_workers=self._meta_conf.num_workers
            if hasattr(self._meta_conf, "num_workers")
            else 2,
            pin_memory=True,
            drop_last=False,
        ):
            x = batch._x
            raw = self._model.forward_raw_features(x)  # [B, L, D]
            cls_feat = raw[:, 0]                      # [B, D]

            if use_entropy_filter:
                import math
                logits = self._model(x)
                ent = adaptation_utils.softmax_entropy(logits)  # [B]
                num_classes = logits.shape[1]
                thr = math.log(num_classes) / 2.0 - 1.0
                keep = torch.where(ent < thr)[0]
                if keep.numel() > 0:
                    cls_feat = cls_feat[keep]
                else:
                    continue

            feats.append(cls_feat)
            n_collected += cls_feat.size(0)
            if n_collected >= num_samples:
                break

        self._model.train()
        feats = torch.cat(feats, dim=0)[:num_samples]  # [N, D]
        std, mean = torch.std_mean(feats, dim=0)
        self.train_info = (std.to(self._meta_conf.device), mean.to(self._meta_conf.device))

        if save_to is None:
            save_to = getattr(self._meta_conf, "dpcore_src_stat_path", None)
        if save_to:
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            torch.save({"std": std.cpu(), "mean": mean.cpu()}, save_to)
            print(f"[DPCore] Saved source stats to {save_to} with N={feats.size(0)}")

        print(f"[DPCore] Source stats computed: N={feats.size(0)}, D={feats.size(1)}")

    @torch.no_grad()
    def forward_and_get_loss(self, images: torch.Tensor, with_prompt=False):
        if with_prompt:
            cls_features = self._model.forward_features(images)[:, 0]
        else:
            cls_features = self._model.forward_raw_features(images)[:, 0]
        """discrepancy loss"""
        batch_std, batch_mean = torch.std_mean(cls_features, dim=0)
        std_loss = torch.norm(batch_std - self.train_info[0], p=2)
        mean_loss = torch.norm(batch_mean - self.train_info[1], p=2)
        loss = self.lamda * std_loss + mean_loss
        return loss, batch_mean, batch_std

    @torch.enable_grad()
    def forward_and_adapt(self, x: torch.Tensor, optimizer: torch.optim.Optimizer):
        """One forward and adapt step, only updating prompts."""
        features = self._model.forward_features(x)
        cls_features = features[:, 0]
        batch_std, batch_mean = torch.std_mean(cls_features, dim=0)

        std_loss = torch.norm(batch_std - self.train_info[0], p=2)
        mean_loss = torch.norm(batch_mean - self.train_info[1], p=2)
        loss = self.lamda * std_loss + mean_loss

        output = self._model.vit.forward_head(features)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return output, loss, batch_mean, batch_std

    @staticmethod
    def calculate_weights(coreset, batch_mean, batch_std, lamda, temp_tau):
        mean_tensor = torch.stack([p[0] for p in coreset]).to(batch_mean.device) # [K, D]
        std_tensor = torch.stack([p[1] for p in coreset]).to(batch_std.device)   # [K, D]
        mean_match = torch.norm(batch_mean - mean_tensor, p=2, dim=1)
        std_match = torch.norm(batch_std - std_tensor, p=2, dim=1)
        match_loss = mean_match + lamda * std_match
        weights = torch.softmax(-match_loss / temp_tau, dim=0)
        return weights.detach().cpu()

    def _update_coreset(self, weights, batch_mean, batch_std):
        """EMA update the coreset statistics and prompts."""
        updated_prompts = self._model.prompts.clone().detach().cpu()
        for p_idx in range(len(self.coreset)):
            w = self.ema_alpha * weights[p_idx]
            self.coreset[p_idx][0] += w * (batch_mean.cpu() - self.coreset[p_idx][0])
            self.coreset[p_idx][1] += w * torch.clamp(batch_std.cpu() - self.coreset[p_idx][1], min=0.0)
            self.coreset[p_idx][2] += w * (updated_prompts - self.coreset[p_idx][2])

    @torch.no_grad()
    def _eval_coreset(self, x: torch.Tensor):
        """DPCore ID/OOD decision and weighted prompts calculation."""
        loss_raw, batch_mean, batch_std = self.forward_and_get_loss(x, with_prompt=False)

        is_ID = False
        weights = None
        weighted_prompts = None
        
        if self.coreset:
            weights = self.calculate_weights(self.coreset, batch_mean, batch_std, self.lamda, self.temp_tau)
            weighted_prompts = torch.stack([w * p[2] for w, p in zip(weights, self.coreset)], dim=0).sum(dim=0)
            assert weighted_prompts.shape == self._model.prompts.shape, f"{weighted_prompts.shape} != {self._model.prompts.shape}"
            # Testing with weighted prompts
            self._model.prompts = nn.Parameter(weighted_prompts.to(self._meta_conf.device))
            self._model.prompts.requires_grad_(False)

            loss_new, _, _ = self.forward_and_get_loss(x, with_prompt=True)
            # Determine ID/OOD
            if loss_new < loss_raw * self.thr_rho:
                self._model.prompts.requires_grad_(True)
                is_ID = True
        else:
            loss_new = loss_raw

        return is_ID, batch_mean, batch_std, weighted_prompts, weights, loss_raw, loss_new

    def one_adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        timer: Timer,
        random_seed: int = None,
    ):
        """Adapt the model in one step, with DPCore ID/OOD decision and inner loops."""
        grads = None

        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                is_ID, batch_mean, batch_std, weighted_prompts, weights, loss_raw, loss_new = self._eval_coreset(batch._x)

        with timer("backward"):
            last_logits = None
            last_loss = None
            
            if is_ID:
                # ID: initialize prompts with weighted prompts from coreset, and then E_ID steps of optimization
                self._model.prompts = nn.Parameter(weighted_prompts.to(self._meta_conf.device))
                inner_optim = torch.optim.AdamW([self._model.prompts], lr=self.inner_lr)
                for _ in range(self.E_ID):
                    logits, loss, batch_mean, batch_std = self.forward_and_adapt(batch._x, inner_optim)
                    last_logits, last_loss = logits, loss

                # EMA update the coreset statistics and prompts
                self._update_coreset(weights, batch_mean, batch_std)
                # grads = {"prompts": self._model.prompts.grad.clone().detach() if self._model.prompts.grad is not None else None} # Collect gradients
                grads = None
            else:
                # OOD: reset to initial model state, E_OOD steps of optimization, and add new coreset entry
                self._model.load_state_dict(self.model_state_dict, strict=True)
                self._model.prompts.requires_grad_(True)
                self._optimizer = torch.optim.AdamW([self._model.prompts], lr=self.inner_lr)
                for _ in range(self.E_OOD):
                    logits, loss, batch_mean, batch_std = self.forward_and_adapt(batch._x, self._optimizer)
                    last_logits, last_loss = logits, loss

                # Add new coreset entry
                self.coreset.append([
                    batch_mean.clone().detach().cpu(),
                    batch_std.clone().detach().cpu(),
                    self._model.prompts.clone().detach().cpu(),])

            y_hat = last_logits.detach()
            loss_val = float(last_loss.item())

        return {
            "optimizer": copy.deepcopy(self._optimizer).state_dict(),
            "loss": loss_val,
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
                    "lr": getattr(self._meta_conf, "lr", self.inner_lr),
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
                    yhat_pre = self._model(current_batch._x)
                self._model.train()
                metrics.eval_auxiliary_metric(
                    current_batch._y, yhat_pre, metric_name="preadapted_accuracy_top1"
                )

        # adaptation.
        with timer("test_time_adaptation"):
            nbsteps = self._get_adaptation_steps(index=len(previous_batches))
            if nbsteps <= 0:
                nbsteps = 1
            log(f"\tadapt the model for {nbsteps} steps (DPCore inner loops: E_ID={self.E_ID}, E_OOD={self.E_OOD}).")
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

    @property
    def name(self):
        return "dpcore"