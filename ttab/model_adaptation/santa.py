# -*- coding: utf-8 -*-
import copy
import functools
from typing import List

import PIL
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import ttab.loads.define_dataset as define_dataset
import ttab.model_adaptation.utils as adaptation_utils
from numpy import random
from torchvision.transforms import ColorJitter, Compose, Lambda
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer

class SANTA(BaseAdaptation):
    """Source Anchoring Network and Target Alignment for Continual Test Time Adaptation"""

    def __init__(self, meta_conf, model: nn.Module):
        super(SANTA, self).__init__(meta_conf, model)
        
        self.contrast_mode = getattr(meta_conf, "contrast_mode", "all")
        self.temperature = getattr(meta_conf, "temperature", 0.1)
        self.base_temperature = self.temperature
        self.projection_dim = getattr(meta_conf, "projection_dim", 128)
        self.lambda_ce_trg = getattr(meta_conf, "lambda_ce_trg", 1.0)
        self.lambda_cont = getattr(meta_conf, "lambda_cont", 1.0)

    def _initialize_model(self, model: nn.Module):
        # eval mode to avoid stochastic depth in swin. test-time normalization is still applied: follow the setting of original paper
        model.train()
        # disable grad, to (re-)enable only what specified adaptation method updates
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # bn module always uses batch statistics, in both training and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
        return model.to(self._meta_conf.device)
    
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
                module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)
            ):  # only bn is used in the paper.
                self._adapt_module_names.append(name_module)
                for name_parameter, parameter in module.named_parameters():
                    if name_parameter in ["weight", "bias"]:
                        adapt_params.append(parameter)
                        adapt_param_names.append(f"{name_module}.{name_parameter}")

        assert (
            len(self._adapt_module_names) > 0
        ), "SANTA needs some adaptable model parameters."
        return adapt_params, adapt_param_names

    def initialize(self, seed: int):
        """initialize the algorithm."""
        if self._meta_conf.model_selection_method == "oracle_model_selection":
            self._oracle_model_selection = True
            self.oracle_adaptation_steps = []
        else:
            self._oracle_model_selection = False

        self._model = self._initialize_model(model=copy.deepcopy(self._base_model))
        self._base_model = copy.deepcopy(self._model) # update base model
        
        if self._meta_conf.base_data_name == "cifar10":
            self.num_classes = 10
        elif self._meta_conf.base_data_name == "cifar100":
            self.num_classes = 100
        elif self._meta_conf.base_data_name == "imagenet":
            self.num_classes = 1000
        elif self._meta_conf.base_data_name == "officehome":
            self.num_classes = 65
        elif self._meta_conf.base_data_name == "pacs":
            self.num_classes = 7
        elif self._meta_conf.base_data_name == "coloredmnist":
            self.num_classes = 10
        elif self._meta_conf.base_data_name == "waterbirds":
            self.num_classes = 2
        elif self._meta_conf.base_data_name == "yearbook":
            self.num_classes = 5
        else:
            raise ValueError(f"Unsupported base data name: {self._meta_conf.base_data_name}")
        
        # 1. Set anchor model, different from base model!
        self.anchor_model = copy.deepcopy(self._model)
        for param in self.anchor_model.parameters():
            param.detach_()
        # 2. Split the model into feature extractor and classifier
        self.feature_extractor, self.classifier = self._split_model(self._model)
        # 3. Load or compute source prototypes (different from original paper implementation)
        self._load_source_prototypes()
        # 4. Set projector to align features to prototypes
        self._setup_projector()
        
        params, _ = self._initialize_trainable_parameters()
        if hasattr(self, 'projector') and not isinstance(self.projector, nn.Identity):
            for param in self.projector.parameters():
                params.append(param)            
        self._optimizer = self._initialize_optimizer(params)
        self._base_optimizer = copy.deepcopy(self._optimizer)
        self._auxiliary_data_cls = define_dataset.ConstructAuxiliaryDataset(
            config=self._meta_conf
        )
        self.transform = self.get_aug_transforms(img_shape=self._meta_conf.img_shape)
        # compute fisher regularizer
        self.fishers = None
        self.ewc_optimizer = torch.optim.SGD(params, 0.001)
        self.model_states = [
            copy.deepcopy(self._model.state_dict()),
            copy.deepcopy(self.anchor_model.state_dict())
        ]
        if hasattr(self, 'projector') and not isinstance(self.projector, nn.Identity):
            self.model_states.append(copy.deepcopy(self.projector.state_dict()))
        self.optimizer_state = copy.deepcopy(self._optimizer.state_dict())
        
    def _preprocess_image(self, x, input_resolution=224):
        if x.shape[2] == input_resolution:
                return x
        elif len(x.shape) == 3:
            x = x.unsqueeze(0)
            x = torch.nn.functional.interpolate(x, size=(input_resolution, input_resolution), 
                            mode='bilinear', align_corners=False)
            return x.squeeze(0)
        elif len(x.shape) == 4:
            return torch.nn.functional.interpolate(x, size=(input_resolution, input_resolution), 
                            mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}. Expected (B,C,H,W) or (C,H,W)")
        
    @staticmethod
    def get_aug_transforms(
        img_shape: tuple, gaussian_std: float = 0.005, soft: bool = False
    ):
        """Get augmentation transforms used at test time."""
        n_pixels = img_shape[0]

        clip_min, clip_max = 0.0, 1.0

        p_hflip = 0.5

        tta_transforms = transforms.Compose(
            [
                Clip(0.0, 1.0),
                ColorJitterPro(
                    brightness=[0.8, 1.2] if soft else [0.6, 1.4],
                    contrast=[0.85, 1.15] if soft else [0.7, 1.3],
                    saturation=[0.75, 1.25] if soft else [0.5, 1.5],
                    hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
                    gamma=[0.85, 1.15] if soft else [0.7, 1.3],
                ),
                transforms.Pad(padding=int(n_pixels / 2), padding_mode="edge"),
                transforms.RandomAffine(
                    degrees=[-8, 8] if soft else [-15, 15],
                    translate=(1 / 16, 1 / 16),
                    scale=(0.95, 1.05) if soft else (0.9, 1.1),
                    shear=None,
                    interpolation=F.InterpolationMode.BILINEAR, # using "interpolation" instead of "resample"
                    fill=None,
                ),
                transforms.GaussianBlur(
                    kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]
                ),
                transforms.CenterCrop(size=n_pixels),
                transforms.RandomHorizontalFlip(p=p_hflip),
                GaussianNoise(0, gaussian_std),
                Clip(clip_min, clip_max),
            ]
        )
        return tta_transforms

    def _split_model(self, model):
        """Split the model into feature extractor and classifier. The split position is determined by the architecture of the model."""
        if "vit" in self._meta_conf.model_name.lower():
            class ViTFeatureExtractor(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                    self.feature_dim = model.head.weight.shape[1]
                
                def forward(self, x):
                    if hasattr(self.model, 'forward_features'):
                        features = self.model.forward_features(x)
                        if features.dim() == 3:
                            # Using the CLS token as the feature representation
                            features = features[:, 0]  # [B, D]
                        return features
                    else:
                        raise NotImplementedError("Unsupported ViT architecture for feature extraction.")
            
            if hasattr(model, 'head'):
                classifier = model.head
                classifier = copy.deepcopy(classifier)
            else:
                raise ValueError("ViT model does not have a head attribute.")
            return ViTFeatureExtractor(model), classifier
        else:
            # CNN models like ResNet
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
            # Construct a new feature extractor with a flatten layer
            feature_extractor_with_flatten = nn.Sequential(
                feature_extractor,
                nn.Flatten()
            )
            classifier = list(model.children())[-1]
            return feature_extractor_with_flatten, classifier

    def _load_source_prototypes(self):
        """
        Load or compute source prototypes for each class. 
        In the original paper, prototypes are computed using the source data and kept fixed during adaptation. Here, we implement a more flexible version where prototypes can be updated online during adaptation using the features of incoming test samples and their predicted pseudo-labels. 
        This allows the method to better handle distribution shifts where the feature space may change over time.
        """
        # Delay prototype initialization until the first test batch is seen
        self.prototypes_src = None
        self.prototype_labels_src = torch.arange(self.num_classes).to(self._meta_conf.device)
        
        # Store source features and labels for prototype computation
        self.features_src_dict = {i: [] for i in range(self.num_classes)}
        self.prototype_counts = torch.zeros(self.num_classes).to(self._meta_conf.device)
        
        # Check whether the prototypes have been updated at least once
        self.prototype_updated = torch.zeros(self.num_classes, dtype=torch.bool).to(self._meta_conf.device)
        
    def update_prototypes(self, features, labels):
        with torch.no_grad():
            if self.prototypes_src is None:
                feature_dim = features.shape[1]
                self.prototypes_src = torch.randn(self.num_classes, 1, feature_dim).to(self._meta_conf.device)
                print((f"Initialize prototypes with feature dimension: {feature_dim}"))

            if features.shape[1] != self.prototypes_src.shape[2]:
                print(f"Feature dimension ({features.shape[1]}) does not match prototype dimension ({self.prototypes_src.shape[2]}), reinitializing prototypes")
                feature_dim = features.shape[1]
                self.prototypes_src = torch.randn(self.num_classes, 1, feature_dim).to(self._meta_conf.device)
                self.prototype_updated = torch.zeros(self.num_classes, dtype=torch.bool).to(self._meta_conf.device)
            
            for i in range(labels.shape[0]):
                label = labels[i].item()
                if label >= self.num_classes: 
                    continue
                feature = features[i:i+1] 
                
                # If it's the first time we see a sample of this class, initialize
                if not self.prototype_updated[label]:
                    self.prototypes_src[label, 0, :] = feature.squeeze(0)
                    self.prototype_updated[label] = True
                    self.prototype_counts[label] = 1
                else:
                    # Update prototype with EMA
                    alpha = 0.9
                    self.prototypes_src[label, 0, :] = alpha * self.prototypes_src[label, 0, :] + (1 - alpha) * feature.squeeze(0)
                    self.prototype_counts[label] += 1

    def _setup_projector(self):
        """ Set up the projector for feature alignment. """
        if getattr(self._meta_conf, "use_projector", True):
            self.projector_initialized = False
            self.use_projector = True
        else:
            self.projector = nn.Identity()
            self.projector_initialized = True
            self.use_projector = False

     # Integrated from: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    def contrastive_loss(self, features, labels=None, mask=None):
        batch_size = features.shape[0]
        
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self._meta_conf.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self._meta_conf.device)
        else:
            mask = mask.float().to(self._meta_conf.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = self.projector(contrast_feature)
        contrast_feature = nn.functional.normalize(contrast_feature, p=2, dim=1)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Conpute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self._meta_conf.device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss

    def reset(self):
        self._model.load_state_dict(self.model_states[0])
        self.anchor_model.load_state_dict(self.model_states[1])
        if hasattr(self, 'projector') and not isinstance(self.projector, nn.Identity) and len(self.model_states) > 2:
            self.projector.load_state_dict(self.model_states[2])
        
        self._optimizer.load_state_dict(self.optimizer_state)
        
        # For SANTA: reset prototype state
        feature_dim = self.prototypes_src.shape[-1]
        self.prototypes_src = torch.randn(self.num_classes, 1, feature_dim).to(self._meta_conf.device)
        self.prototype_updated = torch.zeros(self.num_classes, dtype=torch.bool).to(self._meta_conf.device)
        self.prototype_counts = torch.zeros(self.num_classes).to(self._meta_conf.device)

    def one_adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        timer: Timer,
        random_seed: int = None,
    ):
        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                # 1. Forward original test data, get features and predictions
                # forward original test data
                if "vit" in self._meta_conf.model_name.lower():
                    features_test = self.feature_extractor(self._preprocess_image(batch._x))
                else:
                    features_test = self.feature_extractor(batch._x)
                outputs_test = self.classifier(features_test)
                # update prototypes with test features and pseudo-labels
                pseudo_labels = outputs_test.argmax(dim=1)
                self.update_prototypes(features_test, pseudo_labels)
                # forward augmented test data
                if "vit" in self._meta_conf.model_name.lower():
                    aug_x = self.transform(self._preprocess_image(batch._x))
                else:
                    aug_x = self.transform(batch._x)
                features_aug_test = self.feature_extractor(aug_x)
                outputs_aug_test = self.classifier(features_aug_test)
                # forward original test data through the anchor model
                if "vit" in self._meta_conf.model_name.lower():
                    outputs_anchor = self.anchor_model(self._preprocess_image(batch._x))
                else:
                    outputs_anchor = self.anchor_model(batch._x)
            
            # 2. Compute prototype similarity
            with torch.no_grad():
                # dist[:, i] contains the distance from every source sample to one test sample
                dist = nn.functional.cosine_similarity(
                    x1=self.prototypes_src.repeat(1, features_test.shape[0], 1),
                    x2=features_test.view(1, features_test.shape[0], features_test.shape[1]).repeat(
                        self.prototypes_src.shape[0], 1, 1),
                    dim=-1
                )
                # for every test feature, get the nearest source prototype and derive the label
                _, indices = dist.topk(1, largest=True, dim=0)
                indices = indices.squeeze(0)
            
            # 3. Construct contrastive features
            features = torch.cat([
                self.prototypes_src[indices],
                features_test.view(features_test.shape[0], 1, features_test.shape[1]),
                features_aug_test.view(features_test.shape[0], 1, features_test.shape[1])
            ], dim=1)
            
            # 4. Delayed projector initialization
            if hasattr(self, 'use_projector') and self.use_projector and not self.projector_initialized and self.prototypes_src is not None:
                num_channels = self.prototypes_src.shape[2]
                self.projector = nn.Sequential(
                    nn.Linear(num_channels, self._meta_conf.projection_dim),
                    nn.ReLU(),
                    nn.Linear(self._meta_conf.projection_dim, self._meta_conf.projection_dim)
                ).to(self._meta_conf.device)
                self.projector_initialized = True
                # add the projector parameters to the optimizer
                for param in self.projector.parameters():
                    param.requires_grad = True
                self._optimizer.add_param_group({'params': self.projector.parameters()})
            
            # 5. Compute losses
            loss_contrastive = self.contrastive_loss(features=features, labels=None) # contrastive loss
            loss_self_training = adaptation_utils.AugCrossEntropy()(
                outputs_test, outputs_aug_test, outputs_anchor
            ).mean(0) # self-training loss with anchor guidance
            loss = self.lambda_ce_trg * loss_self_training + self.lambda_cont * loss_contrastive

        with timer("backward"):
            loss.backward()
            grads = dict(
                (name, param.grad.clone().detach())
                for name, param in model.named_parameters()
                if param.grad is not None
            )
            optimizer.step()
            optimizer.zero_grad()

        return {
            "optimizer": copy.deepcopy(optimizer).state_dict(),
            "loss": loss.item(),
            "grads": grads,
            "yhat": outputs_test,
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
        return "SANTA"
    

class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class Clip(torch.nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + "(min_val={0}, max_val={1})".format(
            self.min_val, self.max_val
        )


class ColorJitterPro(ColorJitter):
    """Randomly change the brightness, contrast, saturation, and gamma correction of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, "gamma")

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue, gamma):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                Lambda(lambda img: F.adjust_brightness(img, brightness_factor))
            )

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                Lambda(lambda img: F.adjust_contrast(img, contrast_factor))
            )

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                Lambda(lambda img: F.adjust_saturation(img, saturation_factor))
            )

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        if gamma is not None:
            gamma_factor = random.uniform(gamma[0], gamma[1])
            transforms.append(Lambda(lambda img: F.adjust_gamma(img, gamma_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = (
                    torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                )
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = (
                    torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                )
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = (
                    torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                )
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

            if fn_id == 4 and self.gamma is not None:
                gamma = self.gamma
                gamma_factor = torch.tensor(1.0).uniform_(gamma[0], gamma[1]).item()
                img = img.clamp(
                    1e-8, 1.0
                )  # to fix Nan values in gradients, which happens when applying gamma
                # after contrast
                img = F.adjust_gamma(img, gamma_factor)

        return img