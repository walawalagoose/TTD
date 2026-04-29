# -*- coding: utf-8 -*-

# 1. This file collects significant hyperparameters for the configuration of TTA methods.
# 2. We are only concerned about method-related hyperparameters here.
# 3. We provide default hyperparameters from the paper or official repo if users have no idea how to set up reasonable values.
import math

algorithm_defaults = {
    "no_adaptation": {"model_selection_method": "last_iterate"},
    #
    "bn_adapt": {
        "adapt_prior": 0,  # the ratio of training set statistics.
    },
    "shot": {
        "optimizer": "SGD",  # Adam for officehome
        "auxiliary_batch_size": 32,
        "threshold_shot": 0.9,  # confidence threshold for online shot.
        "ent_par": 1.0,
        "cls_par": 0.3,  # 0.1 for officehome.
        "offline_nepoch": 10,
    },
    "ttt": {
        "optimizer": "SGD",
        "entry_of_shared_layers": "layer2",
        "aug_size": 32,
        "threshold_ttt": 1.0,
        "dim_out": 4,  # For rotation prediction self-supervision task.
        "rotation_type": "rand",
    },
    "pseudo": {
        "optimizer": "SGD",
    },
    "tent": {
        "optimizer": "SGD",
    },
    "t3a": {"top_M": 100},
    "cotta": {
        "optimizer": "SGD",
        "alpha_teacher": 0.999,  # weight of moving average for updating the teacher model.
        "aug_size": 32,
        "restore_prob": 0.01,  # the probability of restoring model parameters.
        # # CIFAR10-C
        # "threshold_cotta": 0.62,  # Threshold choice discussed in supplementary.
        # # CIFAR100-C
        # "threshold_cotta": 0.52,  # Threshold choice discussed in supplementary.
        # Imagenet-C
        "threshold_cotta": 0.1,  # Threshold choice discussed in supplementary.
    },
    "eata": {
        "optimizer": "SGD",
        "eata_margin_e0": math.log(1000) * 0.40,  # The threshold for reliable minimization in EATA.
        # "eata_margin_e0": math.log(100) * 0.40,
        # "eata_margin_e0": math.log(10) * 0.40,
        "eata_margin_d0": 0.05,  # for filtering redundant samples.
        "fishers": True, # whether to use fisher regularizer.
        "fisher_size": 2000,  # number of samples to compute fisher information matrix.
        "fisher_alpha": 50,  # the trade-off between entropy and regularization loss.
    },
    "memo": {
        "optimizer": "SGD",
        "episodic": "true",
        "aug_size": 32,
        "bn_prior_strength": 16,
    },
    "ttt_plus_plus": {
        "optimizer": "SGD",
        "entry_of_shared_layers": None,
        "batch_size_align": 256,
        "queue_size": 256,
        "offline_nepoch": 500,
        "bnepoch": 2,  # first few epochs to update bn stat.
        "delayepoch": 0,  # In first few epochs after bnepoch, we dont do both ssl and align (only ssl actually).
        "stopepoch": 25,
        "scale_ext": 0.5,
        "scale_ssh": 0.2,
        "align_ext": True,
        "align_ssh": True,
        "fix_ssh": False,
        "method": "align",  # choices = ['ssl', 'align', 'both']
        "divergence": "all",  # choices = ['all', 'coral', 'mmd']
    },
    "note": {
        "optimizer": "SGD",  # use Adam in the paper
        "memory_size": 64,
        "update_every_x": 64,  # This param may change in our codebase.
        "memory_type": "PBRS",
        "bn_momentum": 0.01,
        "temperature": 1.0,
        "iabn": False,  # replace bn with iabn layer
        "iabn_k": 4,
        "threshold_note": 1,  # skip threshold to discard adjustment.
        "use_learned_stats": True,
    },
    "conjugate_pl": {
        "optimizer": "SGD",
        "temperature_scaling": 1.0,
        "model_eps": 0.0,  # this should be added for Polyloss model.
    },
    "sar": {
        "optimizer": "SGD",
        "sar_margin_e0": math.log(1000) * 0.40,  # The threshold for reliable minimization in SAR.
        # "sar_margin_e0": math.log(100) * 0.40, 
        # "sar_margin_e0": math.log(10) * 0.40, 
        "reset_constant_em": 0.2,  # threshold e_m for model recovery scheme
    },
    "rotta":{
        "optimizer": "Adam",
        "nu": 0.001,
        "memory_size": 64,
        "update_frequency": 64,
        "lambda_t": 1.0,
        "lambda_u": 1.0,
        "alpha": 0.05,
    },
    "deyo": {
        "optimizer": "SGD",
        "deyo_margin": math.log(10) * 0.50, # CIFAR-10-C
        # "deyo_margin": math.log(100) * 0.50, # CIFAR-100-C
        # "deyo_margin": math.log(1000) * 0.50, # IMAGENET-C
        "filter_ent": True, # whether to filter samples by entropy
        "aug_type": "patch", # the augmentation type for prime
        "occlusion_size": 112, # choises for occ
        "row_start": 56, # choises for occ
        "column_start": 56, # choises for occ
        "patch_len": 4, # choises for patch
        "filter_plpd": True, # whether to filter samples by plpd
        "plpd_threshold": 0.3, # plpd threshold for DeYO
        "reweight_ent": 1, # reweight entropy loss
        "reweight_plpd": 1, # reweight plpd loss
        "margin": math.log(10) * 0.40, # CIFAR-10-C ent0 margin for DeYO
        # "margin": math.log(100) * 0.40, # CIFAR-100-C ent0 margin for DeYO
        # "margin": math.log(1000) * 0.40, # IMAGENET-C ent0 margin for DeYO
    },
    
    "vida":{
        "optimizer": "SGD",
        'ViDALR': 1e-4,
        'WD': 0.,
        'MT': 0.999,
        'MT_ViDA': 0.999,
        'beta': 0.9,
        "vida_rank1": 1,
        "vida_rank2": 128,
        "unc_thr":0.2,
        "alpha_teacher": 0.999,
        "alpha_vida": 0.99,
        "bn_momentum": 0.01,
        "aug_size": 10,
        "rst_prob": 0.001,
        },
    
    "santa":{
        "contrast_mode": "all",
        "temperature": 0.1,
        "projection_dim": 128,
        "lambda_ce_trg": 1.0,
        "lambda_cont": 1.0,
        "use_projector": True,
    },
    
    "tpt":{
        "optimizer": "SGD",
        "aug_size": 64,
        "rou": 0.1,
    },
    
    "clip_zs":{
        "model_selection_method": "last_iterate",
    },
    
    "tda": {
        "pos_config": {
            'enabled': True,
            'shot_capacity': 3,
            'alpha': 2.0,
            'beta': 5.0
        },
        
        "neg_config": {
            'enabled': True,
            'shot_capacity': 2,
            'alpha': 0.117,
            'beta': 1.0,
            'entropy_threshold': {'lower': 0.2, 'upper': 0.5},
            'mask_threshold': {'lower': 0.03, 'upper': 1.0}
        }
    },
    
    "boostadapter": {
        "infer_ori_image": True,
        "rou": 0.1, # entropy selection
        "delta": 0,
        "aug_size": 64,
        
        "pos_config": {
            'enabled': True,
            'shot_capacity': 3,
            'alpha': 2.0,
            'beta': 5.0
        },
        
        "neg_config": {
            'enabled': True,
            'shot_capacity': 2,
            'alpha': 0.117,
            'beta': 1.0,
            'entropy_threshold': {'lower': 0.2, 'upper': 0.5},
            'mask_threshold': {'lower': 0.03, 'upper': 1.0}
        }
    },
    
    "zero": {
      "rou": 0.3,  # entropy selection, gamma in original paper
      "aug_size": 64, 
    },
    
    "dpcore": {
        "optimizer": "AdamW",
        "inner_lr": 0.01,
        "temp_tau": 3.0,
        "ema_alpha": 0.999,
        "thr_rho": 0.8,
        "num_prompts": 8,
        "lamda": 1.0,
        "E_ID": 1,
        "E_OOD": 50,
        "dpcore_train_info": None,
        "dpcore_src_stat_path": "ttab/configs/dpcore_stats_vitsmall_cifar10.pth",
    },
    
    "batclip": {
        "optimizer": "SGD",
        "episodic": "true",
    },

    "codire": {
        "optimizer": "SGD",
        "reset_ratio": 0.2,  # the ratio of layers to reset when domain switch is detected.
        "reset threshold": 0.25,  # the threshold to detect domain switch
        "anchor_update_step": 20,  # the update frequency for the anchor in domain switch detection.
    },
}
