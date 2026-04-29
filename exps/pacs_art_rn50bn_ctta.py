class NewConf(object):
    # create the list of hyper-parameters to be replaced.
    to_be_replaced = dict(
        # general for world.
        # seed=[2023],
        seed=[2022, 2023, 2024],
        main_file=[
            "run_exp.py",
            ],
        job_name=[
            # "pacs_episodic_oracle_model_selection",
            "pacs_art_online_last_iterate",
            # "pacs_art_episodic_last_iterate",
        ],
        base_data_name=[
            "pacs",
        ],
        src_data_name=[
            "pacs_art",
            # "pacs_cartoon",
            # "pacs_photo",
            # "pacs_sketch",
        ],
        data_names=[
            # art -> others
            "pacs_cartoon;pacs_photo;pacs_sketch",
            # # cartoon -> others
            # "pacs_art;pacs_photo;pacs_sketch",
            # # photo -> others
            # "pacs_art;pacs_cartoon;pacs_sketch",
            # # sketch -> others
            # "pacs_art;pacs_cartoon;pacs_photo",
        ],
        model_name=[
            "resnet50",
        ],
        model_adaptation_method=[
            # "no_adaptation",
            # "tent",
            # "bn_adapt",
            # "t3a",
            # "memo",
            # "shot",
            # "ttt",
            # "note",
            # "sar",
            # "conjugate_pl",
            # "cotta",
            # "rotta",
            # "eata",
            # "tpt",
            # "clip_zs",
            # "codire",
            # "deyo",
            # "vida",
            # "tda",
            # "boostadapter",
            # "santa",
            
            # "zero",
            # "batclip",
            "bca",
        ],
        model_selection_method=[
            # "oracle_model_selection", 
            "last_iterate",
        ],
        offline_pre_adapt=[
            "false",
        ],
        data_wise=["batch_wise"],
        batch_size=[1],
        episodic=[
            "false", 
            # "true",
        ],
        inter_domain=["HomogeneousNoMixture"],
        python_path=["/home/chenxiao25/miniconda3/envs/tta/bin/python3"],
        data_path=["../dataset"],
        ckpt_path=[
            "./pretrained_ckpts/pacs/resnet50_bn_ssh_art_pacs.pth",
            # "./pretrained_ckpts/pacs/resnet50_bn_ssh_cartoon.pth",
            # "./pretrained_ckpts/pacs/resnet50_bn_ssh_photo.pth",
            # "./pretrained_ckpts/pacs/resnet50_bn_ssh_sketch.pth",
        ],
        lr=[
            1e-4,
            # 2.5e-4,
        ],
        n_train_steps=[
            1, 
            # 2,
            # 3,
        ],
        entry_of_shared_layers=["layer3"],
        intra_domain_shuffle=["true"],
        record_preadapted_perf=["true"],
        device=[
            "cuda:0",
            "cuda:1",
        ],
        grad_checkpoint=["false"],
        coupled=[
            "data_names",
            "ckpt_path",
        ],
    )
