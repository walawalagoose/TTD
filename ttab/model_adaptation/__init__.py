# -*- coding: utf-8 -*-
from .bn_adapt import BNAdapt
from .conjugate_pl import ConjugatePL
from .cotta import CoTTA
from .eata import EATA
from .memo import MEMO
from .no_adaptation import NoAdaptation
from .note import NOTE
from .sar import SAR
from .shot import SHOT
from .t3a import T3A
from .pseudo import Pseudo
from .tent import TENT
from .ttt import TTT
from .ttt_plus_plus import TTTPlusPlus
from .rotta import Rotta
from .tpt import TPT
from .clip_zs import CLIP_ZS
from .deyo import DEYO
from .vida import ViDA
from .tda import TDA
from .boostadapter import BoostAdapter
from .santa import SANTA
from .zero import ZERO
from .dpcore import DPCore
from .batclip import BATCLIP
from .codire import CoDiRe


def get_model_adaptation_method(adaptation_name):
    return {
        "no_adaptation": NoAdaptation,
        "pseudo": Pseudo,
        "tent": TENT,
        "bn_adapt": BNAdapt,
        "memo": MEMO,
        "shot": SHOT,
        "t3a": T3A,
        "ttt": TTT,
        "ttt_plus_plus": TTTPlusPlus,
        "note": NOTE,
        "sar": SAR,
        "conjugate_pl": ConjugatePL,
        "cotta": CoTTA,
        "eata": EATA,
        "rotta": Rotta,
        "tpt": TPT,
        "clip_zs": CLIP_ZS,
        "deyo": DEYO,
        "vida": ViDA,
        "tda": TDA,
        "boostadapter": BoostAdapter,
        "santa": SANTA,
        "zero": ZERO,
        "dpcore": DPCore,
        "batclip": BATCLIP,
        "codire": CoDiRe,
    }[adaptation_name]
