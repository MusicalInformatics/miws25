#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The top level of the package contains functions to extract score information
with basis functions and create performance codecs
"""

from pathlib import Path

from basismixer.performance_codec import (
    PerformanceCodec,
    OnsetwiseDecompositionDynamicsCodec,
    TimeCodec,
)
from basismixer.data import make_datasets

# define a version variable
__version__ = "0.1.2"

# Base directory (directory of this file)
BASE_DIR = Path(__file__).resolve().parent

# An example basis configuration for didactic purposes
BASIS_CONFIG_EXAMPLE = BASE_DIR / "assets" / "basis_config_example.json"

MODEL_CONFIG_EXAMPLE = BASE_DIR / "assets" / "model_config_example.json"

# A small trained model for didactic purposes
VD_TIM_ART_TRAINED_MODEL_CONFIG_EXAMPLE = (
    BASE_DIR
    / "assets"
    / "sample_models"
    / "velocity_dev-timing-articulation_log-notewise"
    / "config.json"
)

VD_TIM_ART_TRAINED_MODEL_PARAMS_EXAMPLE = (
    BASE_DIR
    / "assets"
    / "sample_models"
    / "velocity_dev-timing-articulation_log-notewise"
    / "best_model.pth"
)

VT_BP_TRAINED_MODEL_CONFIG_EXAMPLE = (
    BASE_DIR
    / "assets"
    / "sample_models"
    / "velocity_trend-beat_period_ratio_log-onsetwise"
    / "config.json"
)

VT_BP_TRAINED_MODEL_PARAMS_EXAMPLE = (
    BASE_DIR
    / "assets"
    / "sample_models"
    / "velocity_trend-beat_period_ratio_log-onsetwise"
    / "best_model.pth"
)

TOY_MODEL_CONFIG = [
    [
        VD_TIM_ART_TRAINED_MODEL_CONFIG_EXAMPLE,
        VD_TIM_ART_TRAINED_MODEL_PARAMS_EXAMPLE,
    ],
    [
        VT_BP_TRAINED_MODEL_CONFIG_EXAMPLE,
        VT_BP_TRAINED_MODEL_PARAMS_EXAMPLE,
    ],
]