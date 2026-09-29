"""
data_preprocessing.py
---------------------
Alias to src.preprocessing for backward compatibility.
"""

from src.preprocessing import (
    CORE_FEATURES,
    load_dataset,
    inspect_dataset,
    clean_dataset,
    prepare_features_and_target,
    build_preprocessor,
    get_train_test_split,
)

__all__ = [
    "CORE_FEATURES",
    "load_dataset",
    "inspect_dataset",
    "clean_dataset",
    "prepare_features_and_target",
    "build_preprocessor",
    "get_train_test_split",
]
