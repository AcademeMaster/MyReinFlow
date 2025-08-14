# Scripts模块
# 包含各种训练脚本

__version__ = "0.1.0"

from .train_behavioral_cloning import main as train_bc
from .train_fql import main as train_fql

__all__ = [
    "train_bc",
    "train_fql",
]
