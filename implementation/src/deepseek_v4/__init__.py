"""DeepSeek-V4-Flash native-PyTorch bring-up (runbook P0-P9) support code.

See neuron/deepseek-v4-flash-native-pytorch-48xl-runbook.md.
"""
from .p1_reference import build_shrunk_config, build_and_run, CAPTURE_CLASSES, MODEL_ID

__all__ = ["build_shrunk_config", "build_and_run", "CAPTURE_CLASSES", "MODEL_ID"]
