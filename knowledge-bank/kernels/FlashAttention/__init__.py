# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashAttention kernel package — the first genuinely-useful INVENTED kernel
banked into the framework's kernel corpus.

``flash_nki_opt.py`` is the on-device-validated streaming / online-softmax
flash-attention NKI kernel (return-form ``neuronxcc.nki``). ``adapter.py`` is the
generic-injection forward-factory that lets ``backends.kernel_inject`` reach it.
``kernel.json`` is the registry manifest (status ``passed-on-device`` = rank 4).
"""
