# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeltaNet kernel package — harvested + on-device-validated Qwen3.5 / Qwen3-Next
Gated DeltaNet (gated delta-rule) NKI kernels, banked into the framework's kernel
corpus (canonical primitive slot ``DeltaNet``; harvest identity ``GatedDeltaNet``).

``gdn_src/`` vendors the kernel package verbatim under Apache-2.0 (see ./NOTICE):
``gdn_src/nki_kernels/*.py`` (@nki.jit kernels — recurrent, tkg decode, and
chunked-parallel forward/backward) plus ``gdn_src/constants.py``. ``adapter.py``
is the generic-injection forward-factory (with shape-based variant dispatch) that
lets ``backends.kernel_inject`` reach them. ``kernel.json`` is the registry
manifest (status ``passed-on-device`` = rank 4).
"""
