# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mamba2 kernel package — the harvested + on-device-validated Mamba-2 SSD
(state-space duality) selective-scan kernel, banked into the framework's kernel
corpus (canonical primitive slot ``Mamba2``; harvest identity ``Mamba2SSD``).

The kernel source itself lives in the installed ``nkilib`` package
(``nkilib.experimental.scan.ssd``) and is referenced, not vendored — see
``adapter.py`` and ``kernel.json`` for provenance. ``adapter.py`` is the
generic-injection forward-factory that lets ``backends.kernel_inject`` reach it.
``kernel.json`` is the registry manifest (status ``passed-on-device`` = rank 4).
"""
