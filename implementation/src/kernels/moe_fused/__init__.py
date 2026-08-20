# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vendored fused MoE megakernel + framework adapter. See ./NOTICE and
./README.md for attribution and the honest on-device gap analysis."""

from .adapter import (
    FUSED_NKI,
    KERNEL_SOURCE,
    MOE_KERNEL_KEY,
    SUPPORTED_CONTRACT,
    is_moe_arch,
    nkilib_available,
    precheck,
    swap_moe_forward,
)

__all__ = [
    "FUSED_NKI",
    "KERNEL_SOURCE",
    "MOE_KERNEL_KEY",
    "SUPPORTED_CONTRACT",
    "is_moe_arch",
    "nkilib_available",
    "precheck",
    "swap_moe_forward",
]
