# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bare package marker for the vendored Gated DeltaNet NKI kernels.

Reduced from the upstream eager re-export __init__ so the adapter can lazily
import a single kernel submodule without importing all seven. See ../../NOTICE.
"""
