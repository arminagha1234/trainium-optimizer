# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA (Kimi Delta Attention) kernel package — harvested + on-device-validated
gated delta-rule linear-attention kernels with PER-CHANNEL log-decay, banked into
the framework's kernel corpus (canonical primitive slot ``KDA``).

``nki_kda.py`` (recurrent / decode) and ``nki_kda_chunked_exact.py`` (exact
chunked / prefill) are vendored verbatim under Apache-2.0 (see ./LICENSE).
``adapter.py`` is the generic-injection forward-factory (with prefill/decode
variant dispatch) that lets ``backends.kernel_inject`` reach them. ``kernel.json``
is the registry manifest (status ``passed-on-device`` = rank 4).
"""
