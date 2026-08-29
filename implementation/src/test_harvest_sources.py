# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for harvest_sources — loads the repo-root kernel_sources.yaml. Pure CPU."""

from __future__ import annotations

import harvest_sources as hs


def test_loads_the_repo_sources_file():
    data = hs.load_sources()
    assert isinstance(data, dict) and data, "kernel_sources.yaml should load non-empty"
    # the categories the harvest stage relies on
    assert "kernel_repos" in data and "docs" in data


def test_jburtoft_and_neuron_docs_present():
    urls = " ".join(e.get("url", "") for e in hs.all_sources())
    assert "jburtoft" in urls                       # the kernel goldmine
    assert "awsdocs-neuron" in urls                 # the Neuron docs


def test_all_sources_are_tagged_with_category():
    for e in hs.all_sources():
        assert e.get("category") in ("kernel_repos", "official_libraries",
                                     "tutorials", "docs")
        assert e.get("url")


def test_summary_is_human_readable():
    s = hs.summary()
    assert "Harvest sources" in s and "jburtoft" in s


def test_missing_file_degrades_gracefully(monkeypatch):
    monkeypatch.setenv("TRN_OPT_KERNEL_SOURCES", "/no/such/file.yaml")
    # env points at a missing file, but the walk-up fallback still finds the real
    # one; force a hard miss by also stubbing the finder.
    monkeypatch.setattr(hs, "_find_sources_file", lambda: None)
    assert hs.load_sources() == {}
    assert "none found" in hs.summary()
