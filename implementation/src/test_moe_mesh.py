"""The 2-D mesh layout must be arithmetically sound before any collective runs on it.

A wrong grid -- a rank in the wrong row or column -- would shard attention or experts
across the wrong ranks and silently corrupt the output. These are pure-math checks of
the (ep, tp) partition, so they run with no device and no distributed init.
"""
from __future__ import annotations

import pytest

from backends.moe_mesh import MeshPlan, plan_2d_mesh, valid_splits


def test_tp_and_ep_groups_intersect_at_exactly_this_rank():
    """The independence property: a rank's TP row and EP column share only itself.

    That is what lets attention reduce along the row and experts along the column
    without the two shardings interfering.
    """
    for world, tp, ep in [(64, 16, 4), (64, 8, 8), (8, 2, 4), (8, 4, 2), (4, 2, 2)]:
        for rank in range(world):
            p = plan_2d_mesh(world, tp, ep, rank)
            shared = set(p.tp_group) & set(p.ep_group)
            assert shared == {rank}, (world, tp, ep, rank, shared)


def test_tp_rows_partition_the_world():
    """Every rank in exactly one TP row; rows are contiguous blocks of size tp."""
    world, tp, ep = 64, 16, 4
    rows = {}
    for rank in range(world):
        rows.setdefault(plan_2d_mesh(world, tp, ep, rank).tp_group, []).append(rank)
    assert len(rows) == ep                       # ep distinct rows
    for row in rows:
        assert len(row) == tp                    # each tp wide
    covered = sorted(r for row in rows for r in row)
    assert covered == list(range(world))         # every rank once


def test_ep_columns_partition_the_world():
    """Every rank in exactly one EP column; columns are strided sets of size ep."""
    world, tp, ep = 64, 16, 4
    cols = {}
    for rank in range(world):
        cols.setdefault(plan_2d_mesh(world, tp, ep, rank).ep_group, []).append(rank)
    assert len(cols) == tp                        # tp distinct columns
    for col in cols:
        assert len(col) == ep                     # each ep tall
    covered = sorted(r for col in cols for r in col)
    assert covered == list(range(world))


def test_ranks_within_a_group_span_the_full_index_range():
    """tp_rank covers [0,tp) within a row; ep_rank covers [0,ep) within a column --
    so the head/expert partition over the group is complete."""
    world, tp, ep = 64, 8, 8
    for row_start in range(0, world, tp):
        ranks = list(range(row_start, row_start + tp))
        tp_ranks = sorted(plan_2d_mesh(world, tp, ep, r).tp_rank for r in ranks)
        assert tp_ranks == list(range(tp))
    for col_start in range(tp):
        ranks = list(range(col_start, world, tp))
        ep_ranks = sorted(plan_2d_mesh(world, tp, ep, r).ep_rank for r in ranks)
        assert ep_ranks == list(range(ep))


def test_coupled_default_is_the_degenerate_grid():
    """ep=1 is pure TP=world (one row); tp=1 is pure EP=world (one column)."""
    world = 8
    for rank in range(world):
        p = plan_2d_mesh(world, world, 1, rank)     # tp=world, ep=1
        assert p.tp_group == tuple(range(world)) and p.ep_group == (rank,)
        q = plan_2d_mesh(world, 1, world, rank)     # tp=1, ep=world
        assert q.ep_group == tuple(range(world)) and q.tp_group == (rank,)


def test_bad_split_is_rejected():
    with pytest.raises(ValueError):
        plan_2d_mesh(64, 16, 8, 0)                   # 16*8 != 64


def test_valid_splits_are_power_of_two_pairs_multiplying_to_world():
    splits = valid_splits(64)
    assert splits == [(1, 64), (2, 32), (4, 16), (8, 8), (16, 4), (32, 2), (64, 1)]
    for tp, ep in splits:
        assert tp * ep == 64
        assert tp & (tp - 1) == 0 and ep & (ep - 1) == 0


def test_valid_splits_respect_head_and_expert_divisibility():
    # 24 heads: tp must divide 24 AND be a power of two -> tp in {1,2,4,8}
    # 128 experts: ep divides 128 (all powers of two <=64 do)
    splits = valid_splits(64, heads=24, num_experts=128)
    tps = {tp for tp, _ in splits}
    assert tps <= {1, 2, 4, 8}
    assert (8, 8) in splits and (16, 4) not in splits   # tp=16 doesn't divide 24
