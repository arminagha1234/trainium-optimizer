"""The orthogonal 2-D TP x EP device mesh for MoE.

The coupled path shards experts and attention across the SAME `world` ranks (EP=TP=
world). The decoupled-with-replication path (TRN_OPT_EP_DEGREE) keeps TP=world and
replicates experts across the spare ranks -- measured flat-to-worse, because the spare
ranks do redundant expert compute instead of useful work.

This module is the third option, the one Megatron / NeuronxDistributed / Pumice use:
arrange the `world` ranks into a 2-D grid of shape (ep, tp) with `tp * ep == world`,
and give attention and the MoE block SEPARATE axes:

    * attention shards across the TP axis (a row): tp ranks that together hold all
      heads, o_proj all-reducing within that row.
    * experts shard across the EP axis (a column): ep ranks that together hold all
      experts, the mixture all-reducing within that column.
    * the router is replicated on every rank.

So no rank does redundant work: each holds 1/tp of the heads AND 1/ep of the experts.
"TP=16, EP=4" on 64 cores is exactly this grid.

Layout (matches the reference convention -- ep rows, tp columns, row-major):

    rank = ep_index * tp + tp_index          ep_index in [0,ep)   tp_index in [0,tp)

    TP group (fixed ep_index): the contiguous row [ep_index*tp : ep_index*tp + tp]
    EP group (fixed tp_index): the strided column {tp_index, tp_index+tp, ...}

The two groups a rank belongs to intersect at exactly that rank, which is the property
that makes the two shardings independent.

`plan_2d_mesh` is pure arithmetic (no torch), so the layout is unit-testable without a
device or a distributed init. `build_2d_groups` turns it into real process subgroups.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MeshPlan", "plan_2d_mesh", "valid_splits", "build_2d_groups"]


@dataclass(frozen=True)
class MeshPlan:
    tp: int            # tensor-parallel degree (attention), = row width
    ep: int            # expert-parallel degree, = column height
    tp_rank: int       # this rank's index within its TP row, in [0, tp)
    ep_rank: int       # this rank's index within its EP column, in [0, ep)
    tp_group: tuple[int, ...]   # the ranks of this rank's TP row
    ep_group: tuple[int, ...]   # the ranks of this rank's EP column


def plan_2d_mesh(world: int, tp: int, ep: int, rank: int) -> MeshPlan:
    """This rank's place in the (ep, tp) grid. Requires ``tp * ep == world``."""
    if tp * ep != world:
        raise ValueError(f"tp*ep must equal world: {tp}*{ep} != {world}")
    if not (0 <= rank < world):
        raise ValueError(f"rank {rank} out of range for world {world}")
    ep_index = rank // tp
    tp_index = rank % tp
    tp_group = tuple(range(ep_index * tp, ep_index * tp + tp))       # contiguous row
    ep_group = tuple(range(tp_index, world, tp))                     # strided column
    return MeshPlan(tp=tp, ep=ep, tp_rank=tp_index, ep_rank=ep_index,
                    tp_group=tp_group, ep_group=ep_group)


def valid_splits(world: int, *, heads: int | None = None,
                 num_experts: int | None = None,
                 powers_of_two_only: bool = True) -> list[tuple[int, int]]:
    """Every (tp, ep) with tp*ep == world that the hardware and the model can express.

    * both tp and ep are powers of two -- a subgroup that is not cannot form a Neuron
      collective (measured: world sizes 3/6/12/24 fail the device barrier), and that
      applies to the TP and EP subgroups just as to the whole world.
    * tp divides the query-head count (attention shards by head), when known.
    * ep divides the expert count (experts shard by expert), when known.
    """
    def _p2(n: int) -> bool:
        return n >= 1 and (n & (n - 1)) == 0

    out = []
    for tp in range(1, world + 1):
        if world % tp:
            continue
        ep = world // tp
        if powers_of_two_only and not (_p2(tp) and _p2(ep)):
            continue
        if heads is not None and heads % tp:
            continue
        if num_experts is not None and num_experts % ep:
            continue
        out.append((tp, ep))
    return out


def build_2d_groups(world: int, tp: int, ep: int, rank: int):
    """Create the TP and EP process subgroups; return (MeshPlan, tp_pg, ep_pg).

    new_group is collective -- EVERY rank must create EVERY group (even ones it does
    not join) or the group it does join will hang. So both full group sets are built
    in a fixed order on all ranks, and each rank keeps handles to the two it is in.
    """
    import torch.distributed as dist

    plan = plan_2d_mesh(world, tp, ep, rank)
    my_tp_pg = my_ep_pg = None
    # TP rows: ep of them, each a contiguous block of tp ranks.
    for e in range(ep):
        members = list(range(e * tp, e * tp + tp))
        pg = dist.new_group(ranks=members)
        if rank in members:
            my_tp_pg = pg
    # EP columns: tp of them, each a strided set of ep ranks.
    for t in range(tp):
        members = list(range(t, world, tp))
        pg = dist.new_group(ranks=members)
        if rank in members:
            my_ep_pg = pg
    return plan, my_tp_pg, my_ep_pg
