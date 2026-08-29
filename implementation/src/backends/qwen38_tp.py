"""Manual tensor-parallel surgery for Qwen3.8-27B (hybrid GatedDeltaNet+attn).
Head-parallel: each rank keeps a whole-head slice; rowwise outputs all-reduce.
Delta rule + attention are per-head, so this is numerically exact."""
import os, torch, torch.nn as nn, torch.distributed as dist
import torch.nn.functional as F

class AllReduceLinear(nn.Module):
    """Rowwise-sharded Linear: local partial matmul, then sum across TP ranks."""
    def __init__(self, lin):
        super().__init__(); self.lin = lin
    def forward(self, x):
        o = self.lin(x)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(o)
        return o

def _slice_linear(lin, rows=None, cols=None):
    W = lin.weight.data
    if rows is not None:
        if rows[1] <= rows[0]:
            raise ValueError(
                f"empty row slice {rows} of a {tuple(W.shape)} weight -- a shard "
                f"went to zero width. This is the bug that reads as 'size of "
                f"tensor a (N) must match tensor b (0)' several layers later; "
                f"raise here instead so the cause is visible.")
        W = W[rows[0]:rows[1], :]
    if cols is not None:
        if cols[1] <= cols[0]:
            raise ValueError(
                f"empty column slice {cols} of a {tuple(W.shape)} weight -- a "
                f"shard went to zero width.")
        W = W[:, cols[0]:cols[1]]
    n = nn.Linear(W.shape[1], W.shape[0], bias=lin.bias is not None, dtype=W.dtype)
    n.weight.data = W.contiguous()
    if lin.bias is not None:
        b = lin.bias.data
        if rows is not None: b = b[rows[0]:rows[1]]
        n.bias.data = b.contiguous()
    return n

def shard_attention(a, r, tp):
    """Head-parallel attention. Handles tp > num_key_value_heads by REPLICATING
    KV heads rather than slicing them to nothing.

    GQA models often have very few KV heads -- Qwen3.5-35B-A3B has 16 query heads
    and only 2 KV heads. At the tp=16 needed to fit its experts, `nkv // tp` is 0,
    so k_proj/v_proj were sliced to zero width. Nothing failed at shard time; the
    model died 6 minutes later inside attention with
    `size of tensor a (2) must match the size of tensor b (0)`.

    Each rank needs the KV head that SERVES its query heads, and when tp exceeds
    the KV-head count several ranks legitimately want the same one. Replicating a
    KV head costs a k/v projection slice and one head of KV cache per rank, which
    is small, and it is what makes tp>nkv reachable at all.
    """
    nh = a.config.num_attention_heads; nkv = a.config.num_key_value_heads; hd = a.head_dim
    qpr = nh // tp
    # KV heads this rank needs, derived from the q heads it owns. Reduces to the
    # old `nkv // tp` / `r * kpr` whenever tp divides nkv, and floors at 1 head
    # instead of 0 when it does not.
    kpr = max(1, (qpr * nkv) // nh)
    kv_start = (r * qpr * nkv) // nh
    # q_proj rows: per head block = hd*2 (q|gate interleaved)
    a.q_proj = _slice_linear(a.q_proj, rows=(r*qpr*hd*2, (r*qpr+qpr)*hd*2))
    a.k_proj = _slice_linear(a.k_proj, rows=(kv_start*hd, (kv_start+kpr)*hd))
    a.v_proj = _slice_linear(a.v_proj, rows=(kv_start*hd, (kv_start+kpr)*hd))
    # o_proj input = nh*hd; slice cols by q-heads, all-reduce
    o = _slice_linear(a.o_proj, cols=(r*qpr*hd, (r*qpr+qpr)*hd))
    if o.bias is not None and r != 0: o.bias.data.zero_()  # add bias once
    a.o_proj = AllReduceLinear(o)
    # repeat_kv reads num_key_value_groups at runtime, so it must describe the
    # LOCAL shapes: qpr query heads over kpr KV heads. Set on the module, never on
    # a.config -- the config object is shared by every layer.
    a.num_key_value_groups = max(1, qpr // kpr)

def shard_deltanet(a, r, tp):
    nkv, nvh = a.num_k_heads, a.num_v_heads          # 16, 48
    hkd, hvd = a.head_k_dim, a.head_v_dim            # 128, 128
    kd, vd = a.key_dim, a.value_dim                  # 2048, 6144
    kpr, vpr = nkv // tp, nvh // tp                  # 4, 12
    # in_proj_qkv rows = [q:0..kd][k:kd..2kd][v:2kd..2kd+vd]; slice each block by head
    W = a.in_proj_qkv.weight.data
    q = W[r*kpr*hkd:(r*kpr+kpr)*hkd]
    k = W[kd + r*kpr*hkd: kd + (r*kpr+kpr)*hkd]
    v = W[2*kd + r*vpr*hvd: 2*kd + (r*vpr+vpr)*hvd]
    newW = torch.cat([q, k, v], 0).contiguous()
    nin = nn.Linear(W.shape[1], newW.shape[0], bias=False, dtype=W.dtype); nin.weight.data = newW
    a.in_proj_qkv = nin
    # conv1d depthwise: same channel slices
    C = a.conv1d.weight.data  # (conv_dim,1,K)
    cq = C[r*kpr*hkd:(r*kpr+kpr)*hkd]; ck = C[kd + r*kpr*hkd: kd + (r*kpr+kpr)*hkd]
    cv = C[2*kd + r*vpr*hvd: 2*kd + (r*vpr+vpr)*hvd]
    newC = torch.cat([cq, ck, cv], 0).contiguous()
    cd = newC.shape[0]
    nc = nn.Conv1d(cd, cd, kernel_size=a.conv1d.kernel_size[0], groups=cd,
                   bias=a.conv1d.bias is not None, padding=a.conv1d.padding[0], dtype=C.dtype)
    nc.weight.data = newC
    if a.conv1d.bias is not None:
        cb = a.conv1d.bias.data
        nc.bias.data = torch.cat([cb[r*kpr*hkd:(r*kpr+kpr)*hkd], cb[kd+r*kpr*hkd:kd+(r*kpr+kpr)*hkd],
                                  cb[2*kd+r*vpr*hvd:2*kd+(r*vpr+vpr)*hvd]]).contiguous()
    a.conv1d = nc
    # in_proj_z (value_dim) by v-head
    a.in_proj_z = _slice_linear(a.in_proj_z, rows=(r*vpr*hvd, (r*vpr+vpr)*hvd))
    # in_proj_a, in_proj_b (num_v_heads) by v-head
    a.in_proj_a = _slice_linear(a.in_proj_a, rows=(r*vpr, (r*vpr+vpr)))
    a.in_proj_b = _slice_linear(a.in_proj_b, rows=(r*vpr, (r*vpr+vpr)))
    # per-v-head params
    a.dt_bias = nn.Parameter(a.dt_bias.data[r*vpr:(r*vpr+vpr)].contiguous())
    a.A_log = nn.Parameter(a.A_log.data[r*vpr:(r*vpr+vpr)].contiguous())
    # out_proj input = value_dim; slice cols by v-head, all-reduce
    a.out_proj = AllReduceLinear(_slice_linear(a.out_proj, cols=(r*vpr*hvd, (r*vpr+vpr)*hvd)))
    # update dims used by forward split/reshape
    a.num_k_heads = kpr; a.num_v_heads = vpr
    a.key_dim = kpr*hkd; a.value_dim = vpr*hvd; a.conv_dim = cd
    # norm is over head_v_dim (128) -> replicate unchanged

def shard_mlp(m, r, tp):
    inter = m.gate_proj.out_features; ipr = inter // tp
    m.gate_proj = _slice_linear(m.gate_proj, rows=(r*ipr, (r*ipr+ipr)))
    m.up_proj = _slice_linear(m.up_proj, rows=(r*ipr, (r*ipr+ipr)))
    d = _slice_linear(m.down_proj, cols=(r*ipr, (r*ipr+ipr)))
    if d.bias is not None and r != 0: d.bias.data.zero_()
    m.down_proj = AllReduceLinear(d)

def shard_model(model, r, tp):
    layers = model.model.layers
    n_attn = n_dn = n_mlp = 0
    for L in layers:
        if hasattr(L, "self_attn"):
            shard_attention(L.self_attn, r, tp); n_attn += 1
        if hasattr(L, "linear_attn"):
            shard_deltanet(L.linear_attn, r, tp); n_dn += 1
        # NOTE: a sparse MoE block has L.mlp.experts + L.mlp.gate and NO top-level
        # gate_proj, so this dense branch SKIPS it — leaving every expert whole on
        # every rank (the Qwen3.5-30B OOM). To shard MoE experts add a shard_moe
        # (expert-TP) branch gated on hasattr(L.mlp, "experts"). Full method +
        # ready snippet: docs/large-model-playbook.md ("MoE placement fix").
        if hasattr(L, "mlp") and hasattr(L.mlp, "gate_proj"):
            shard_mlp(L.mlp, r, tp); n_mlp += 1
    return n_attn, n_dn, n_mlp
