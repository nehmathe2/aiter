# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Plan checks for get_mla_metadata_v1 (fast_mode) with per-request query lengths.

With uni_seqlen_qo = -1 the v1_2 planner reads each request's query length from
qo_indptr. Above one workgroup's packed rows (96 or 128 heads at qlen > 1) that length
sets the request's qo tiles, so:
  * a uniform batch planned with uni_seqlen_qo = -1 gets the same plan as with
    uni_seqlen_qo = qlen (qlen 3 / 5 / 6, where both run the serial planner);
  * with mixed query lengths, the works of every (request, qo tile) tile its KV range
    exactly, and the reduce maps describe the split tiles.
"""

import sys
from itertools import pairwise

import numpy as np
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.ops.attention import get_mla_decode_shape_support

KV_GRAN = 16
PLAN_ORDER = (
    "work_meta_data",
    "work_info_set",
    "work_indptr",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)
INFO_ORDER = (
    "work_meta_data",
    "work_indptr",
    "work_info_set",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)


def _skip_unless_native(nhead, max_qlen, dtype):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    dt = getattr(dtypes, dtype)
    s = get_mla_decode_shape_support(nhead, max_qlen, dt, dt)
    if not (s.natively_supported and s.native and s.kernel_num_heads == nhead):
        pytest.skip(f"{nhead} heads {dtype} at qlen {max_qlen} are not native here")


def _kv_lens(bs, seed):
    # median 8k: one context-parallel rank's share of a long-context batch
    rng = np.random.default_rng(seed)
    lens = np.exp(np.log(8000) + 0.7 * rng.standard_normal(bs))
    return np.clip(lens, 64, 60000).astype(np.int64).tolist()


def _indptr(lens):
    out = np.zeros(len(lens) + 1, dtype=np.int32)
    out[1:] = np.cumsum(lens)
    return torch.from_numpy(out).cuda()


def _plan(qlens, kv_lens, nhead, dtype, uni_seqlen_qo, cp_rr):
    bs = len(qlens)
    max_qlen = max(qlens)
    dt = getattr(dtypes, dtype)
    info = aiter.get_mla_metadata_info_v1(
        bs, max_qlen, nhead, dt, dt, is_sparse=False, fast_mode=True
    )
    meta = {
        name: torch.zeros(shape, dtype=t, device="cuda")
        for name, (shape, t) in zip(INFO_ORDER, info)
    }
    meta["work_info_set"].fill_(-7)
    aiter.get_mla_metadata_v1(
        _indptr(qlens),
        _indptr(kv_lens),
        torch.ones(bs, dtype=torch.int32, device="cuda"),
        nhead,
        1,
        True,
        *(meta[k] for k in PLAN_ORDER),
        page_size=1,
        kv_granularity=KV_GRAN,
        max_seqlen_qo=max_qlen,
        uni_seqlen_qo=uni_seqlen_qo,
        fast_mode=True,
        max_split_per_batch=-1,
        dtype_q=dt,
        dtype_kv=dt,
        is_cp_round_robin=cp_rr,
    )
    torch.cuda.synchronize()
    return meta


def _check_plan(meta, qlens, kv_lens, cp_rr):
    num_cu = torch.cuda.get_device_properties(0).multi_processor_count
    wip = meta["work_indptr"].cpu().numpy()[: num_cu + 1].astype(np.int64)
    nw = int(wip[-1])
    assert wip[0] == 0 and (np.diff(wip) >= 0).all(), wip
    assert 0 < nw <= meta["work_info_set"].shape[0]
    works = meta["work_info_set"].cpu().numpy()[:nw, :6]
    assert (works != -7).all(), "a work slot was left unwritten"
    ri = meta["reduce_indptr"].cpu().numpy().astype(np.int64)
    rfm = meta["reduce_final_map"].cpu().numpy()
    rpm = meta["reduce_partial_map"].cpu().numpy()

    bs = len(qlens)
    qop = np.concatenate([[0], np.cumsum(qlens)])
    kvp = np.concatenate([[0], np.cumsum(kv_lens)])
    tiles = {}
    for b, part, q0, q1, k0, k1 in works.tolist():
        assert 0 <= b < bs and qop[b] <= q0 < q1 <= qop[b + 1], (b, q0, q1)
        tiles.setdefault((b, q0, q1), []).append((k0, k1, part))

    gti, parts = 0, []
    for b in range(bs):
        mine = sorted(t for t in tiles if t[0] == b)
        # the tiles of a request cover its query tokens in order
        assert mine and mine[0][1] == qop[b] and mine[-1][2] == qop[b + 1], (b, mine)
        assert all(x[2] == y[1] for x, y in pairwise(mine)), (b, mine)
        for i, key in enumerate(mine):
            size = key[2] - key[1]
            tail = 0 if cp_rr else len(mine) - 1 - i
            spans = sorted(tiles[key])
            assert spans[0][0] == kvp[b], (key, spans)
            assert spans[-1][1] == kvp[b + 1] - tail, (key, spans)
            assert all(x[1] == y[0] for x, y in pairwise(spans)), (key, spans)
            if len(spans) == 1:
                assert spans[0][2] == -1 and ri[gti + 1] == ri[gti], (key, spans)
            else:
                locs = sorted(s[2] for s in spans)
                assert ri[gti + 1] - ri[gti] == len(spans), (key, spans)
                assert sorted(rpm[ri[gti] : ri[gti + 1]].tolist()) == locs, (key, spans)
                assert rfm[gti].tolist() == [key[1], key[2]], (key, rfm[gti])
                parts += [(loc, loc + size) for loc in locs]
            gti += 1
    assert ri[0] == 0 and (np.diff(ri[: gti + 1]) >= 0).all()
    assert (ri[gti:] == ri[gti]).all() and ri[gti] <= rpm.shape[0]
    parts.sort()
    assert all(p[0] >= 0 for p in parts), parts[:4]
    assert all(x[1] <= y[0] for x, y in pairwise(parts)), "partials overlap"


@pytest.mark.parametrize("cp_rr", [False, True], ids=["local", "cprr"])
@pytest.mark.parametrize("bs", [1, 8, 33, 128])
@pytest.mark.parametrize("qlen", [3, 5, 6])
@pytest.mark.parametrize("dtype", ["fp8", "bf16"])
@pytest.mark.parametrize("nhead", [96, 128])
def test_varlen_planner_matches_uniform(nhead, dtype, qlen, bs, cp_rr):
    _skip_unless_native(nhead, qlen, dtype)
    qlens = [qlen] * bs
    kv_lens = _kv_lens(bs, seed=bs)
    uniform = _plan(qlens, kv_lens, nhead, dtype, qlen, cp_rr)
    varlen = _plan(qlens, kv_lens, nhead, dtype, -1, cp_rr)
    # work_meta_data holds the buffers' addresses
    for name in PLAN_ORDER[1:]:
        assert torch.equal(uniform[name], varlen[name]), f"{name} differs"
    _check_plan(varlen, qlens, kv_lens, cp_rr)


@pytest.mark.parametrize("cp_rr", [False, True], ids=["local", "cprr"])
@pytest.mark.parametrize("bs", [1, 7, 32, 128])
@pytest.mark.parametrize("dtype", ["fp8", "bf16"])
@pytest.mark.parametrize("nhead", [96, 128])
def test_varlen_plan(nhead, dtype, bs, cp_rr):
    rng = np.random.default_rng(bs)
    qlens = rng.integers(1, 7, size=bs).tolist()
    qlens[-1] = 6
    _skip_unless_native(nhead, max(qlens), dtype)
    kv_lens = _kv_lens(bs, seed=bs + 1)
    meta = _plan(qlens, kv_lens, nhead, dtype, -1, cp_rr)
    _check_plan(meta, qlens, kv_lens, cp_rr)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-rs", *sys.argv[1:]]))
