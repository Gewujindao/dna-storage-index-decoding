"""Exact joint index posterior p(i|y) ∝ P(y | strand with index codeword c_i) under the
IDS channel (no product-of-marginals mismatch), normalised over i < M on the product trellis
(drift x partial syndrome x 'below M-1' flag); cost independent of M.
Hard decision / top-K under the joint metric: rescoring of the top-64 list of the marginal metric
(exactness of the argmax certified when the unlisted mass is below the best listed posterior).

  python -X utf8 trellis_joint.py --M 1000000 --nreads 1000 --chunk 250 --budget 240
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np

import trellis_index_sim as ab
import trellis_topk as tk

OUT = Path(__file__).resolve().parents[1] / 'results' / 'trellis_joint'
NS, D, L = ab.NS, ab.D, ab.L
KL = 64


def backward(Y, lens):
    """Drift-only backward pass (markers known, data uniform); returns beta (B, L+1, NS) and per-step logs."""
    B = Y.shape[0]
    beta = np.zeros((B, L + 1, NS))
    beta[np.arange(B), L, lens - L + D] = 1.0
    for t in range(L - 1, -1, -1):
        G = ab.emissions(Y, lens, t)
        acc = np.zeros((B, NS, 4))
        bn = beta[:, t + 1]
        for l in range(4):
            sh = l - 1
            bs = np.zeros((B, NS))
            if sh < 0:
                bs[:, -sh:] = bn[:, :NS + sh]
            elif sh == 0:
                bs = bn
            else:
                bs[:, :NS - sh] = bn[:, sh:]
            acc += G[l] * bs[:, :, None]
        cur = (acc * ab.PRIOR[t]).sum(axis=2)
        beta[:, t] = cur / cur.sum(axis=1, keepdims=True)
    return beta


def shift_add(dst, src, sh):
    """dst[:, d+sh, ...] += src[:, d, ...] along axis 1 (drift)."""
    if sh < 0:
        dst[:, :NS + sh] += src[:, -sh:]
    elif sh == 0:
        dst += src
    else:
        dst[:, sh:] += src[:, :NS - sh]


def joint_scores(reads, cands, M, k, A):
    """log p_joint(cand | y) for candidate codewords cands (B, K, n), normalised over all i < M."""
    B = len(reads)
    lens = np.array([len(r) for r in reads])
    Y = np.full((B, lens.max() + 4), 4, dtype=np.int64)
    for b, r in enumerate(reads):
        Y[b, :len(r)] = r
    r_ = A.shape[1]
    nsyn = 4 ** r_
    pack, _ = tk.tables(k, A)
    bound = ab.digits(M - 1, k)
    syn = np.arange(nsyn)
    n_idx = k + r_
    idx_pos = ab.INFO_POS[:n_idx]
    T = idx_pos[-1] + 1
    role = {p: ('digit', j) if j < k else ('parity', j - k) for j, p in enumerate(idx_pos)}
    K = cands.shape[1]
    al = np.zeros((B, NS, 2, nsyn))
    al[:, D, 0, 0] = 1.0
    ac = np.zeros((B, K, NS))
    ac[:, :, D] = 1.0
    for t in range(T):
        G = ab.emissions(Y, lens, t)
        nal = np.zeros_like(al)
        nac = np.zeros_like(ac)
        kind = role.get(t, ('marker', None))
        for l in range(4):
            g = G[l]                                      # (B, NS, 4)
            if kind[0] == 'marker':
                v = ab.MARKER_VAL[t]
                shift_add(nal, al * g[:, :, v][:, :, None, None], l - 1)
                _add_cand(nac, ac * g[:, :, v][:, None, :], l - 1)
            elif kind[0] == 'digit':
                j = kind[1]
                for a in range(4):
                    perm = syn ^ pack[j, a]
                    ga = g[:, :, a][:, :, None]
                    part = np.zeros_like(al)
                    part[:, :, 1][:, :, perm] += al[:, :, 1] * ga
                    if a < bound[j]:
                        part[:, :, 1][:, :, perm] += al[:, :, 0] * ga
                    elif a == bound[j]:
                        part[:, :, 0][:, :, perm] += al[:, :, 0] * ga
                    shift_add(nal, part, l - 1)
                sym = cands[:, :, j]                          # (B, K)
                gc = np.take_along_axis(g[:, None, :, :].repeat(K, axis=1), sym[:, :, None, None].repeat(NS, 2), 3)[..., 0]
                _add_cand(nac, ac * gc, l - 1)
            else:
                tp = kind[1]
                symofs = (syn >> (2 * tp)) & 3
                gs = g[:, :, symofs]                          # (B, NS, nsyn)
                shift_add(nal, al * gs[:, :, None, :], l - 1)
                sym = cands[:, :, k + tp]
                gc = np.take_along_axis(g[:, None, :, :].repeat(K, axis=1), sym[:, :, None, None].repeat(NS, 2), 3)[..., 0]
                _add_cand(nac, ac * gc, l - 1)
        sc = nal.sum(axis=(1, 2, 3))
        al = nal / sc[:, None, None, None]
        ac = nac / sc[:, None, None]
    beta = backward(Y, lens)[:, T]                        # (B, NS)
    Z = (al.sum(axis=(2, 3)) * beta).sum(axis=1)
    Lc = (ac * beta[:, None, :]).sum(axis=2)
    return np.log(np.maximum(Lc, 1e-300)) - np.log(Z)[:, None]


def _add_cand(dst, src, sh):
    if sh < 0:
        dst[:, :, :NS + sh] += src[:, :, -sh:]
    elif sh == 0:
        dst += src
    else:
        dst[:, :, sh:] += src[:, :, :NS - sh]


def run(a):
    k = math.ceil(math.log(a.M, 4) - 1e-12)
    A = ab.parity_columns(k, a.r)
    n_idx = k + a.r
    od = OUT / f'M{a.M}_r{a.r}_s{a.seed}'
    od.mkdir(parents=True, exist_ok=True)
    pack, nsyn = tk.tables(k, A)
    t0 = time.perf_counter()
    for ci in range(a.nreads // a.chunk):
        f = od / f'chunk_{ci:03d}.npz'
        if f.exists():
            continue
        if time.perf_counter() - t0 > a.budget:
            print('budget reached; rerun to continue', flush=True)
            return
        rng = np.random.default_rng([a.seed, a.M, a.r, ci, 5])
        true = rng.integers(a.M, size=a.chunk)
        Ct = ab.codewords(a.M, k, A, true)
        reads, keep = [], []
        for b in range(a.chunk):
            y = ab.channel(ab.strand(Ct[b], rng), rng)
            if abs(len(y) - L) <= D:
                reads.append(y)
                keep.append(b)
        true, Ct = true[keep], Ct[keep]
        B = len(reads)
        logp = np.log(np.maximum(ab.index_app(reads, n_idx), 1e-300))
        H, bound = tk.heuristic(logp, a.M, k, A, pack, nsyn)
        lists = [tk.topk_one(logp[b], H[b], bound, k, pack, KL) for b in range(B)]
        ids = np.array([[i for i, _ in lst] for lst in lists])
        cands = np.concatenate([ab.codewords(a.M, k, A, ids.reshape(-1)).reshape(B, KL, n_idx), Ct[:, None, :]], axis=1)
        tb = time.perf_counter()
        lj = joint_scores(reads, cands, a.M, k, A)       # (B, KL+1); last column = true codeword
        t_joint = (time.perf_counter() - tb) / B
        lq_true = lj[:, -1]
        ll = lj[:, :KL]
        order = np.argsort(-ll, axis=1)
        ids_sorted = np.take_along_axis(ids, order, 1)
        ll_sorted = np.take_along_axis(ll, order, 1)
        rank = np.array([list(ids_sorted[b]).index(true[b]) if true[b] in ids_sorted[b] else KL for b in range(B)])
        cs = np.logaddexp.accumulate(ll_sorted, axis=1)
        logS = np.stack([cs[:, kk - 1] for kk in (1, 4, 16, 64)], axis=1)
        unlisted = 1 - np.exp(cs[:, -1])
        certified = unlisted < np.exp(ll_sorted[:, 0])
        np.savez(f, logq_true=lq_true, rank=rank, logS=logS, certified=certified, t_joint=t_joint,
                 marg_rank=np.array([list(ids[b]).index(true[b]) if true[b] in ids[b] else KL for b in range(B)]))
        print(f'M={a.M} chunk {ci}: B={B} eps_joint={np.mean(rank >= 1):.3f} certified={certified.mean():.3f} '
              f't_joint={t_joint * 1e3:.1f}ms', flush=True)
    print('done', a.M, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, required=True)
    ap.add_argument('--r', type=int, default=3)
    ap.add_argument('--nreads', type=int, default=1000)
    ap.add_argument('--chunk', type=int, default=250)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--budget', type=float, default=240)
    run(ap.parse_args())
