"""Exact top-K index list without enumerating M: backward max-product over the syndrome trellis
(state = partial syndrome x 'already below M-1' flag) gives an exact heuristic; best-first search then
pops complete codewords in non-increasing score. Checked against brute-force enumeration.

  python -X utf8 trellis_topk.py M r nreads
"""
import heapq
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trellis_index_sim as ab  # noqa: E402


def tables(k, A):
    r = A.shape[1]
    pack = np.array([[int(sum(int(ab.GF_MUL[a, A[j][t]]) << (2 * t) for t in range(r))) for a in range(4)]
                     for j in range(k)])
    return pack, 4 ** r


def heuristic(logp, M, k, A, pack, nsyn):
    """H[b, j, f, s]: best completion score from digit j with flag f (0 tight, 1 below) and syndrome s."""
    B, r = logp.shape[0], A.shape[1]
    bound = ab.digits(M - 1, k)
    syn = np.arange(nsyn)
    par = np.zeros((B, nsyn))
    for t in range(r):
        par += logp[:, k + t, (syn >> (2 * t)) & 3]
    H = np.full((B, k + 1, 2, nsyn), -np.inf)
    H[:, k, 0] = par
    H[:, k, 1] = par
    for j in range(k - 1, -1, -1):
        for a in range(4):
            nxt = syn ^ pack[j, a]
            sc = logp[:, j, a][:, None]
            H[:, j, 1] = np.maximum(H[:, j, 1], sc + H[:, j + 1, 1][:, nxt])
            if a < bound[j]:
                H[:, j, 0] = np.maximum(H[:, j, 0], sc + H[:, j + 1, 1][:, nxt])
            elif a == bound[j]:
                H[:, j, 0] = np.maximum(H[:, j, 0], sc + H[:, j + 1, 0][:, nxt])
    return H, bound


def topk_one(lp, H, bound, k, pack, K):
    out = []
    heap = [(-H[0, 0, 0], 0, 0, 0, 0.0, 0)]      # (-priority, j, flag, syndrome, g, index-so-far)
    while heap and len(out) < K:
        negp, j, f, s, g, idx = heapq.heappop(heap)
        if j == k:
            out.append((idx, -negp))
            continue
        for a in range(4):
            if f == 0 and a > bound[j]:
                continue
            f2 = 1 if (f == 1 or a < bound[j]) else 0
            s2 = s ^ int(pack[j, a])
            g2 = g + lp[j, a]
            h = H[j + 1, f2, s2]
            if np.isfinite(h):
                heapq.heappush(heap, (-(g2 + h), j + 1, f2, s2, g2, idx * 4 + a))
    return out


if __name__ == '__main__':
    M, r, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    k = math.ceil(math.log(M, 4) - 1e-12)
    A = ab.parity_columns(k, r)
    C = ab.codewords(M, k, A)
    rng = np.random.default_rng(11)
    true = rng.integers(M, size=n)
    reads = [ab.channel(ab.strand(C[t], rng), rng) for t in true]
    keep = [b for b, y in enumerate(reads) if abs(len(y) - ab.L) <= ab.D]
    reads, true = [reads[b] for b in keep], true[keep]
    logp = np.log(np.maximum(ab.index_app(reads, k + r), 1e-300))
    pack, nsyn = tables(k, A)
    t0 = time.perf_counter()
    H, bound = heuristic(logp, M, k, A, pack, nsyn)
    lists = [topk_one(logp[b], H[b], bound, k, pack, 16) for b in range(len(reads))]
    t_tree = (time.perf_counter() - t0) / len(reads)
    t0 = time.perf_counter()
    mism = 0
    for b in range(len(reads)):
        s = sum(logp[b, t][C[:, t]] for t in range(k + r))
        top = np.sort(s)[::-1][:16]
        got = np.array([v for _, v in lists[b]])
        if not np.allclose(top, got, atol=1e-9) or any(abs(s[i] - v) > 1e-9 for i, v in lists[b]):
            mism += 1
    t_brute = (time.perf_counter() - t0) / len(reads)
    print(f'M={M} r={r} reads={len(reads)} top16 mismatches={mism} t_trellis_top16={t_tree * 1e3:.2f}ms '
          f't_brute={t_brute * 1e3:.2f}ms', flush=True)
