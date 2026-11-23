"""How much of the soft-hard gap a top-K list recovers, K = 1..1024, versus the full
(untruncated) posterior; exact ranks by enumeration. Resumable per chunk.

  python -X utf8 trellis_list_size.py --M 1000000 --nreads 1000 --chunk 250 --budget 240
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np

import trellis_index_sim as ab

KS = (1, 4, 16, 64, 256, 1024)
OUT = Path(__file__).resolve().parents[1] / 'results' / 'trellis_list_size'


def run(a):
    k = math.ceil(math.log(a.M, 4) - 1e-12)
    A = ab.parity_columns(k, a.r)
    n_idx = k + a.r
    od = OUT / f'M{a.M}_r{a.r}_s{a.seed}'
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    C = ab.codewords(a.M, k, A)
    for ci in range(a.nreads // a.chunk):
        f = od / f'chunk_{ci:03d}.npz'
        if f.exists():
            continue
        if time.perf_counter() - t0 > a.budget:
            print('budget reached; rerun to continue', flush=True)
            return
        rng = np.random.default_rng([a.seed, a.M, a.r, ci, 2])
        true = rng.integers(a.M, size=a.chunk)
        reads, keep = [], []
        for b in range(a.chunk):
            y = ab.channel(ab.strand(C[true[b]], rng), rng)
            if abs(len(y) - ab.L) <= ab.D:
                reads.append(y)
                keep.append(b)
        true = true[keep]
        logp = np.log(np.maximum(ab.index_app(reads, n_idx), 1e-300))
        B = len(reads)
        lq, rank, logS = np.zeros(B), np.zeros(B, dtype=np.int64), np.zeros((B, len(KS)))
        for b in range(B):
            s = logp[b, 0][C[:, 0]].astype(np.float64)
            for t in range(1, n_idx):
                s += logp[b, t][C[:, t]]
            mx = s.max()
            lz = mx + math.log(np.exp(s - mx).sum())
            lq[b] = s[true[b]] - lz
            rank[b] = int((s > s[true[b]]).sum())
            kmax = min(KS[-1], a.M)
            top = np.sort(np.partition(s, -kmax)[-kmax:])[::-1]
            cs = np.logaddexp.accumulate(top)
            for ii, kk in enumerate(KS):
                logS[b, ii] = cs[min(kk, kmax) - 1] - lz
        np.savez(f, logq_true=lq, rank=rank, logS=logS, dropped=a.chunk - B)
        print(f'M={a.M} chunk {ci}: B={B} eps={np.mean(rank >= 1):.3f} miss1024={np.mean(rank >= 1024):.4f}', flush=True)
    print('done', a.M, flush=True)


def summary(M, r=3, seed=1):
    LN2 = math.log(2)
    d = [np.load(f) for f in sorted((OUT / f'M{M}_r{r}_s{seed}').glob('chunk_*.npz'))]
    lq = np.concatenate([x['logq_true'] for x in d])
    rank = np.concatenate([x['rank'] for x in d])
    logS = np.concatenate([x['logS'] for x in d])
    pen = {'soft': -lq / LN2}
    for ii, kk in enumerate(KS):
        inl = rank < kk
        dk = 1 - inl.mean()
        on = -(np.log(max(1 - dk, 1e-300)) + lq - logS[:, ii]) / LN2
        off = -(math.log(dk) - math.log(M - kk)) / LN2 if dk > 0 else 0.0
        pen[kk] = np.where(inl, on, off)
    gap = pen[1] - pen['soft']
    out = dict(M=M, n=len(lq), pen_soft=pen['soft'].mean(), pen_hard=pen[1].mean(), gap=gap.mean(),
               gap_se=gap.std(ddof=1) / math.sqrt(len(lq)))
    for kk in KS[1:]:
        out[f'rec{kk}'] = (pen[1].mean() - pen[kk].mean()) / gap.mean()
        out[f'miss{kk}'] = float((rank >= kk).mean())
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int)
    ap.add_argument('--r', type=int, default=3)
    ap.add_argument('--nreads', type=int, default=1000)
    ap.add_argument('--chunk', type=int, default=250)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--budget', type=float, default=240)
    ap.add_argument('--summary', nargs='*', type=int)
    a = ap.parse_args()
    if a.summary:
        for M in a.summary:
            print(' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in summary(M).items()))
    else:
        run(a)
