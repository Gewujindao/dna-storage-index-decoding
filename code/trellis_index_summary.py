"""Summarise chunks: identity penalties (bits/read) for soft, hard, top-k; rate-loss conversion.

  python -X utf8 trellis_index_summary.py M256_r3_s1 M10000_r3_s1 ...
"""
import math
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'results'
L = 110
RW = {5: 1.497, 8: 1.663}
LN2 = math.log(2)


def penalties(lq, rank, logS, M):
    """Per-read penalty arrays (bits) for soft and for list sizes 1, 4, 16 with calibrated off-list mass."""
    out = {'soft': -lq / LN2}
    for ii, k in enumerate((1, 4, 16)):
        inl = rank < k
        dk = 1 - inl.mean()
        on = -(np.log(max(1 - dk, 1e-300)) + lq - logS[:, ii]) / LN2
        off = -(math.log(dk) - math.log(M - k)) / LN2 if dk > 0 else 0.0
        out[k] = np.where(inl, on, off)
    return out


def summary(tag):
    fs = sorted((OUT / tag).glob('chunk_*.npz'))
    d = [np.load(f) for f in fs]
    M = int(tag.split('_')[0][1:])
    lq = np.concatenate([x['logq_true'] for x in d])
    rank = np.concatenate([x['rank'] for x in d])
    logS = np.concatenate([x['logS'] for x in d])
    n = len(lq)
    pen = penalties(lq, rank, logS, M)
    res = dict(tag=tag, M=M, n=n, chunks=len(fs), dropped=int(sum(int(x['dropped']) for x in d)),
               eps=float((rank >= 1).mean()), miss4=float((rank >= 4).mean()), miss16=float((rank >= 16).mean()),
               t_bcjr_ms=1e3 * float(np.mean([x['t_bcjr'] for x in d])),
               t_brute_ms=1e3 * float(np.mean([x['t_brute'] for x in d])),
               t_dp_ms=1e3 * float(np.mean([x['t_dp'] for x in d])),
               dp_err=float(max(float(x['dp_err']) for x in d)))
    for key in ('soft', 1, 4, 16):
        res[f'pen_{key}'] = float(pen[key].mean())
    gap = pen[1] - pen['soft']
    res['gap_bits'] = float(gap.mean())
    res['gap_se'] = float(gap.std(ddof=1) / math.sqrt(n))
    for c, R in RW.items():
        f = c / (L * R)
        res[f'gap_rel_c{c}'] = res['gap_bits'] * f
        res[f'gap_rel_se_c{c}'] = res['gap_se'] * f
        res[f'hard_rel_c{c}'] = res['pen_1'] * f
        res[f'soft_rel_c{c}'] = res['pen_soft'] * f
    for k in (4, 16):
        res[f'recover_{k}'] = (res['pen_1'] - res[f'pen_{k}']) / res['gap_bits'] if res['gap_bits'] > 0 else float('nan')
    return res


if __name__ == '__main__':
    for tag in sys.argv[1:]:
        r = summary(tag)
        print(' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in r.items()), flush=True)
