"""Summarise trellis_joint runs: identity penalties under the exact joint posterior (soft), the rescored hard
decision and top-K lists (K <= 64, from the marginal top-64), and the rate-loss conversion.

  python -X utf8 trellis_joint_summary.py 10000 100000 1000000
"""
import math
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'results' / 'trellis_joint'
LN2 = math.log(2)
F5 = 5 / (110 * 1.497)

for M in map(int, sys.argv[1:]):
    d = [np.load(f) for f in sorted((OUT / f'M{M}_r3_s1').glob('chunk_*.npz'))]
    lq = np.concatenate([x['logq_true'] for x in d])
    rank = np.concatenate([x['rank'] for x in d])
    mrank = np.concatenate([x['marg_rank'] for x in d])
    logS = np.concatenate([x['logS'] for x in d])
    cert = np.concatenate([x['certified'] for x in d])
    n = len(lq)
    soft = -lq / LN2
    pen = {}
    for ii, kk in enumerate((1, 4, 16, 64)):
        inl = rank < kk
        dk = 1 - inl.mean()
        on = -(np.log(max(1 - dk, 1e-300)) + lq - logS[:, ii]) / LN2
        off = -(math.log(dk) - math.log(M - kk)) / LN2 if dk > 0 else 0.0
        pen[kk] = np.where(inl, on, off)
    gap = pen[1] - soft
    print(f'M={M} n={n} eps_joint={np.mean(rank >= 1):.4f} eps_marginal={np.mean(mrank >= 1):.4f} '
          f'certified={cert.mean():.3f} true_outside_marg64={np.mean(mrank >= 64):.4f} '
          f'pen_soft_joint={soft.mean():.3f} pen_hard_joint={pen[1].mean():.3f} gap={gap.mean():.3f}+-{gap.std(ddof=1) / math.sqrt(n):.3f} '
          f'gap_rel_c5={gap.mean() * F5:.4f} hard_rel_c5={pen[1].mean() * F5:.4f} soft_rel_c5={soft.mean() * F5:.4f} '
          f'rec4={(pen[1].mean() - pen[4].mean()) / gap.mean():.3f} rec16={(pen[1].mean() - pen[16].mean()) / gap.mean():.3f} '
          f'rec64={(pen[1].mean() - pen[64].mean()) / gap.mean():.3f} t_joint_ms={1e3 * np.mean([float(x["t_joint"]) for x in d]):.2f}',
          flush=True)
