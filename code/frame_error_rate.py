"""Small outer-code frame-error calculation that turns index-decision quality into
outer code rate / coverage. Assumptions: ideal MDS outer code over strands, block of
N_O = 1000 strands; inner decoding perfect; index decision is the only error source; reads per strand
Poisson(c); a read is accepted iff its maximum posterior >= theta; an accepted wrong read contaminates one
other strand (error); a strand with no accepted correct read and no contamination is an erasure; a block
decodes iff 2*errors + erasures <= N_O*(1 - R). (max posterior, correct) pairs are the held-out empirical
distribution. Exact multinomial tail (errors, erasures).
  python -X utf8 frame_error_rate.py <eval_dir_baseline> <tau_baseline> <eval_dir_design> <tau_design>
"""
import math
import sys
from pathlib import Path

import numpy as np

N_O, TARGET = 1000, 1e-3
TAUS = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2])


def load(d, tau):
    fs = sorted(Path(d).glob('chunk_*.npz'))
    it = int(np.where(np.isclose(TAUS, tau))[0][0])
    conf = np.concatenate([np.exp(np.load(f)['conf_tau'][:, it]) for f in fs])
    ok = np.concatenate([np.load(f)['rank'] == 0 for f in fs])
    return conf, ok


LF = np.array([math.lgamma(i + 1) for i in range(N_O + 2)])


def logbinom_pmf(n, k, p):
    k = np.asarray(k)
    if p <= 0:
        return np.where(k == 0, 0.0, -np.inf)
    if p >= 1:
        return np.where(k == n, 0.0, -np.inf)
    return LF[n] - LF[k] - LF[n - k] + k * math.log(p) + (n - k) * math.log1p(-p)


def fer(pe, px, R):
    """P(2E + X > N_O (1 - R)) with (E, X) multinomial(N_O; pe, px)."""
    budget = math.floor(N_O * (1 - R) + 1e-9)
    E = np.arange(0, budget // 2 + 1)
    lpE = logbinom_pmf(N_O, E, pe)
    q = px / (1 - pe) if pe < 1 else 0.0
    X = np.arange(0, budget + 1)
    m = (N_O - E)[:, None]
    Xm = np.broadcast_to(X[None, :], (len(E), len(X)))
    mask = Xm <= (budget - 2 * E)[:, None]
    if q <= 0:
        lx = np.where(Xm == 0, 0.0, -np.inf)
    elif q >= 1:
        lx = np.where(Xm == m, 0.0, -np.inf)
    else:
        Xc = np.minimum(Xm, m)
        lx = LF[m] - LF[Xc] - LF[m - Xc] + Xc * math.log(q) + (m - Xc) * math.log1p(-q)
        lx = np.where(Xm <= m, lx, -np.inf)
    ok = (np.exp(lpE[:, None] + lx) * mask).sum()
    return 1 - ok


def best_rate(conf, ok, c, thetas):
    best = (0.0, None)
    for th in thetas:
        acc = conf >= th
        a, b = np.mean(acc & ok), np.mean(acc & ~ok)
        pe = 1 - math.exp(-c * b)
        px = math.exp(-c * b) * math.exp(-c * a)
        lo, hi = 0.0, 1.0                     # largest R with FER <= TARGET (FER increases with R)
        if fer(pe, px, 0.0) > TARGET:
            continue
        for _ in range(10):
            mid = (lo + hi) / 2
            if fer(pe, px, mid) <= TARGET:
                lo = mid
            else:
                hi = mid
        if lo > best[0]:
            best = (lo, th)
    return best


def main():
    db, tb, dd, td = sys.argv[1], float(sys.argv[2]), sys.argv[3], float(sys.argv[4])
    thetas = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999]
    rows = {}
    for nm, d, tau in (('baseline', db, tb), ('design', dd, td)):
        conf, ok = load(d, tau)
        rows[nm] = (conf, ok)
        print(f'{nm}: reads={len(ok)} eps={1 - ok.mean():.4f}', flush=True)
    cs = np.arange(1.0, 16.01, 1.0)
    table = {nm: [best_rate(*rows[nm], c, thetas) for c in cs] for nm in rows}
    for c in (2, 3, 4, 5, 6, 8):
        i = int(np.where(np.isclose(cs, c))[0][0])
        print(f'coverage {c}: ' + ' | '.join(f'{nm} R*={table[nm][i][0]:.3f} (theta {table[nm][i][1]})' for nm in rows), flush=True)
    for R in (0.7, 0.8, 0.9):
        out = []
        for nm in rows:
            cmin = next((float(c) for c, (r, _) in zip(cs, table[nm]) if r >= R), None)
            out.append(f'{nm} c_min={cmin}')
        print(f'outer rate {R}: ' + ' | '.join(out), flush=True)
    print('done fer', flush=True)


if __name__ == '__main__':
    main()
