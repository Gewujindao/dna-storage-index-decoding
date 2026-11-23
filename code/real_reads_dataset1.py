"""Real nanopore reads (Microsoft clustered reads, MIT) versus the i.i.d. IDS model.

(1) Align a sample of reads to their design strand (banded unit-cost edit distance + traceback):
    error rates, position profile, burstiness, length spread.
(2) Index-like test: the first NC symbols of the 10,000 design strands form an (unstructured) index set.
    Mismatched index posterior from the read (forward-backward, no markers, rates from (1)),
    soft / hard / top-k identity penalties by enumeration over the 10,000 candidates; the same
    pipeline on reads simulated from the same strands with the i.i.d. model at the same rates.
  python -X utf8 real_reads_dataset1.py
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trellis_index_sim as ab  # noqa: E402

DATA = HERE.parent / 'downloads' / 'microsoft_clustered_nanopore'
OUT = HERE.parent / 'results' / 'real_reads_dataset1'
NSAMPLE, NC, BAND = 3000, 10, 15
B2I = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def load():
    centers = [l.strip() for l in (DATA / 'Centers.txt').read_text().split('\n') if l.strip()]
    clusters, cur = [], None
    for line in (DATA / 'Clusters.txt').read_text().split('\n'):
        s = line.strip()
        if not s:
            continue
        if set(s) == {'='}:
            if cur is not None:
                clusters.append(cur)
            cur = []
        else:
            cur.append(s)
    clusters.append(cur)
    assert len(centers) == 10000 and len(clusters) == 10000, (len(centers), len(clusters))
    return centers, clusters


def align(x, y):
    """Banded unit-cost global alignment; returns per-x-position ops ('M','S','D') and insertions before each position."""
    n, m = len(x), len(y)
    INF = 10 ** 9
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0
    for j in range(1, min(m, BAND) + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        lo, hi = max(0, i - BAND), min(m, i + BAND)
        Di, Dp = D[i], D[i - 1]
        if lo == 0:
            Di[0] = i
        xi = x[i - 1]
        for j in range(max(1, lo), hi + 1):
            v = Dp[j - 1] + (xi != y[j - 1])
            if Dp[j] + 1 < v:
                v = Dp[j] + 1
            if Di[j - 1] + 1 < v:
                v = Di[j - 1] + 1
            Di[j] = v
    dist = D[n][m]
    ops, ins = [''] * n, [0] * (n + 1)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (x[i - 1] != y[j - 1]):
            ops[i - 1] = 'M' if x[i - 1] == y[j - 1] else 'S'
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops[i - 1] = 'D'
            i -= 1
        else:
            ins[i] += 1
            j -= 1
    return dist, ops, ins


def penalties(lq, rank, logS, M, ks):
    LN2 = math.log(2)
    out = {'soft': float(np.mean(-lq / LN2))}
    for ii, kk in enumerate(ks):
        inl = rank < kk
        dk = 1 - inl.mean()
        on = -(np.log(max(1 - dk, 1e-300)) + lq - logS[:, ii]) / LN2
        off = -(math.log(dk) - math.log(M - kk)) / LN2 if dk > 0 else 0.0
        out[kk] = float(np.mean(np.where(inl, on, off)))
    return out


def index_test(reads, true, C, tag):
    ks = (1, 4, 16, 64)
    app = ab.index_app(reads, NC, markers=False)
    logp = np.log(np.maximum(app, 1e-300))
    B, M = len(reads), C.shape[0]
    taus = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    lq_tau = np.zeros((B, len(taus)))
    lq, rank, logS = np.zeros(B), np.zeros(B, dtype=np.int64), np.zeros((B, len(ks)))
    for b in range(B):
        s = logp[b, 0][C[:, 0]].astype(np.float64)
        for t in range(1, NC):
            s += logp[b, t][C[:, t]]
        mx = s.max()
        lz = mx + math.log(np.exp(s - mx).sum())
        st = s[true[b]]
        for it, tau in enumerate(taus):           # tempered metric q^tau (calibration of over-confidence)
            lq_tau[b, it] = tau * (st - mx) - math.log(np.exp(tau * (s - mx)).sum())
        lq[b] = st - lz
        rank[b] = int((s > st).sum())
        top = np.sort(np.partition(s, -64)[-64:])[::-1]
        cs = np.logaddexp.accumulate(top)
        for ii, kk in enumerate(ks):
            logS[b, ii] = cs[kk - 1] - lz
    pen = penalties(lq, rank, logS, M, ks)
    gap = pen[1] - pen['soft']
    print(f'[{tag}] reads={B} eps={np.mean(rank >= 1):.4f} miss16={np.mean(rank >= 16):.4f} '
          f'pen_soft={pen["soft"]:.3f} pen_hard={pen[1]:.3f} gap={gap:.3f} '
          f'rec4={(pen[1] - pen[4]) / gap:.3f} rec16={(pen[1] - pen[16]) / gap:.3f} rec64={(pen[1] - pen[64]) / gap:.3f}',
          flush=True)
    pt = -lq_tau / math.log(2)
    fit, ev = np.arange(B) % 2 == 0, np.arange(B) % 2 == 1      # choose tau on even reads, report odd reads
    best = int(np.argmin(pt[fit].mean(axis=0)))
    hard_ev = penalties(lq[ev], rank[ev], logS[ev], M, ks)[1]
    print(f'[{tag}] tempered soft: tau*={taus[best]} pen_soft_tau(held-out)={pt[ev, best].mean():.3f} '
          f'pen_soft_tau1(held-out)={pt[ev, -1].mean():.3f} pen_hard(held-out)={hard_ev:.3f} '
          f'gap_tau(held-out)={hard_ev - pt[ev, best].mean():.3f}', flush=True)
    np.savez(OUT / f'index_{tag}.npz', logq_true=lq, rank=rank, logS=logS, lq_tau=lq_tau, taus=taus)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    centers, clusters = load()
    sizes = np.array([len(c) for c in clusters])
    print(f'clusters={len(clusters)} empty={int((sizes == 0).sum())} reads={int(sizes.sum())} '
          f'median_size={int(np.median(sizes))}', flush=True)
    pool = [(ci, ri) for ci, c in enumerate(clusters) for ri in range(len(c))]
    rng = np.random.default_rng(2026)
    pick = rng.choice(len(pool), size=NSAMPLE, replace=False)
    lens, dists, kept = [], [], []
    nsub = ndel = nins = nmat = 0
    pos_err = np.zeros(110)
    pos_n = np.zeros(110)
    pairs = np.zeros((2, 2))
    for p in pick:
        ci, ri = pool[p]
        x, y = centers[ci], clusters[ci][ri]
        lens.append(len(y))
        if abs(len(y) - len(x)) > BAND:
            dists.append(np.nan)
            continue
        dist, ops, ins = align(x, y)
        dists.append(dist / len(x))
        if dist / len(x) > 0.3:
            continue
        kept.append((ci, y))
        e = np.array([(o != 'M') or (ins[t] > 0) for t, o in enumerate(ops)], dtype=int)
        nsub += ops.count('S'); ndel += ops.count('D'); nmat += ops.count('M'); nins += sum(ins)
        pos_err += e
        pos_n += 1
        for t in range(len(e) - 1):
            pairs[e[t], e[t + 1]] += 1
    dists = np.array(dists)
    nx = len(kept) * 110
    p_s = nsub / (nsub + nmat)
    p_d = ndel / nx
    p_i = nins / (nx + nins)
    lens = np.array(lens)
    print(f'sampled={NSAMPLE} kept(dist<=0.3)={len(kept)} band_fail={int(np.isnan(dists).sum())} '
          f'dist_median={np.nanmedian(dists):.3f} dist>0.3={np.nanmean(dists > 0.3):.3f}', flush=True)
    print(f'rates: ins={p_i:.4f} del={p_d:.4f} sub={p_s:.4f} total_per_symbol={(nsub + ndel + nins) / nx:.4f}', flush=True)
    print(f'read length: mean={lens.mean():.1f} sd={lens.std():.1f} (i.i.d. model sd ~ {math.sqrt(110 * (p_i + p_d)):.1f})', flush=True)
    prof = pos_err / pos_n
    print('error rate by position: first10={:.4f} 10-99={:.4f} last10={:.4f}'.format(
        prof[:10].mean(), prof[10:100].mean(), prof[100:].mean()), flush=True)
    pe = pairs[:, 1].sum() / pairs.sum()
    print(f'burstiness: P(err)={pe:.4f} P(err|prev err)={pairs[1, 1] / pairs[1].sum():.4f}', flush=True)
    np.savez(OUT / 'channel_stats.npz', prof=prof, pairs=pairs, lens=lens, dists=dists,
             rates=np.array([p_i, p_d, p_s]))
    # index-like test on real reads and on i.i.d. simulated reads from the same strands
    ab.set_rates(p_i, p_d, p_s)
    C = np.array([[B2I[ch] for ch in s[:NC]] for s in centers], dtype=np.uint8)
    nuniq = len({bytes(r) for r in C})
    print(f'index set: first {NC} symbols of 10,000 strands, distinct={nuniq}', flush=True)
    real = [(ci, [B2I[ch] for ch in y]) for ci, y in kept if abs(len(y) - ab.L) <= ab.D]
    index_test([y for _, y in real], np.array([ci for ci, _ in real]), C, 'real')
    sim = [(ci, ab.channel([B2I[ch] for ch in centers[ci]], rng)) for ci, _ in real]
    sim = [(ci, y) for ci, y in sim if abs(len(y) - ab.L) <= ab.D]
    index_test([y for _, y in sim], np.array([ci for ci, _ in sim]), C, 'sim_iid')
    print('done realreads', flush=True)


if __name__ == '__main__':
    main()
