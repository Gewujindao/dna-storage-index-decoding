"""The original paper's index layout (two q-ary VT component codes, one at the strand start,
one at the strand end; index i = i1*Q + i2 < M) in simulation with the design's channel (fitted on
dataset 1's design half). Exact joint soft posterior in O(Q) = O(sqrt M) per read; hard decision by the full
Q x Q pair matrix. Compared with the paper-style rule (product of symbol marginals, temperature on half).
  python -X utf8 vt_index_sim.py --M 1000000 --nreads 200 --budget 20
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import channel_model as ch  # noqa: E402
import evaluate_dataset1 as ev  # noqa: E402

OUT = HERE.parent / 'results' / 'vt_index'
TAUS = ev.TAUS


def vt_codebook(n):
    """Largest class of the q-ary (q=4) VT construction: sum_i i*alpha_i = a (mod n), sum x = b (mod 4)."""
    words = np.array(np.meshgrid(*[np.arange(4)] * n, indexing='ij')).reshape(n, -1).T
    alpha = (words[:, 1:] >= words[:, :-1]).astype(int)
    a = (alpha * np.arange(1, n)).sum(1) % n
    b = words.sum(1) % 4
    key = a * 4 + b
    best = np.bincount(key).argmax()
    return words[key == best]


def seg_len(Q):
    for n in range(4, 12):
        if len(vt_codebook(n)) >= Q:
            return n


def run(a):
    mods, sims = ev.models()
    model, chan = mods['hmm2x'], sims['fit2x']
    Q = math.ceil(math.sqrt(a.M))
    n = seg_len(Q)
    CB = vt_codebook(n)[:Q]
    F, R = divmod(a.M, Q)
    od = OUT / f'M{a.M}'
    od.mkdir(parents=True, exist_ok=True)
    L, W = ch.L, ch.NS * model.S
    t0 = time.perf_counter()
    for ci in range(a.nreads // a.chunk):
        f = od / f'chunk_{ci:03d}.npz'
        if f.exists():
            continue
        if time.perf_counter() - t0 > a.budget:
            print('budget reached; rerun to continue', flush=True)
            return
        rng = np.random.default_rng([31, a.M, ci])
        rows = []
        while len(rows) < a.chunk:
            i = int(rng.integers(a.M)); i1, i2 = divmod(i, Q)
            x = list(CB[i1]) + list(rng.integers(4, size=L - 2 * n)) + list(CB[i2])
            y = ch.simulate(x, chan, rng)
            if abs(len(y) - L) > ch.D:
                continue
            Y, lens = ch.pad([y])
            ts = time.perf_counter()
            Tm = [ch.step_matrices(model, Y, lens, t) for t in range(L)]
            # start segment: forward for all Q component words
            A0 = np.zeros((Q, W)); A0[:, ch.D * model.S:(ch.D + 1) * model.S] = model.pi0
            for t in range(n):
                new = np.zeros_like(A0)
                for s in range(4):
                    rr_ = CB[:, t] == s
                    new[rr_] = A0[rr_] @ Tm[t][s]
                A0 = new / new.max()
            # end segment: backward for all Q component words
            b = np.zeros(W); fin = lens[0] - L + ch.D
            b[fin * model.S:(fin + 1) * model.S] = 1.0
            B0 = np.repeat(b[None], Q, axis=0)
            for t in range(L - 1, L - n - 1, -1):
                new = np.zeros_like(B0)
                for s in range(4):
                    rr_ = CB[:, t - (L - n)] == s
                    new[rr_] = B0[rr_] @ Tm[t][s].T
                B0 = new / new.max()
            mids = [Tm[t].mean(0) for t in range(n, L - n)]

            def through(v):
                for Mx in mids:
                    v = v @ Mx
                    v = v / np.abs(v).max()
                return v
            # soft: Z over valid pairs and q(true) need only 3 vectors through the middle (O(Q) per read)
            vF = through(np.vstack([A0[:F].sum(0), A0[min(F, Q - 1)] if R else np.zeros(W), A0[i1]]))
            Z = vF[0] @ B0.sum(0) + (vF[1] @ B0[:R].sum(0) if R else 0.0)
            lq_joint = math.log(max(vF[2] @ B0[i2], 1e-300)) - math.log(Z)
            t_soft = time.perf_counter() - ts
            # hard: full pair matrix (all Q start vectors through the middle)
            th = time.perf_counter()
            U = through(A0.copy())
            P = U @ B0.T
            valid = np.zeros((Q, Q), bool); valid[:F] = True
            if R:
                valid[F, :R] = True
            chk = abs(lq_joint - (math.log(max(P[i1, i2], 1e-300)) - math.log(P[valid].sum())))   # O(M) cross-check
            P[~valid] = -np.inf
            j1, j2 = np.unravel_index(np.argmax(P), P.shape)
            hard_ok = (j1 == i1) and (j2 == i2)
            t_hard = time.perf_counter() - th + t_soft
            # paper-style: product of symbol marginals (same channel model), all temperatures
            lp = ch.marginal_logapp(model, [y], L)[0]
            s1 = np.array([lp[np.arange(n), w].sum() for w in CB])
            s2 = np.array([lp[np.arange(L - n, L), w].sum() for w in CB])
            lqm = []
            for tau in TAUS:
                m1, m2 = tau * s1, tau * s2
                e1, e2 = np.exp(m1 - m1.max()), np.exp(m2 - m2.max())
                z = e1[:F].sum() * e2.sum() + (e1[min(F, Q - 1)] * e2[:R].sum() if R else 0.0)
                lqm.append(math.log(e1[i1] * e2[i2]) - math.log(z))
            Pm = s1[:, None] + s2[None, :]
            Pm[~valid] = -np.inf
            k1, k2 = np.unravel_index(np.argmax(Pm), Pm.shape)
            rows.append((lq_joint, hard_ok, lqm, (k1 == i1) and (k2 == i2), t_soft, t_hard, chk))
        np.savez(f, lq_joint=np.array([r[0] for r in rows]), hard_ok=np.array([r[1] for r in rows]),
                 chk=np.array([r[6] for r in rows]),
                 lq_marg=np.array([r[2] for r in rows]), hard_ok_marg=np.array([r[3] for r in rows]),
                 t_soft=np.array([r[4] for r in rows]), t_hard=np.array([r[5] for r in rows]), Q=Q, n=n)
        print(f'M={a.M} Q={Q} n={n} chunk {ci}: eps_joint={1 - np.mean([r[1] for r in rows]):.3f} '
              f't_soft={np.mean([r[4] for r in rows]) * 1e3:.1f}ms t_hard={np.mean([r[5] for r in rows]) * 1e3:.1f}ms', flush=True)
    print('done', a.M, flush=True)


def summary(Ms):
    LN2 = math.log(2)
    for M in Ms:
        d = [np.load(f) for f in sorted((OUT / f'M{M}').glob('chunk_*.npz'))]
        lj = np.concatenate([x['lq_joint'] for x in d]); ok = np.concatenate([x['hard_ok'] for x in d])
        lm = np.concatenate([x['lq_marg'] for x in d]); okm = np.concatenate([x['hard_ok_marg'] for x in d])
        nr = len(lj)
        h = lambda e: (-(1 - e) * math.log2(1 - e) - e * math.log2(e / (M - 1))) if 0 < e < 1 else 0.0
        fit = np.arange(nr) % 2 == 0
        it = int(np.argmin((-lm[fit]).mean(0)))
        ej, em = 1 - ok.mean(), 1 - okm.mean()
        print(f'M={M} Q={int(d[0]["Q"])} n={int(d[0]["n"])} reads={nr} | joint: eps={ej:.4f} soft={(-lj / LN2).mean():.3f} '
              f'hard={h(ej):.3f} gap={h(ej) - (-lj / LN2).mean():.3f} | marginal: eps={em:.4f} tau*={TAUS[it]} '
              f'soft(held)={(-lm[~fit, it] / LN2).mean():.3f} soft(tau=1)={(-lm[:, -2] / LN2).mean():.3f} hard={h(em):.3f} | '
              f'joint soft vs marginal hard: {h(em) - (-lj / LN2).mean():.3f} | t_soft={np.concatenate([x["t_soft"] for x in d]).mean() * 1e3:.1f}ms '
              f't_hard={np.concatenate([x["t_hard"] for x in d]).mean() * 1e3:.1f}ms', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int)
    ap.add_argument('--nreads', type=int, default=200)
    ap.add_argument('--chunk', type=int, default=25)
    ap.add_argument('--budget', type=float, default=18)
    ap.add_argument('--summary', nargs='*', type=int)
    a = ap.parse_args()
    if a.summary:
        summary(a.summary)
    else:
        run(a)
