"""Hard vs soft index decoding as the DNA-storage pool grows.

Channel: i.i.d. Davey-MacKay IDS (Welter et al. 2026 file3 acc-BC rates).
Index: k = ceil(log4 M) GF(4) digits (MSB first) + r parity symbols of a shortened GF(4)
Hamming code (r=3, d_min=3) or no parity (r=0), placed at the strand start; uniform data follows.
Per read: full forward-backward over the strand -> index symbol APPs -> mismatched index
posterior q(i|y) ∝ prod_s P_s(c_s(i)) over i < M (Welter's metric).

Usage (resumable; each call stops after --budget seconds):
  python -X utf8 trellis_index_sim.py --M 1000000 --r 3 --nreads 2000 --chunk 250 --budget 240
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np

L = 110
P_I, P_D, P_S = 0.009, 0.014, 0.020
D = 10                      # drift window +-D
NS = 2 * D + 1
# Inner marker code: Welter's acc-BC final inner rate 192/110 bit/nt = 96 info + 14 marker symbols.
# Modelled as a known marker symbol at every 8th position (13 markers, 97 info symbols); the
# marker values and positions are ours (the paper uses a marker-repeat code not read in detail).
MARKER_PERIOD = 8
MARKER_POS = [p for p in range(L) if (p + 1) % MARKER_PERIOD == 0]
INFO_POS = [p for p in range(L) if (p + 1) % MARKER_PERIOD != 0]
MARKER_VAL = {p: (3 * p + 1) % 4 for p in MARKER_POS}
PRIOR = np.full((L, 4), 0.25)
for _p, _v in MARKER_VAL.items():
    PRIOR[_p] = 0.0
    PRIOR[_p, _v] = 1.0
NO_MARKER_PRIOR = np.full((L, 4), 0.25)
OUT = Path(__file__).resolve().parents[1] / 'results'

# GF(4) with codes 0,1,2(=w),3(=w^2=w+1); addition is XOR of the 2-bit codes
GF_MUL = np.array([[0, 0, 0, 0],
                   [0, 1, 2, 3],
                   [0, 2, 3, 1],
                   [0, 3, 1, 2]], dtype=np.int64)


def parity_columns(k, r):
    """k columns of GF(4)^r, pairwise non-collinear, excluding unit vectors (so [A|I] has d_min=3)."""
    if r == 0:
        return np.zeros((k, 0), dtype=np.int64)
    pts = []
    for v in range(1, 4 ** r):
        vec = [(v >> (2 * t)) & 3 for t in range(r)]
        lead = next(x for x in vec if x)
        if lead != 1:                       # normalise: first nonzero symbol equals 1
            continue
        if sum(1 for x in vec if x) == 1:   # unit vector -> reserved for the parity identity part
            continue
        pts.append(vec)
    assert len(pts) >= k, (k, r, len(pts))
    return np.array(pts[:k], dtype=np.int64)


def digits(i, k):
    return [(i // 4 ** (k - 1 - j)) % 4 for j in range(k)]


def codewords(M, k, A, idx=None):
    """Index codewords for i = 0..M-1 (or for the given indices), shape (len, k+r) uint8."""
    i = np.arange(M, dtype=np.int64) if idx is None else np.asarray(idx, dtype=np.int64)
    dig = np.stack([(i // 4 ** (k - 1 - j)) % 4 for j in range(k)], axis=1)
    r = A.shape[1]
    par = np.zeros((len(i), r), dtype=np.int64)
    for j in range(k):
        par ^= GF_MUL[dig[:, j][:, None], A[j][None, :]]
    return np.concatenate([dig, par], axis=1).astype(np.uint8)


def channel(x, rng):
    out = []
    for s in x:
        m = 0
        while m < 2 and rng.random() < P_I:
            out.append(int(rng.integers(4)))
            m += 1
        if rng.random() < P_D:
            continue
        if rng.random() < P_S:
            out.append(int((s + rng.integers(1, 4)) % 4))
        else:
            out.append(int(s))
    return out


E_TAB = np.full((5, 4), P_S / 3)
for _a in range(4):
    E_TAB[_a, _a] = 1 - P_S
E_TAB[4, :] = 0.0                                   # sentinel (beyond read end)
C0 = (1 - P_I) * P_D
C1D, C1T = P_I * (1 - P_I) * P_D / 4, (1 - P_I) * (1 - P_D)
C2D, C2T = P_I ** 2 * (1 - P_I) * P_D / 16, P_I * (1 - P_I) * (1 - P_D) / 4
C3T = P_I ** 2 * (1 - P_I) * (1 - P_D) / 16


def set_rates(pi, pd, ps):
    """Reset the channel rates used by channel() and the forward-backward (module-level constants)."""
    global P_I, P_D, P_S, C0, C1D, C1T, C2D, C2T, C3T
    P_I, P_D, P_S = pi, pd, ps
    E_TAB[:4, :] = ps / 3
    for a in range(4):
        E_TAB[a, a] = 1 - ps
    C0 = (1 - pi) * pd
    C1D, C1T = pi * (1 - pi) * pd / 4, (1 - pi) * (1 - pd)
    C2D, C2T = pi ** 2 * (1 - pi) * pd / 16, pi * (1 - pi) * (1 - pd) / 4
    C3T = pi ** 2 * (1 - pi) * (1 - pd) / 16


def emissions(Y, lens, t):
    """G[l] of shape (B, NS, 4): prob of emitting l read symbols for input x at position t, all drifts."""
    B = Y.shape[0]
    p = t + np.arange(-D, D + 1)                    # read pointer per drift state
    pc = np.clip(p, 0, None)
    G = []
    for l in range(4):
        valid = (p[None, :] >= 0) & (p[None, :] + l <= lens[:, None])
        if l == 0:
            g = np.broadcast_to(C0, (B, NS, 4)).copy()
        else:
            ylast = Y[:, np.minimum(pc + l - 1, Y.shape[1] - 1)]
            e = E_TAB[ylast]                        # (B, NS, 4)
            if l == 1:
                g = C1D + C1T * e
            elif l == 2:
                g = C2D + C2T * e
            else:
                g = C3T * e
        g = g * valid[:, :, None]
        G.append(g)
    return G


def strand(index_cw, rng, markers=True):
    """Index codeword on the first info positions, uniform data on the rest, known markers."""
    x = rng.integers(4, size=L)
    pos = INFO_POS if markers else list(range(L))
    x[pos[:len(index_cw)]] = index_cw
    if markers:
        for p, v in MARKER_VAL.items():
            x[p] = v
    return [int(s) for s in x]


def index_app(reads, n_idx, markers=True):
    """Forward-backward over the whole strand; APPs of the n_idx index positions, shape (B, n_idx, 4)."""
    prior = PRIOR if markers else NO_MARKER_PRIOR
    idx_pos = (INFO_POS if markers else list(range(L)))[:n_idx]
    slot = {p: s for s, p in enumerate(idx_pos)}
    B = len(reads)
    lens = np.array([len(r) for r in reads])
    Y = np.full((B, lens.max() + 4), 4, dtype=np.int64)
    for b, r in enumerate(reads):
        Y[b, :len(r)] = r
    alpha = np.zeros((B, L + 1, NS))
    alpha[:, 0, D] = 1.0
    for t in range(L):
        G = emissions(Y, lens, t)
        nxt = np.zeros((B, NS))
        a = alpha[:, t]
        for l in range(4):
            w = a * (G[l] * prior[t]).sum(axis=2)
            sh = l - 1                              # d' = d + sh
            if sh < 0:
                nxt[:, :NS + sh] += w[:, -sh:]
            elif sh == 0:
                nxt += w
            else:
                nxt[:, sh:] += w[:, :NS - sh]
        nxt /= nxt.sum(axis=1, keepdims=True)
        alpha[:, t + 1] = nxt
    beta = np.zeros((B, L + 1, NS))
    fin = lens - L + D
    beta[np.arange(B), L, fin] = 1.0
    app = np.zeros((B, n_idx, 4))
    for t in range(L - 1, -1, -1):
        G = emissions(Y, lens, t)
        bn = beta[:, t + 1]
        cur = np.zeros((B, NS))
        acc = np.zeros((B, NS, 4))
        for l in range(4):
            sh = l - 1
            bs = np.zeros((B, NS))                  # bs[:, d] = beta_{t+1}(d + sh)
            if sh < 0:
                bs[:, -sh:] = bn[:, :NS + sh]
            elif sh == 0:
                bs = bn
            else:
                bs[:, :NS - sh] = bn[:, sh:]
            term = G[l] * bs[:, :, None]
            acc += term
        cur = (acc * prior[t]).sum(axis=2)
        cur /= cur.sum(axis=1, keepdims=True)
        beta[:, t] = cur
        if t in slot:
            ap = (alpha[:, t][:, :, None] * acc).sum(axis=1) * prior[t]
            app[:, slot[t]] = ap / ap.sum(axis=1, keepdims=True)
    return app


def dp_logz(app, M, k, A):
    """Exact log sum_{i<M} prod_s P_s(c_s(i)) via the syndrome trellis; cost independent of M."""
    B = app.shape[0]
    r = A.shape[1]
    nsyn = 4 ** r
    pack = [int(sum(int(GF_MUL[a, A[j][t]]) << (2 * t) for t in range(r))) for j in range(k) for a in range(4)]
    pack = np.array(pack).reshape(k, 4)
    bound = digits(M - 1, k)
    v = np.zeros((B, 2, nsyn))                      # flag 0 = tight to M-1, 1 = already smaller
    v[:, 0, 0] = 1.0
    syn = np.arange(nsyn)
    for j in range(k):
        nv = np.zeros_like(v)
        for a in range(4):
            perm = syn ^ pack[j, a]                 # nv[..., s ^ c] gets v[..., s]
            pa = app[:, j, a][:, None]
            nv[:, 1, perm] += v[:, 1, :] * pa
            if a < bound[j]:
                nv[:, 1, perm] += v[:, 0, :] * pa
            elif a == bound[j]:
                nv[:, 0, perm] += v[:, 0, :] * pa
        v = nv
    w = np.ones((B, nsyn))
    for t in range(r):
        st = (syn >> (2 * t)) & 3
        w *= app[:, k + t, st]
    return np.log((v.sum(axis=1) * w).sum(axis=1))


def run(a):
    k = max(1, math.ceil(math.log(a.M, 4) - 1e-12))
    A = parity_columns(k, a.r)
    n_idx = k + a.r
    mk = not a.no_markers
    tag = f'M{a.M}_r{a.r}_s{a.seed}' + ('' if mk else '_nomk')
    od = OUT / tag
    od.mkdir(parents=True, exist_ok=True)
    if a.method == 'trellis':
        tag += '_trellis'
        od = OUT / tag
        od.mkdir(parents=True, exist_ok=True)
        import trellis_topk as tk
    t0 = time.perf_counter()
    C = codewords(a.M, k, A) if a.method == 'brute' else None
    nchunks = a.nreads // a.chunk
    for ci in range(nchunks):
        f = od / f'chunk_{ci:03d}.npz'
        if f.exists():
            continue
        if time.perf_counter() - t0 > a.budget:
            print('budget reached; rerun to continue', flush=True)
            return
        rng = np.random.default_rng([a.seed, a.M, a.r, ci])
        true = rng.integers(a.M, size=a.chunk)
        Ct = codewords(a.M, k, A, true)
        reads, keep = [], []
        for b in range(a.chunk):
            x = strand(Ct[b], rng, mk)
            y = channel(x, rng)
            if abs(len(y) - L) <= D:
                reads.append(y)
                keep.append(b)
        true = true[keep]
        Ct = Ct[keep]
        tb = time.perf_counter()
        app = index_app(reads, n_idx, mk)
        t_bcjr = time.perf_counter() - tb
        logp = np.log(np.maximum(app, 1e-300))
        B = len(reads)
        if a.method == 'trellis':
            tb = time.perf_counter()
            lz = dp_logz(app, a.M, k, A)
            t_dp = (time.perf_counter() - tb) / B
            tb = time.perf_counter()
            pack, nsyn = tk.tables(k, A)
            H, bound = tk.heuristic(logp, a.M, k, A, pack, nsyn)
            st = logp[np.arange(B)[:, None], np.arange(n_idx)[None, :], Ct].sum(axis=1)
            logq_true = st - lz
            rank, logS, hard_ok = np.full(B, 16), np.zeros((B, 3)), np.zeros(B, dtype=bool)
            for b in range(B):
                lst = tk.topk_one(logp[b], H[b], bound, k, pack, 16)
                ids = [i for i, _ in lst]
                vals = np.array([v for _, v in lst])
                if true[b] in ids:
                    rank[b] = ids.index(true[b])
                hard_ok[b] = ids[0] == true[b]
                for ii, kk in enumerate((1, 4, 16)):
                    tkv = vals[:kk]
                    logS[b, ii] = tkv.max() + math.log(np.exp(tkv - tkv.max()).sum()) - lz[b]
            t_top = (time.perf_counter() - tb) / B
            np.savez(f, logq_true=logq_true, rank=rank, logS=logS, hard_ok=hard_ok, t_bcjr=t_bcjr / B,
                     t_brute=np.nan, t_dp=t_dp, t_top16=t_top, dropped=a.chunk - B, dp_err=np.nan)
            print(f'{tag} chunk {ci}: B={B} hard_err={1 - hard_ok.mean():.4f} t_dp={t_dp * 1e3:.4f}ms '
                  f't_top16={t_top * 1e3:.3f}ms', flush=True)
            continue
        logq_true, rank, logS = np.zeros(B), np.zeros(B, dtype=np.int64), np.zeros((B, 3))
        hard = np.zeros(B, dtype=np.int64)
        tb = time.perf_counter()
        for b in range(B):
            s = logp[b, 0][C[:, 0]].astype(np.float64)
            for t in range(1, n_idx):
                s += logp[b, t][C[:, t]]
            mx = s.max()
            lz = mx + math.log(np.exp(s - mx).sum())
            st = s[true[b]]
            logq_true[b] = st - lz
            rank[b] = int((s > st).sum())
            top = np.sort(np.partition(s, -16)[-16:])[::-1] if a.M > 16 else np.sort(s)[::-1]
            for ii, kk in enumerate((1, 4, 16)):
                tk = top[:kk]
                logS[b, ii] = tk.max() + math.log(np.exp(tk - tk.max()).sum()) - lz
            hard[b] = int(np.argmax(s))
        t_brute = (time.perf_counter() - tb) / B
        tb = time.perf_counter()
        lz_dp = dp_logz(app, a.M, k, A)
        t_dp = (time.perf_counter() - tb) / B
        lz_brute = np.array([logp[b][np.arange(n_idx), Ct[b]].sum() for b in range(B)]) - logq_true
        np.savez(f, logq_true=logq_true, rank=rank, logS=logS, hard_ok=(hard == true),
                 t_bcjr=t_bcjr / B, t_brute=t_brute, t_dp=t_dp, dropped=a.chunk - B,
                 dp_err=np.abs(lz_dp - lz_brute).max())
        print(f'{tag} chunk {ci}: B={B} hard_err={np.mean(hard != true):.4f} '
              f't_bcjr={t_bcjr / B * 1e3:.2f}ms t_brute={t_brute * 1e3:.2f}ms t_dp={t_dp * 1e3:.4f}ms '
              f'dp_err={np.abs(lz_dp - lz_brute).max():.2e}', flush=True)
    print('done', tag, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, required=True)
    ap.add_argument('--r', type=int, default=3)
    ap.add_argument('--nreads', type=int, default=2000)
    ap.add_argument('--chunk', type=int, default=250)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--budget', type=float, default=240)
    ap.add_argument('--no_markers', action='store_true')
    ap.add_argument('--method', choices=('brute', 'trellis'), default='brute')
    run(ap.parse_args())
