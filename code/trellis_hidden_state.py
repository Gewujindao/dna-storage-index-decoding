"""The hidden-state joint index posterior on the product trellis
(drift x channel state x partial syndrome x 'below M-1' flag) for a systematic GF(4) index code;
cost independent of M. Checked against enumeration at small M; timed at M = 10^4, 10^5, 10^6.

  python -X utf8 trellis_hidden_state.py
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trellis_index_sim as ab  # noqa: E402
import trellis_topk as tk  # noqa: E402
import channel_model as ch  # noqa: E402
import evaluate_dataset1 as ev  # noqa: E402

NS, S_ = ch.NS, None


def trellis_logq(model, Y1, len1, beta_n, M, k, A, true_cw):
    """log p(true | y) with Z summed over all i < M on the product trellis (index at positions 0..n-1)."""
    S = model.S
    r = A.shape[1]
    nsyn = 4 ** r
    pack, _ = tk.tables(k, A)
    bound = ab.digits(M - 1, k)
    syn = np.arange(nsyn)
    W = NS * S
    al = np.zeros((2, nsyn, W))
    al[0, 0, ch.D * S:(ch.D + 1) * S] = model.pi0
    at = np.zeros(W)
    at[ch.D * S:(ch.D + 1) * S] = model.pi0
    for t in range(k + r):
        T = ch.step_matrices(model, Y1, len1, t)
        new = np.zeros_like(al)
        if t < k:
            for a in range(4):
                perm = syn ^ pack[t, a]
                new[1, perm] += al[1] @ T[a]
                if a < bound[t]:
                    new[1, perm] += al[0] @ T[a]
                elif a == bound[t]:
                    new[0, perm] += al[0] @ T[a]
        else:
            sym = (syn >> (2 * (t - k))) & 3
            for a in range(4):
                rows = np.nonzero(sym == a)[0]
                new[:, rows] = al[:, rows] @ T[a]
        sc = new.max()
        al = new / sc
        at = (at @ T[int(true_cw[t])]) / sc
    b = beta_n.reshape(-1)
    Z = (al.sum((0, 1)) @ b)
    return math.log(max(at @ b, 1e-300)) - math.log(Z)


def main():
    mods, sims = ev.models()
    model, chan = mods['hmm2x'], sims['fit2x']
    rng = np.random.default_rng(11)
    # 1) check against enumeration at small M
    for M in (200, 1000):
        k = math.ceil(math.log(M, 4) - 1e-12)
        A = ab.parity_columns(k, 3)
        C = ab.codewords(M, k, A).astype(np.int64)
        worst = 0.0
        for rep in range(5):
            i = int(rng.integers(M))
            x = list(C[i]) + list(rng.integers(4, size=ch.L - C.shape[1]))
            y = ch.simulate(x, chan, rng)
            if abs(len(y) - ch.L) > ch.D:
                continue
            Y, lens = ch.pad([y])
            beta = ch.backward(model, Y, lens, t_stop=C.shape[1])[0, C.shape[1]]
            lq_t = trellis_logq(model, Y, lens, beta, M, k, A, C[i])
            lq_e = ch.joint_logpost(model, y, C)[i]
            worst = max(worst, abs(lq_t - lq_e))
        print(f'check M={M}: max |log q_trellis - log q_enum| = {worst:.2e}', flush=True)
    # 2) timing at three pool sizes (forward over the index region only; backward is shared and M-free)
    for M in (10 ** 4, 10 ** 5, 10 ** 6):
        k = math.ceil(math.log(M, 4) - 1e-12)
        A = ab.parity_columns(k, 3)
        ts, lqs = [], []
        for rep in range(40):
            i = int(rng.integers(M))
            cw = ab.codewords(M, k, A, [i])[0].astype(np.int64)
            x = list(cw) + list(rng.integers(4, size=ch.L - len(cw)))
            y = ch.simulate(x, chan, rng)
            if abs(len(y) - ch.L) > ch.D:
                continue
            Y, lens = ch.pad([y])
            beta = ch.backward(model, Y, lens, t_stop=len(cw))[0, len(cw)]
            t0 = time.perf_counter()
            lqs.append(trellis_logq(model, Y, lens, beta, M, k, A, cw))
            ts.append(time.perf_counter() - t0)
        print(f'M={M}: n_idx={k + 3} forward on product trellis {np.mean(ts) * 1e3:.1f} ms/read '
              f'(state count {ch.NS * model.S * 2 * 64}); mean -log2 q(true) = {-np.mean(lqs) / math.log(2):.3f}', flush=True)
    # same timing for the 1-state model (cost vs number of channel states)
    m1 = mods['iidx']
    M = 10 ** 6
    k = math.ceil(math.log(M, 4) - 1e-12); A = ab.parity_columns(k, 3)
    ts = []
    for rep in range(40):
        i = int(rng.integers(M)); cw = ab.codewords(M, k, A, [i])[0].astype(np.int64)
        y = ch.simulate(list(cw) + list(rng.integers(4, size=ch.L - len(cw))), chan, rng)
        if abs(len(y) - ch.L) > ch.D:
            continue
        Y, lens = ch.pad([y]); beta = ch.backward(m1, Y, lens, t_stop=len(cw))[0, len(cw)]
        t0 = time.perf_counter(); trellis_logq(m1, Y, lens, beta, M, k, A, cw); ts.append(time.perf_counter() - t0)
    print(f'M=10^6, 1-state model: {np.mean(ts) * 1e3:.1f} ms/read', flush=True)
    print('done trellis', flush=True)


if __name__ == '__main__':
    main()
