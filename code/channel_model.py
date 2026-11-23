"""Channel models with a hidden channel state and the index posteriors built on them.

Model (per input position t): hidden state z_t (Markov chain pi0, A); given z_t an event
(m insertions before the symbol, m <= 2; kind M/S/D) with probabilities E[z, 3*m + kind];
inserted bases uniform; substitutions x -> y with matrix SM (rows sum to 1, zero diagonal).
Optional per-position and homopolymer multipliers on the error part (simulation only).
S = 1 recovers the i.i.d. Davey-MacKay channel.

Posteriors over an index set C (Mc x NC, positions 0..NC-1 of the strand, no markers):
  'marg'  : forward-backward with uniform priors -> symbol APPs -> product over index positions
  'joint' : exact p(i|y) ∝ P(y | strand starts with C[i]) (data uniform), by enumeration over C here
            (the product-trellis version for linear index codes is in trellis_hidden_state.py)
"""
import math

import numpy as np

L = 110
D = 10
NS = 2 * D + 1
KIND = {'M': 0, 'S': 1, 'D': 2}


class Model:
    def __init__(self, pi0, A, E, SM=None, pos_scale=None, homo_scale=None, ins_same=None):
        self.pi0, self.A, self.E = np.asarray(pi0, float), np.asarray(A, float), np.asarray(E, float)
        self.S = len(self.pi0)
        self.SM = np.full((4, 4), 1 / 3) - np.eye(4) / 3 if SM is None else np.asarray(SM, float)
        self.pos_scale, self.homo_scale = pos_scale, homo_scale
        # inserted-base distribution given the strand base x at this position (uniform if None)
        self.Q = np.full((4, 4), 0.25) if ins_same is None else \
            np.full((4, 4), (1 - ins_same) / 3) + np.eye(4) * (ins_same - (1 - ins_same) / 3)

    @staticmethod
    def iid(pi, pd, ps, SM=None):
        E = np.zeros(9)
        for m in range(3):
            pm = pi ** m * (1 - pi) if m < 2 else pi ** 2
            E[3 * m + 0] = pm * (1 - pd) * (1 - ps)
            E[3 * m + 1] = pm * (1 - pd) * ps
            E[3 * m + 2] = pm * pd
        return Model([1.0], [[1.0]], E[None, :], SM)


def _event_dist(model, z, t, x):
    e = model.E[z].copy()
    f = 1.0
    if model.pos_scale is not None:
        f *= model.pos_scale[t]
    if model.homo_scale is not None:
        hom = (t > 0 and x[t] == x[t - 1]) or (t + 1 < len(x) and x[t] == x[t + 1])
        f *= model.homo_scale[1] if hom else model.homo_scale[0]
    if f != 1.0:
        err = e.copy(); err[0] = 0
        e = err * f
        e[0] = max(1 - e.sum(), 1e-6)
        e /= e.sum()
    return e


def simulate(x, model, rng):
    z = rng.choice(model.S, p=model.pi0)
    out = []
    for t, s in enumerate(x):
        e = _event_dist(model, z, t, x)
        v = rng.choice(9, p=e)
        m, kind = divmod(v, 3)
        out.extend(int(rng.choice(4, p=model.Q[s])) for _ in range(m))
        if kind == 0:
            out.append(int(s))
        elif kind == 1:
            out.append(int(rng.choice(4, p=model.SM[s])))
        z = rng.choice(model.S, p=model.A[z])
    return out


def emissions(model, Y, lens, t):
    """G[z, l] of shape (S, 4, B, NS, 4): prob. that input x at position t in state z emits l read symbols."""
    B = Y.shape[0]
    p = t + np.arange(-D, D + 1)
    pc = np.clip(p, 0, None)
    S = model.S
    G = np.zeros((S, 4, B, NS, 4))
    yk = [Y[:, np.minimum(pc + k, Y.shape[1] - 1)] for k in range(3)]                  # (B, NS); 4 = beyond
    qk = [model.Q[:, np.minimum(y, 3)].transpose(1, 2, 0) * (y < 4)[:, :, None] for y in yk]   # Q[x, y_k] -> (B, NS, x)
    qprod = [np.ones((B, NS, 4)), qk[0], qk[0] * qk[1]]                                  # m inserted symbols
    for l in range(4):
        valid = ((p[None, :] >= 0) & (p[None, :] + l <= lens[:, None]))[:, :, None]
        if l > 0:
            yl = yk[l - 1]
            ylc = np.minimum(yl, 3)
            match = (yl[:, :, None] == np.arange(4)[None, None, :]).astype(float)
            sub = model.SM[:, ylc].transpose(1, 2, 0) * (yl < 4)[:, :, None]   # SM[x, y] -> (B, NS, x)
        for z in range(S):
            E = model.E[z]
            if l == 0:
                g = np.full((B, NS, 4), E[2])
            else:
                m = l - 1
                g = np.zeros((B, NS, 4))
                if l <= 2:
                    g += E[3 * l + 2] * qprod[l]                      # l insertions, symbol deleted
                g += qprod[m] * (E[3 * m + 0] * match + E[3 * m + 1] * sub)
            G[z, l] = g * valid
    return G


def _shift(dst, src, sh):
    if sh < 0:
        dst[:, :NS + sh] += src[:, -sh:]
    elif sh == 0:
        dst += src
    else:
        dst[:, sh:] += src[:, :NS - sh]


def pad(reads):
    lens = np.array([len(r) for r in reads])
    Y = np.full((len(reads), lens.max() + 4), 4, dtype=np.int64)
    for b, r in enumerate(reads):
        Y[b, :len(r)] = r
    return Y, lens


def backward(model, Y, lens, t_stop=0, prior=None):
    """beta_t(d, z) for t = L..t_stop (normalised per step); data prior uniform unless given (L, 4)."""
    B, S = Y.shape[0], model.S
    prior = np.full((L, 4), 0.25) if prior is None else prior
    beta = np.zeros((B, L + 1, NS, S))
    beta[np.arange(B), L, lens - L + D, :] = 1.0
    for t in range(L - 1, t_stop - 1, -1):
        G = emissions(model, Y, lens, t)
        bz = beta[:, t + 1] @ model.A.T                      # sum_z' A[z, z'] beta_{t+1}(d, z')  -> (B, NS, S)
        cur = np.zeros((B, NS, S))
        for z in range(S):
            for l in range(4):
                bs = np.zeros((B, NS))
                _shift_back(bs, bz[:, :, z], l - 1)
                cur[:, :, z] += (G[z, l] * prior[t]).sum(2) * bs
        beta[:, t] = cur / cur.sum((1, 2), keepdims=True)
    return beta


def _shift_back(dst, src, sh):
    """dst[:, d] = src[:, d + sh]."""
    if sh < 0:
        dst[:, -sh:] = src[:, :NS + sh]
    elif sh == 0:
        dst[:] = src
    else:
        dst[:, :NS - sh] = src[:, sh:]


def marginal_logapp(model, reads, nc, offset=0):
    """Symbol APPs (log) at positions offset..offset+nc-1 with uniform priors everywhere."""
    Y, lens = pad(reads)
    B, S = len(reads), model.S
    beta = backward(model, Y, lens)
    al = np.zeros((B, NS, S))
    al[:, D, :] = model.pi0
    out = np.zeros((B, offset + nc, 4))
    for t in range(offset + nc):
        G = emissions(model, Y, lens, t)
        bz = beta[:, t + 1] @ model.A.T
        app = np.zeros((B, 4))
        nal = np.zeros((B, NS, S))
        for z in range(S):
            for l in range(4):
                bs = np.zeros((B, NS))
                _shift_back(bs, bz[:, :, z], l - 1)
                app += (al[:, :, z][:, :, None] * G[z, l] * bs[:, :, None]).sum(1)
                w = al[:, :, z] * G[z, l].mean(2)
                tmp = np.zeros((B, NS))
                _shift(tmp, w, l - 1)
                nal += tmp[:, :, None] * model.A[z][None, None, :]
        out[:, t] = np.log(np.maximum(app / app.sum(1, keepdims=True), 1e-300))
        al = nal / nal.sum((1, 2), keepdims=True)
    return out[:, offset:]


class Trie:
    """Prefix tree of an index set C (Mc, nc): per depth, parent node and symbol of every node."""

    def __init__(self, C):
        self.nc = C.shape[1]
        self.parent, self.sym = [], []
        prev = {(): 0}
        for t in range(self.nc):
            cur = {}
            par, sy = [], []
            for row in C:
                key = tuple(row[:t + 1])
                if key not in cur:
                    cur[key] = len(cur)
                    par.append(prev[key[:-1]]); sy.append(int(key[-1]))
            self.parent.append(np.array(par)); self.sym.append(np.array(sy))
            prev = cur
        self.leaf = np.array([prev[tuple(row)] for row in C])
        self.nodes = sum(len(p) for p in self.parent)


def step_matrices(model, Y1, len1, t):
    """T[a] (NS*S x NS*S): one input symbol a at position t, state z -> z', drift d -> d + l - 1."""
    S = model.S
    G = emissions(model, Y1, len1, t)[:, :, 0]            # (S, 4, NS, 4)
    T = np.zeros((4, NS, S, NS, S))
    d = np.arange(NS)
    for l in range(4):
        d2 = d + l - 1
        ok = (d2 >= 0) & (d2 < NS)
        val = G[:, l][:, d[ok], :].transpose(1, 2, 0)[:, :, :, None] * model.A[None, None, :, :]   # (n_ok, a, z, z')
        T[:, d[ok], :, d2[ok], :] = val
    return T.reshape(4, NS * S, NS * S)


def joint_logpost_trie(model, Y1, len1, beta_nc, trie, offset=0):
    """Exact log p(i | y) over the index set, forward along the prefix tree with small matrix products.
    The index occupies positions offset..offset+nc-1; positions before it have a uniform prior.
    beta_nc: (NS, S) backward vector at position offset+nc for this read."""
    S = model.S
    alpha = np.zeros((1, NS * S))
    alpha[0, D * S:(D + 1) * S] = model.pi0
    for t in range(offset):
        alpha = alpha @ step_matrices(model, Y1, len1, t).mean(axis=0)
        alpha = alpha / alpha.max()
    for t in range(trie.nc):
        T = step_matrices(model, Y1, len1, offset + t)
        par, sy = trie.parent[t], trie.sym[t]
        new = np.zeros((len(par), NS * S))
        for a in range(4):
            rows = np.nonzero(sy == a)[0]
            if len(rows):
                new[rows] = alpha[par[rows]] @ T[a]
        alpha = new / new.max()
    lik = alpha[trie.leaf] @ beta_nc.reshape(-1)
    lik = np.maximum(lik, 1e-300)
    return np.log(lik) - math.log(lik.sum())


def joint_logpost(model, read, C):
    """Exact log p(i | y) over the candidate set C (Mc, nc) for one read (enumeration over C)."""
    Y, lens = pad([read])
    nc = C.shape[1]
    S = model.S
    beta = backward(model, Y, lens, t_stop=nc)[0, nc]       # (NS, S)
    Mc = C.shape[0]
    al = np.zeros((Mc, NS, S))
    al[:, D, :] = model.pi0
    for t in range(nc):
        G = emissions(model, Y, lens, t)[:, :, 0]           # (S, 4, NS, 4)
        xs = C[:, t]
        nal = np.zeros((Mc, NS, S))
        for z in range(S):
            for l in range(4):
                w = al[:, :, z] * G[z, l][:, xs].T             # (Mc, NS)
                tmp = np.zeros((Mc, NS))
                _shift(tmp, w, l - 1)
                nal += tmp[:, :, None] * model.A[z][None, None, :]
        al = nal / nal.sum()
    lik = (al * beta[None]).sum((1, 2))
    lik = np.maximum(lik, 1e-300)
    return np.log(lik) - math.log(lik.sum())
