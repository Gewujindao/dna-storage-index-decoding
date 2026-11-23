"""Reed-Solomon over GF(256) (primitive polynomial 0x11d, roots alpha^1..alpha^(n-k)),
shortened to length n <= 255. Systematic encoding; errors-and-erasures bounded-distance decoding
(Berlekamp-Massey initialised with the erasure locator, Chien search, Forney). Own implementation.
Vector convention: c[i] is the coefficient of x^i; parity in positions 0..n-k-1.
  python -X utf8 reed_solomon.py      (self-test on random patterns)
"""
import numpy as np

EXP = np.zeros(512, dtype=np.int64)
LOG = np.zeros(256, dtype=np.int64)
_x = 1
for _i in range(255):
    EXP[_i] = _x
    LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11d
EXP[255:510] = EXP[:255]


def mul(a, b):
    return 0 if a == 0 or b == 0 else int(EXP[LOG[a] + LOG[b]])


def inv(a):
    return int(EXP[255 - LOG[a]])


def poly_mul(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                if b:
                    r[i + j] ^= mul(a, b)
    return r


def poly_eval(p, x):
    """p[i] coefficient of x^i."""
    y = 0
    for a in reversed(p):
        y = mul(y, x) ^ a
    return y


def generator(nsym):
    g = [1]
    for i in range(1, nsym + 1):
        g = poly_mul(g, [EXP[i], 1])           # (x + alpha^i)
    return g


def encode(msg, nsym):
    """msg: k symbols -> codeword of length k + nsym, parity first."""
    g = generator(nsym)
    shifted = [0] * nsym + list(msg)            # m(x) x^nsym
    rem = list(shifted)
    for i in range(len(rem) - 1, nsym - 1, -1):  # long division by monic g (degree nsym)
        coef = rem[i]
        if coef:
            for j in range(nsym + 1):
                rem[i - nsym + j] ^= mul(coef, g[j])
    return [rem[i] for i in range(nsym)] + list(msg)


def syndromes(r, nsym):
    return [poly_eval(r, int(EXP[j])) for j in range(1, nsym + 1)]


def decode(r, nsym, erasures=()):
    """Returns the corrected word or None (failure). Bounded distance: succeeds iff 2e + x <= nsym."""
    n = len(r)
    r = list(r)
    for p in erasures:
        r[p] = 0
    S = syndromes(r, nsym)
    if not any(S) and not erasures:
        return r
    rho = len(erasures)
    if rho > nsym:
        return None
    gam = [1]
    for p in erasures:
        gam = poly_mul(gam, [1, int(EXP[p])])  # (1 + X_p x)
    lam, B, L = list(gam), list(gam), rho
    for k in range(rho + 1, nsym + 1):          # S_k with k = 1..nsym, S[k-1]
        delta = 0
        for j in range(min(L, len(lam) - 1) + 1):
            if lam[j] and k - j >= 1:
                delta ^= mul(lam[j], S[k - j - 1])
        xB = [0] + B
        if delta == 0:
            B = xB
            continue
        T = lam + [0] * max(0, len(xB) - len(lam))
        for j, b in enumerate(xB):
            T[j] ^= mul(delta, b)
        if 2 * L <= k - 1 + rho:
            dinv = inv(delta)
            B = [mul(dinv, a) for a in lam]
            L = k + rho - L
        else:
            B = xB
        lam = T
    while len(lam) > 1 and lam[-1] == 0:
        lam.pop()
    deg = len(lam) - 1
    if deg != L or 2 * (L - rho) + rho > nsym:
        return None
    locs = [i for i in range(n) if poly_eval(lam, int(EXP[(255 - i) % 255])) == 0]
    if len(locs) != deg:
        return None
    Sx = S                                       # S(x) = sum_j S_{j+1} x^j
    omega = poly_mul(Sx, lam)[:nsym]
    dlam = [lam[j] if j % 2 == 1 else 0 for j in range(1, len(lam))]   # formal derivative (char 2)
    for i in locs:
        xinv = int(EXP[(255 - i) % 255])
        den = poly_eval(dlam, xinv)
        if den == 0:
            return None
        r[i] ^= mul(poly_eval(omega, xinv), inv(den))
    if any(syndromes(r, nsym)):
        return None
    return r


def selftest(trials=300, n=250, seed=1):
    rng = np.random.default_rng(seed)
    agree = 0
    for t in range(trials):
        nsym = int(rng.choice([25, 50, 75, 100]))
        c = encode([int(v) for v in rng.integers(256, size=n - nsym)], nsym)
        assert not any(syndromes(c, nsym))
        x = int(rng.integers(0, nsym + 3))
        e = int(rng.integers(0, nsym // 2 + 3))
        pos = rng.choice(n, size=min(n, x + e), replace=False)
        er, ep = list(pos[:x]), list(pos[x:])
        r = list(c)
        for p in er:
            r[p] = int(rng.integers(256))                 # erased, value arbitrary (may be correct)
        for p in ep:
            r[p] ^= int(rng.integers(1, 256))
        out = decode(r, nsym, er)
        ok = out == c
        agree += ok == (2 * len(ep) + len(er) <= nsym)
    return agree, trials


if __name__ == '__main__':
    a, t = selftest()
    print(f'RS self-test: decoder success matches 2e+x<=n-k in {a}/{t} random patterns')
