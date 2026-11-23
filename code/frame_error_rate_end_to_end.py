"""The frame-error conversion of frame_error_rate re-run with the end-to-end parameters (prediction only).
Same model as frame_error_rate (ideal outer code, perfect inner code, Poisson coverage, dataset-1 held-out statistics),
but block length 250 strands and per-block FER target 1 - 0.9**(1/8) (file of 8 blocks recovered with prob 0.9).
  python -X utf8 frame_error_rate_end_to_end.py
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frame_error_rate as fer  # noqa: E402

fer.N_O = 250
fer.TARGET = 1 - 0.9 ** (1 / 8)
fer.LF = np.array([math.lgamma(i + 1) for i in range(fer.N_O + 2)])
EV = HERE.parent / 'results' / 'dataset1' / 'eval'


def main():
    thetas = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    rows = {'baseline': fer.load(EV / 'real_heldout__marg-iid', 0.6), 'design': fer.load(EV / 'real_heldout__joint-hmm2x', 1.0)}
    cs = np.arange(1.0, 10.01, 0.5)
    table = {nm: [fer.best_rate(*rows[nm], c, thetas) for c in cs] for nm in rows}
    for i, c in enumerate(cs):
        print(f'c={c}: ' + ' | '.join(f'{nm} R*={table[nm][i][0]:.3f} (theta {table[nm][i][1]})' for nm in rows), flush=True)
    for R in (0.6, 0.7, 0.8, 0.9):
        print(f'outer rate {R}: ' + ' | '.join(
            f'{nm} c_min={next((float(c) for c, (r, _) in zip(cs, table[nm]) if r >= R), None)}' for nm in rows), flush=True)
    print('done predict', flush=True)


if __name__ == '__main__':
    main()
