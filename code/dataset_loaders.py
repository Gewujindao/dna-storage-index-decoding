"""Dataset loaders (data files only, read as text)."""
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DL = HERE.parent / 'downloads'


def load_dnaformer(fname='BinnedPilotNanopore.txt'):
    """Binned format: header line = encoded (design) sequence, then a line of asterisks, then reads, then blank lines."""
    centers, clusters = [], []
    state = 'between'                                  # between -> header -> stars -> reads
    with open(DL / 'dnaformer_pilot_nanopore' / fname, encoding='ascii') as fh:   # streamed line by line
        for line in fh:
            s = line.strip()
            if state == 'between':
                if s:
                    centers.append(s); clusters.append([]); state = 'stars'
            elif state == 'stars':
                assert set(s) == {'*'}, s[:30]
                state = 'reads'
            else:
                if s:
                    clusters[-1].append(s)
                else:
                    state = 'between'
    return centers, clusters


def describe(centers, clusters):
    lens = Counter(len(c) for c in centers)
    sizes = [len(c) for c in clusters]
    rl = Counter()
    for cl in clusters[:2000]:
        for r in cl:
            rl[len(r) // 5 * 5] += 1
    return dict(n_clusters=len(centers), center_len=lens.most_common(5), reads=sum(sizes),
                empty=sum(1 for s in sizes if s == 0), median_size=sorted(sizes)[len(sizes) // 2],
                read_len_hist=sorted(rl.items())[:30])
