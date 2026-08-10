#!/usr/bin/env python3
"""Controls for the cap-relation PMI measurement.

`cap_relation_pmi.py` found strong mutual information between the caps of
nearby positions, decaying sharply from delta=1 to delta=3. That decay is
suspicious: L0 caps are discovered over a window of 3 tokens, so positions
1 apart share 2 of their 3 input tokens and positions 2 apart share 1.
Neighbouring caps are correlated BY CONSTRUCTION, before any learning.

If that overlap explains the signal, the relation table encodes nothing a
window-3 cap layer does not already feed forward through the residual
stream — and there would be no reason for attention to recover it.

Two controls, both computed from the token stream alone:

  tokens    raw token identity at the same distances. The information
            actually present in the data, and already available to the
            model through its embeddings.

  trigram   an UNTRAINED window-3 quantizer: hash (t[i-2], t[i-1], t[i])
            into the same number of buckets as the real cap layer. Same
            window, same bucket count, zero learning. Whatever MI decay
            this produces is pure overlap artefact.

Real caps must beat `trigram` by a clear margin to be carrying anything
of their own.

Usage:
    python3 scripts/cap_relation_control.py TOKENS.bin N_CAPS [--seq-len 128]
"""

import sys
import numpy as np

from cap_relation_pmi import DELTAS, N_SHUFFLE, RNG, joint, mutual_information


def sequences(tokens, seq_len):
    n = (tokens.size // seq_len) * seq_len
    return tokens[:n].reshape(-1, seq_len).astype(np.int64)


def trigram_buckets(tokens, k, window=3):
    """Untrained window-`window` quantizer over the token stream.

    Deterministic mixing of the window's token ids into `k` buckets. This
    has exactly the cap layer's input footprint and none of its learning,
    so it isolates the overlap artefact.
    """
    n = tokens.size
    acc = np.zeros(n, dtype=np.int64)
    for w in range(window):
        shifted = np.concatenate([np.zeros(w, dtype=np.int64), tokens[: n - w]])
        # Distinct odd multipliers per offset so position within the
        # window matters, as it does for a real cap key.
        acc = acc * 1_000_003 + shifted * (2 * w + 1)
        acc &= 0x7FFF_FFFF
    return (acc * 2_654_435_761 & 0x7FFF_FFFF) % k


def profile(w, k, label):
    print(f"\n  ── {label} (k={k}) ──")
    print(f"  {'delta':>6} {'MI':>7} {'shuffled':>9} {'excess':>8} {'% of H':>8}")
    seq_len = w.shape[1]
    out = {}
    for d in DELTAS:
        if d >= seq_len:
            continue
        q = w[:, d:].ravel()
        o = w[:, :-d].ravel()
        c = joint(q, o, k)
        mi = mutual_information(c)
        null = np.mean([mutual_information(joint(RNG.permutation(q), o, k))
                        for _ in range(N_SHUFFLE)])
        excess = mi - null
        h_offer = mutual_information(joint(o, o, k))
        pct = 100.0 * excess / h_offer if h_offer > 0 else 0.0
        out[d] = pct
        print(f"  {d:>6} {mi:>7.3f} {null:>9.3f} {excess:>8.3f} {pct:>7.1f}%")
    return out


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    path, k = sys.argv[1], int(sys.argv[2])
    seq_len = 128
    if "--seq-len" in sys.argv:
        seq_len = int(sys.argv[sys.argv.index("--seq-len") + 1])

    tokens = np.fromfile(path, dtype=np.uint32).astype(np.int64)
    vocab = int(tokens.max()) + 1
    print(f"\n  tokens      {path}  ({tokens.size} tokens, vocab {vocab})")
    print("  Percentages are excess MI as a fraction of the offering side's")
    print("  own entropy — comparable across different bucket counts.")

    tok = profile(sequences(tokens, seq_len), vocab, "CONTROL: raw tokens")
    tri = profile(sequences(trigram_buckets(tokens, k), seq_len), k,
                  "CONTROL: untrained window-3 quantizer")

    print("\n  ── what to compare against ──")
    print("  Real cap numbers come from cap_relation_pmi.py on the same corpus.")
    print(f"  {'delta':>6} {'tokens':>9} {'trigram':>9}")
    for d in sorted(tri):
        print(f"  {d:>6} {tok.get(d, float('nan')):>8.1f}% {tri[d]:>8.1f}%")
    print("\n  If the real caps track `trigram`, their structure is window")
    print("  overlap and the relation table carries nothing new. They must")
    print("  clear it by a wide margin to justify the design.")


if __name__ == "__main__":
    main()
