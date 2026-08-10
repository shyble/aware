#!/usr/bin/env python3
"""Does cap identity carry routing information?

The proposed cap-relation attention replaces the dot-product score with a
table indexed by (querying cap, offering cap, distance):

    w_ij = A[ c_i , c_j , Delta ]      Delta = i - j > 0

That design is only worth building if the table has structure. This script
measures whether it does, from cap winners already dumped by an eval pass —
no training, no forward passes.

Three questions, in the order that can kill the design:

1. DOES CAP IDENTITY ROUTE AT ALL?
   I(c_i ; c_j | Delta), in bits. If knowing the querying cap tells you
   nothing about the offering cap beyond their marginals, the table is
   noise and the design is dead.

   Finite-sample MI is biased UPWARD — with K=128 and ~130k pairs per
   Delta there are 8 samples per cell, so a table of pure noise still
   scores well above zero. Every number here is therefore reported
   against a shuffle control that destroys the pairing while preserving
   both marginals exactly. Only the EXCESS over that floor is evidence.

2. DOES DISTANCE EARN ITS PLACE?
   cap-pair (already benchmarked, reached parity and never beat it) used
   A[c_i, c_j] with no Delta. If the per-distance tables are all the same
   table, Delta adds nothing and the new design predicts parity too —
   which would be the cheapest possible negative result. Measured as the
   correlation between PMI tables at different distances.

3. IS IT PEAKED ENOUGH TO ACT?
   A table that is structured but nearly flat vanishes under softmax.
   Measured as the entropy of P(c_j | c_i, Delta) against uniform.

Usage:
    python3 scripts/cap_relation_pmi.py WINNERS.bin N_CAPS [--seq-len 128]
"""

import sys
import numpy as np

# Distances probed. Dense at short range where syntax lives, sparse
# further out where counts thin and only coarse structure is credible.
DELTAS = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64]
N_SHUFFLE = 3
RNG = np.random.default_rng(0)


def load(path, seq_len):
    w = np.fromfile(path, dtype=np.uint32)
    if w.size % seq_len:
        raise SystemExit(f"{w.size} winners is not a multiple of seq_len={seq_len}")
    return w.reshape(-1, seq_len).astype(np.int64)


def joint(q, o, k):
    """Joint counts as a (k, k) matrix: rows query cap, cols offering cap."""
    return np.bincount(q * k + o, minlength=k * k).reshape(k, k).astype(np.float64)


def shuffle_within_rows(a, rng):
    """Permute each row independently, preserving its multiset exactly.

    The corpus-wide shuffle destroys same-document co-membership along
    with everything else, so a cap distribution that is merely
    story-specific scores as a distance-independent relation. Permuting
    inside each sequence keeps each story's cap composition intact and
    removes only the positional pairing — the null that separates "these
    caps relate at distance d" from "these caps appear in the same
    story".
    """
    idx = np.argsort(rng.random(a.shape), axis=1)
    return np.take_along_axis(a, idx, axis=1)


def mutual_information(c):
    """I(row ; col) in bits from a joint count matrix."""
    n = c.sum()
    if n == 0:
        return 0.0
    p = c / n
    pr = p.sum(1, keepdims=True)
    pc = p.sum(0, keepdims=True)
    denom = pr * pc
    nz = (p > 0) & (denom > 0)
    return float(np.sum(p[nz] * np.log2(p[nz] / denom[nz])))


def pmi_table(c, floor=5):
    """PMI in bits, with cells below `floor` counts zeroed.

    Undersampled cells produce enormous PMI values off a single
    observation; leaving them in would let noise dominate the
    across-distance correlation in question 2.
    """
    n = c.sum()
    if n == 0:
        return np.zeros_like(c)
    p = c / n
    denom = p.sum(1, keepdims=True) * p.sum(0, keepdims=True)
    out = np.zeros_like(p)
    ok = (c >= floor) & (denom > 0)
    out[ok] = np.log2(p[ok] / denom[ok])
    return out


def conditional_row_entropy(c):
    """Mean H(c_j | c_i) in bits, weighted by how often each query cap fires."""
    rows = c.sum(1)
    live = rows > 0
    if not live.any():
        return 0.0
    p = c[live] / rows[live, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.sum(np.where(p > 0, p * np.log2(p, where=p > 0), 0.0), axis=1)
    return float(np.average(h, weights=rows[live]))


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    path, k = sys.argv[1], int(sys.argv[2])
    seq_len = 128
    if "--seq-len" in sys.argv:
        seq_len = int(sys.argv[sys.argv.index("--seq-len") + 1])

    w = load(path, seq_len)
    n_seq = w.shape[0]
    fired = len(np.unique(w))

    print(f"\n  source      {path}")
    print(f"  sequences   {n_seq} x {seq_len} = {w.size} positions")
    print(f"  caps        {fired}/{k} ever fire")
    print(f"  uniform H   {np.log2(k):.2f} bits over {k} caps\n")

    print("  Q1/Q3 — does cap identity route, and is it peaked?")
    print("  `global` shuffles across the corpus; `in-seq` shuffles inside each")
    print("  sequence, so it also absorbs same-story co-membership. The in-seq")
    print("  excess is the honest one — a genuine positional relation.")
    print(f"  {'delta':>6} {'pairs':>9} {'MI':>7} {'global':>8} {'in-seq':>8} "
          f"{'excess':>8} {'% of H':>8} {'H(c_j|c_i)':>11}")

    tables, excesses = {}, {}
    for d in DELTAS:
        if d >= seq_len:
            continue
        q = w[:, d:].ravel()       # querying position, later in the sequence
        o = w[:, :-d].ravel()      # offering position, d tokens earlier
        c = joint(q, o, k)
        mi = mutual_information(c)

        # Null 1: same marginals, no pairing at all. The finite-sample
        # floor a table of pure noise would reach.
        null_global = np.mean([mutual_information(joint(RNG.permutation(q), o, k))
                               for _ in range(N_SHUFFLE)])

        # Null 2: pairing destroyed, story composition preserved. What
        # remains above this is distance-specific relation, not topic.
        null_seq = np.mean([
            mutual_information(joint(shuffle_within_rows(w, RNG)[:, d:].ravel(), o, k))
            for _ in range(N_SHUFFLE)
        ])

        excess = mi - null_seq
        h_cond = conditional_row_entropy(c)
        # Ceiling for the excess: the entropy of the offering cap itself.
        h_offer = mutual_information(joint(o, o, k))
        pct = 100.0 * excess / h_offer if h_offer > 0 else 0.0

        tables[d] = pmi_table(c)
        excesses[d] = excess
        print(f"  {d:>6} {q.size:>9} {mi:>7.3f} {null_global:>8.3f} "
              f"{null_seq:>8.3f} {excess:>8.3f} {pct:>7.1f}% {h_cond:>11.2f}")

    print("\n  Q2 — does distance earn its place? (PMI table correlation)")
    ds = sorted(tables)
    ref = ds[0]
    print(f"  {'delta':>6} {'corr vs d=' + str(ref):>14}")
    for d in ds[1:]:
        a, b = tables[ref].ravel(), tables[d].ravel()
        both = (a != 0) & (b != 0)
        r = np.corrcoef(a[both], b[both])[0, 1] if both.sum() > 10 else float("nan")
        print(f"  {d:>6} {r:>14.3f}")

    # ── Verdict ──────────────────────────────────────────────────────
    # Only distances whose table actually carries signal may be reasoned
    # about. Below this the table is dominated by sampling noise, and
    # noise correlates with nothing — which would read as "distance
    # matters" no matter what the model learned.
    SIGNAL = 0.10  # bits of in-sequence excess
    strong = [d for d in ds if excesses[d] >= SIGNAL]
    weak = [d for d in ds if excesses[d] < SIGNAL]

    print("\n  ── reading it ──")
    if not strong:
        print("  Q1 FAIL   no distance clears the noise floor. Cap identity adds")
        print("            nothing over the marginals; A[c_i,c_j,Delta] is noise.")
        return

    print(f"  Q1 PASS   signal at delta {strong} "
          f"(peak {max(excesses[d] for d in strong):.2f} bits).")
    if weak:
        print(f"            Near-floor at delta {weak}: real but small — compare")
        print("            against the token baseline before crediting it.")

    corrs = []
    for i, d1 in enumerate(strong):
        for d2 in strong[i + 1:]:
            a, b = tables[d1].ravel(), tables[d2].ravel()
            both = (a != 0) & (b != 0)
            if both.sum() > 10:
                corrs.append((d1, d2, np.corrcoef(a[both], b[both])[0, 1]))

    if not corrs:
        print("  Q2 N/A    only one distance carries signal, so distance-specificity")
        print("            cannot be assessed from this dump.")
        return

    m_corr = float(np.mean([c for _, _, c in corrs]))
    detail = ", ".join(f"d{a}~d{b}={c:.2f}" for a, b, c in corrs)
    if m_corr > 0.9:
        print(f"  Q2 FAIL   signal-bearing tables correlate {m_corr:.2f} ({detail}) —")
        print("            one table fits every distance. That is cap-pair, which")
        print("            already reached parity and never beat it.")
    else:
        print(f"  Q2 PASS   signal-bearing tables correlate only {m_corr:.2f} "
              f"({detail}):")
        print("            relations are genuinely distance-specific — precisely")
        print("            what cap-pair's single table could not express.")


if __name__ == "__main__":
    main()
