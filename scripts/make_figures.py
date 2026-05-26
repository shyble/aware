#!/usr/bin/env python3
"""Generate caps_primitive figures.

Produces 5 PNGs into papers/caps_primitive/figures/:
  1. window_sweep.png          - val PPL vs cap-input window size
  2. phase_b_comparison.png    - cap-placement comparison bar chart
  3. architecture_cap_input.png   - cap-input mechanism (Config 3)
  4. architecture_cap_memory.png  - cap-memory attention (Configs 1, 2)
  5. architecture_cap_pair.png    - cap-pair attention (Configs 1, 2)

Run from repo root:
    python3 scripts/make_figures.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTDIR = "papers/caps_primitive/figures"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  wrote {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bar chart 1: window sweep
# ---------------------------------------------------------------------------
def fig_window_sweep():
    # Multi-seed values from paper §6.3 / §6.5
    # (w=5 is single-seed; baseline pure_transformer is 3-seed)
    labels = [
        "pure\ntransformer",
        "cap\nw=1",
        "cap\nw=2",
        "cap\nw=3",
        "cap\nw=4",
        "cap\nw=5",
        "cap\nw=8",
    ]
    means = [28.00, 28.77, 13.95, 13.71, 15.32, 18.86, 27.94]
    stds = [0.11, 0.92, 0.26, 0.33, 0.27, 0.0, 1.46]
    seeds = ["3", "3", "3", "3", "3", "1", "3"]

    # Color scheme: gray for baseline + no-activation, teal for activation regime
    colors = ["#555555"] + ["#999999"] + ["#2a9d8f"] * 3 + ["#999999"] * 2
    # darker teal for the winner
    colors[3] = "#1d6a5e"

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        error_kw={"ecolor": "black", "lw": 1},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Validation perplexity (lower is better)")
    ax.set_title(
        "Cap-input window sweep on TinyStories (~895K params, multi-seed)",
        fontsize=11,
    )
    ax.set_ylim(0, 36)
    ax.axhline(y=28.00, color="black", linestyle="--", linewidth=0.7, alpha=0.4)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

    # Annotate values above each bar
    for bar, m, s, n in zip(bars, means, stds, seeds):
        label = f"{m:.2f}" if s == 0 else f"{m:.2f}\n±{s:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + s + 0.7,
            label,
            ha="center",
            fontsize=8,
        )

    # Highlight winner
    ax.annotate(
        "51% reduction\nvs baseline",
        xy=(3, means[3]),
        xytext=(4.6, 6.5),
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#1d6a5e",
        arrowprops=dict(arrowstyle="->", color="#1d6a5e", lw=1.2),
    )

    # Activation regime band
    ax.axvspan(1.5, 4.5, alpha=0.08, color="#2a9d8f")
    ax.text(
        3.0,
        33.5,
        "activation regime (w ∈ {2, 3, 4})",
        ha="center",
        fontsize=9,
        color="#1d6a5e",
        style="italic",
    )

    save(fig, "window_sweep.png")


# ---------------------------------------------------------------------------
# Bar chart 2: Phase B placement comparison
# ---------------------------------------------------------------------------
def fig_phase_b():
    # From paper §6.2 (single-seed Phase B values)
    labels = [
        "pure\ntransformer",
        "cap-input\nw=3\n(winner)",
        "cap-memory\nlocal",
        "cap-memory\nshared",
        "cap-pair\nlocal",
    ]
    values = [28.42, 13.71, 32.64, 32.48, 27.75]
    params = ["853K", "895K", "722K", "722K", "1.77M"]

    colors = ["#555555", "#1d6a5e", "#e76f51", "#e76f51", "#f4a261"]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Validation perplexity (lower is better)")
    ax.set_title("Cap placement comparison (Phase B, single seed)", fontsize=11)
    ax.set_ylim(0, 40)
    ax.axhline(y=28.42, color="black", linestyle="--", linewidth=0.7, alpha=0.4)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

    for bar, v, p in zip(bars, values, params):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.6,
            f"{v:.2f}\n({p})",
            ha="center",
            fontsize=8,
        )

    # MVP caveat note
    ax.text(
        0.99,
        0.97,
        "(cap-memory / cap-pair: single-head MVP)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        style="italic",
        color="#666",
    )

    save(fig, "phase_b_comparison.png")


# ---------------------------------------------------------------------------
# Architecture diagrams - shared helpers
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, text, color="#e8f1f8", textcolor="black", fontsize=9):
    """Rounded rectangle with centered text."""
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.0,
        edgecolor="black",
        facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
    )


def arrow(ax, x1, y1, x2, y2, label=None, label_offset=(0.1, 0.0), color="black"):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.2,
        color=color,
    )
    ax.add_patch(a)
    if label:
        ax.text(
            (x1 + x2) / 2 + label_offset[0],
            (y1 + y2) / 2 + label_offset[1],
            label,
            fontsize=8,
            color="#444",
            style="italic",
        )


def plus_node(ax, x, y, r=0.18):
    """Draw a circled + (add operator)."""
    circle = plt.Circle((x, y), r, facecolor="white", edgecolor="black", linewidth=1)
    ax.add_patch(circle)
    ax.text(x, y, "+", ha="center", va="center", fontsize=12, fontweight="bold")


# ---------------------------------------------------------------------------
# Architecture diagram: cap-input (Config 3, the headline mechanism)
# ---------------------------------------------------------------------------
# IMPORTANT: the cap layer REPLACES the embedding output in the residual
# stream - it does NOT add to it. The transformer blocks receive
# cap_layer(embed(tokens)), not embed + cap_layer(embed). See
# substrate.rs::forward:
#     let embeds = self.embed.forward(tokens)?;
#     let mut x = embeds.clone();
#     if let Some(cl) = &self.cap_layer { x = cl.forward(&x)?; }
def fig_arch_cap_input():
    fig, ax = plt.subplots(figsize=(11, 11.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 13)
    ax.axis("off")
    ax.set_title(
        "Cap-input architecture (Config 3)",
        fontsize=12,
        pad=12,
    )

    # Single main column
    col_x, col_w = 3.5, 4.0
    col_cx = col_x + col_w / 2

    # ---- Cap layer outer group (green band background) ----
    cap_top = 10.0
    cap_sub_h = 0.7
    cap_gap = 0.18
    cap_steps = [
        "build_windowed_input\n(B, S, W·d_emb)",
        "CapMatrix.fire(...)\n-> (B, S, n_caps)",
        "GELU",
        "Linear (n_caps -> d_model)\n-> (B, S, d_model)",
    ]
    n_cap = len(cap_steps)
    cap_total_h = n_cap * cap_sub_h + (n_cap - 1) * cap_gap + 0.6  # extra for label
    cap_band = mpatches.FancyBboxPatch(
        (col_x - 0.4, cap_top - cap_total_h),
        col_w + 0.8,
        cap_total_h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.0,
        edgecolor="#1d6a5e",
        facecolor="#eef9f4",
        linestyle="--",
    )
    ax.add_patch(cap_band)
    # "Cap Layer" label at the top of the band
    ax.text(
        col_x - 0.25,
        cap_top - 0.32,
        "Cap Layer",
        fontsize=10,
        fontweight="bold",
        color="#1d6a5e",
        ha="left",
    )

    # ---- Top of figure: token_ids -> Embedding ----
    box(ax, col_x, 11.9, col_w, 0.7, "token_ids  (B, S)", color="#fff5e6")
    box(ax, col_x, 10.7, col_w, 0.7, "Embedding (vocab -> d_emb)", color="#e8f1f8")
    arrow(ax, col_cx, 11.9, col_cx, 11.4)
    arrow(ax, col_cx, 10.7, col_cx, 10.0 + 0.0)  # tip just above band top

    # ---- Cap layer steps inside the band ----
    cap_label_offset = 0.55  # below the "Cap Layer" label
    cap_box_ys = []
    for i, txt in enumerate(cap_steps):
        y = cap_top - cap_label_offset - (i + 1) * cap_sub_h - i * cap_gap
        box(ax, col_x, y, col_w, cap_sub_h, txt, color="#d4edda", fontsize=8.5)
        cap_box_ys.append(y)
        if i > 0:
            arrow(ax, col_cx, cap_box_ys[i - 1], col_cx, y + cap_sub_h)

    cap_out_y = cap_box_ys[-1]  # bottom of last cap step (= bottom of Linear box)
    cap_band_bottom = cap_top - cap_total_h

    # ---- Below the band: arrow into transformer blocks ----
    block_y = cap_band_bottom - 1.2
    box(
        ax,
        col_x,
        block_y,
        col_w,
        0.9,
        "Transformer blocks × N\n(MHA + FFN, Pre-LN)",
        color="#e8f1f8",
    )
    arrow(ax, col_cx, cap_band_bottom, col_cx, block_y + 0.9)

    # ---- Final RMSNorm ----
    norm_y = block_y - 1.05
    box(ax, col_x, norm_y, col_w, 0.7, "Final RMSNorm", color="#e8f1f8")
    arrow(ax, col_cx, block_y, col_cx, norm_y + 0.7)

    # ---- Unembed (tied: uses embed.weight().T, no separate Linear) ----
    unembed_y = norm_y - 1.2
    box(
        ax,
        col_x,
        unembed_y,
        col_w,
        0.7,
        "Tied unembed\n(final_hidden @ embed.T)",
        color="#e8f1f8",
    )
    arrow(ax, col_cx, norm_y, col_cx, unembed_y + 0.7)

    # ---- Logits ----
    logits_y = unembed_y - 1.05
    box(
        ax,
        col_x,
        logits_y,
        col_w,
        0.7,
        "logits  (B, S, vocab)",
        color="#fff5e6",
    )
    arrow(ax, col_cx, unembed_y, col_cx, logits_y + 0.7)

    # ---- Side annotation (right side, clear of main column) ----
    ax.text(
        col_x + col_w + 0.5,
        cap_top - 1.5,
        "The Cap Layer is on the critical\npath: it transforms embeddings\n(d_emb) into block input (d_model).\nWhen present, blocks see\ncap_layer(embed(tokens)),\nnot embed(tokens) itself.",
        fontsize=8.5,
        color="#1d6a5e",
        va="top",
    )

    save(fig, "architecture_cap_input.png")


# ---------------------------------------------------------------------------
# Architecture diagram: cap-memory attention
# ---------------------------------------------------------------------------
def fig_arch_cap_memory():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(-1, 10)
    ax.axis("off")
    ax.set_title(
        "Cap-memory attention - caps as a positionless K/V bank",
        fontsize=12,
        pad=12,
    )

    # Input hidden state (top-left)
    box(ax, 0.5, 8.2, 2.8, 0.7, "h (B, S, d_model)", color="#fff5e6")

    # Q projection
    box(ax, 0.5, 6.3, 2.8, 0.8, "Q = h · W_Q\n(B, S, d_model)", color="#e8f1f8")
    arrow(ax, 1.9, 8.2, 1.9, 7.1)

    # CapMatrix on the right
    box(
        ax,
        9.0,
        6.0,
        3.7,
        1.6,
        "CapMatrix\n• keys K_cap (n_caps, d_model)\n• values V_cap (n_caps, d_model)\n• n_caps stable ids",
        color="#d4edda",
    )

    # attention scores
    box(
        ax,
        3.7,
        4.0,
        7.0,
        0.9,
        "scores = softmax(Q · K_capᵀ / √d_model)\n-> (B, S, n_caps)   [no RoPE, no causal mask]",
        color="#fde2cf",
    )
    # Q -> scores
    arrow(ax, 1.9, 6.3, 4.5, 4.9)
    # K_cap -> scores
    arrow(ax, 10.0, 6.0, 9.5, 4.9, color="#666")
    ax.text(10.1, 5.6, "K_cap", fontsize=8, color="#666", style="italic")

    # attention output
    box(
        ax,
        3.7,
        2.2,
        7.0,
        0.9,
        "attn_out = scores · V_cap\n-> (B, S, d_model)",
        color="#fde2cf",
    )
    # scores -> attn_out
    arrow(ax, 7.2, 4.0, 7.2, 3.1)
    # V_cap -> attn_out (clean L-route on the right)
    ax.plot([10.8, 11.6], [6.0, 6.0], color="#666", lw=1)
    ax.plot([11.6, 11.6], [6.0, 2.65], color="#666", lw=1)
    arrow(ax, 11.6, 2.65, 10.7, 2.65, color="#666")
    ax.text(11.7, 4.3, "V_cap", fontsize=8, color="#666", style="italic")

    # Output projection
    box(ax, 3.7, 0.7, 7.0, 0.7, "Output projection -> residual", color="#e8f1f8")
    arrow(ax, 7.2, 2.2, 7.2, 1.4)

    # Distinguishing note (placed clearly below the output box)
    ax.text(
        0.5,
        -0.55,
        "K and V come from the cap matrix (positionless, stable ids),\nnot from input projections.",
        fontsize=8.5,
        color="#1d6a5e",
        style="italic",
    )

    save(fig, "architecture_cap_memory.png")


# ---------------------------------------------------------------------------
# Architecture diagram: cap-pair attention
# ---------------------------------------------------------------------------
def fig_arch_cap_pair():
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(
        "Cap-pair attention - attention scores from cap-cap affinity",
        fontsize=12,
        pad=12,
    )

    # Input hidden state (top-left)
    box(ax, 1.0, 8.6, 3.0, 0.7, "h (B, S, d_model)", color="#fff5e6")
    h_cx = 2.5

    # CapMatrix.keys (top-right)
    box(
        ax,
        9.5,
        8.4,
        3.2,
        1.1,
        "CapMatrix.keys\n(n_caps, d_in)\nstable ids",
        color="#d4edda",
    )

    # cap_acts (center-left, below h)
    box(
        ax,
        0.5,
        6.4,
        5.5,
        0.9,
        "cap_acts = GELU(h · K_capᵀ)\n-> (B, S, n_caps)",
        color="#fde2cf",
    )
    # h -> cap_acts (straight down)
    arrow(ax, h_cx, 8.6, h_cx, 7.3)
    # K_cap -> cap_acts (diagonal, gray)
    arrow(ax, 9.5, 8.6, 6.0, 7.0, color="#666")
    ax.text(7.5, 7.9, "K_cap", fontsize=8.5, color="#666", style="italic")

    # A_pair (center-right)
    box(
        ax,
        7.5,
        4.6,
        4.5,
        1.0,
        "Learned A_pair\n(n_caps, n_caps)\ncap-cap affinity",
        color="#d4edda",
    )

    # attention scores
    box(
        ax,
        2.0,
        2.7,
        9.0,
        1.0,
        "scores = softmax( (cap_acts · A_pair · cap_actsᵀ) / √d_model )\n-> (B, S, S)  (causal masked)",
        color="#fde2cf",
    )
    # cap_acts -> scores
    arrow(ax, 3.25, 6.4, 4.0, 3.7)
    # A_pair -> scores
    arrow(ax, 9.75, 4.6, 8.0, 3.7)

    # V projection (bottom-left, h fans out down-left from its LEFT edge)
    box(ax, 0.5, 0.9, 3.2, 0.8, "V = h · W_V\n(B, S, d_model)", color="#e8f1f8")
    # h -> V: exit h from its LEFT edge midpoint (not from inside the box)
    h_left_x = 1.0      # left edge of h box
    h_left_y = 8.95     # mid-height of h box
    ax.plot([h_left_x, 0.3], [h_left_y, h_left_y], color="black", lw=1.2)
    ax.plot([0.3, 0.3], [h_left_y, 1.3], color="black", lw=1.2)
    arrow(ax, 0.3, 1.3, 0.5, 1.3)

    # output (attn_out -> W_O)
    box(
        ax,
        4.5,
        0.9,
        7.5,
        0.8,
        "attn_out = (scores · V) · W_O  -> residual",
        color="#e8f1f8",
    )
    # scores -> attn_out
    arrow(ax, 8.25, 2.7, 8.25, 1.7)
    # V -> attn_out
    arrow(ax, 3.7, 1.3, 4.5, 1.3)

    # Distinguishing note (bottom-left)
    ax.text(
        0.5,
        0.1,
        "Attention scores come from cap-activation pairs through a learned (n_caps × n_caps) affinity,\nnot from token-to-token Q·Kᵀ. Values still come from input.",
        fontsize=8.5,
        color="#1d6a5e",
        style="italic",
    )

    save(fig, "architecture_cap_pair.png")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Writing figures to {OUTDIR}/")
    fig_window_sweep()
    fig_phase_b()
    fig_arch_cap_input()
    fig_arch_cap_memory()
    fig_arch_cap_pair()
    print("done.")
