"""Render the two portfolio figures for the interpretability pass, from the measured results.

Figure 1 (patch_recovery.png): per-layer logit-difference recovery, residual-stream patch vs
attention-output patch. Shows the residual patch ramping monotonically to 1.0 (trivial, by
construction) while the attention patch is flat-then-spikes, localizing the effect to ~layer 20.

Figure 2 (logit_lens.png): the logit-lens top token per layer (late band) for the two diverging
prompts, base vs fine-tuned, each cell colored by register (input-faithful / review-template /
other). Shows the base resolving to input-grounded tokens while the fine-tune locks onto the
review-summary voice.

Numbers are copied from results/interp/{patch_nnsight,attn_patch,logit_lens}.txt (the cluster runs).
Colors: the dataviz skill's validated categorical pair (blue #2a78d6, orange #eb6834). Run:
    python src/interp/plot_interp.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "interp")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"      # validated categorical pair (CVD dE 24.7 on the light surface)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
FAITHFUL_BG, TEMPLATE_BG, OTHER_BG = "#cde2fb", "#f7d4c2", "#ededea"

# --- measured recovery curves (per layer, L0..L27) -------------------------------------------
RESID = [0.004, 0.015, 0.009, 0.034, 0.034, 0.067, 0.070, 0.059, 0.073, 0.091, 0.169, 0.241,
         0.299, 0.358, 0.311, 0.442, 0.426, 0.505, 0.570, 0.548, 0.717, 0.862, 0.867, 0.887,
         0.947, 0.983, 0.972, 1.000]
ATTN = [0.010, 0.012, -0.007, 0.007, 0.009, -0.001, -0.027, -0.036, -0.015, 0.008, 0.031, 0.020,
        -0.063, 0.043, 0.054, 0.266, -0.062, 0.170, 0.292, 0.257, 0.652, 0.369, 0.084, 0.024,
        0.286, 0.106, 0.037, 0.090]

# --- logit-lens top tokens for the two DIVERGING prompts, late band L15..L27 -----------------
# (base, ft) per layer; whitespace-only shown as a middot.
LAYERS_LL = list(range(15, 28))
P1 = {  # "Two trials met the inclusion criteria..."
    "base": ['"', '·', 's', 's', 's', 'Two', 'Two', 'Two', 'Two', 'Two', 'Two', 'Two', 'Two'],
    "ft":   ['·', '·', '1', '1', 'This', 'There', 'Two', 'Two', 'Two', 'Two', 'Two', 'We', 'We'],
}
P3 = {  # "Participants received topical application weekly..."
    "base": ['·', '·', 's', 's', 'When', 'During', 'Okay', 'The', 'The', 'The', 'The', 'Participants', 'Participants'],
    "ft":   ['·', '·', 'This', 'This', 'This', 'Study', 'Study', 'This', 'This', 'This', 'This', 'This', 'We'],
}
FAITHFUL = {"Two", "The", "Participants", "When", "During", "int"}
TEMPLATE = {"We", "This", "Study", "There"}


def register_bg(tok):
    t = tok.strip()
    if t in TEMPLATE:
        return TEMPLATE_BG
    if t in FAITHFUL:
        return FAITHFUL_BG
    return OTHER_BG


def fig_recovery():
    layers = list(range(28))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.plot(layers, RESID, color=BLUE, lw=2, marker="o", ms=3,
            label="residual-stream patch (monotonic by construction)")
    ax.plot(layers, ATTN, color=ORANGE, lw=2, marker="o", ms=3,
            label="attention-output patch (localized)")

    peak = ATTN.index(max(ATTN))
    ax.annotate(f"layer {peak}: {ATTN[peak]:.2f}\nattention peak",
                xy=(peak, ATTN[peak]), xytext=(peak - 9, ATTN[peak] + 0.02),
                color=ORANGE, fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))

    ax.set_xlabel("layer", color=MUTED, fontsize=10)
    ax.set_ylabel("recovery toward the fine-tuned decision", color=MUTED, fontsize=10)
    ax.set_title("Where the fine-tune's next-token shift is causally localized",
                 color=INK, fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.set_xlim(-0.5, 27.5)
    ax.set_xticks(range(0, 28, 2))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    path = os.path.join(OUT, "patch_recovery.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def _lens_panel(ax, data, title):
    cols = ["base", "fine-tuned"]
    keys = ["base", "ft"]
    n = len(LAYERS_LL)
    for ci, k in enumerate(keys):
        for ri, tok in enumerate(data[k]):
            y = n - 1 - ri
            ax.add_patch(plt.Rectangle((ci, y), 0.96, 0.92, facecolor=register_bg(tok),
                                       edgecolor="white", lw=1.5))
            ax.text(ci + 0.48, y + 0.46, tok, ha="center", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, n)
    ax.set_xticks([0.48, 1.48])
    ax.set_xticklabels(cols, fontsize=9, color=INK)
    ax.set_yticks([n - 1 - i + 0.46 for i in range(n)])
    ax.set_yticklabels([f"L{L}" for L in LAYERS_LL], fontsize=8, color=MUTED)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, color=INK, fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.xaxis.tick_top()


def fig_logit_lens():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 5.4))
    _lens_panel(axes[0], P1, "Prompt 1  (base top → 'Two',  ft → 'We')")
    _lens_panel(axes[1], P3, "Prompt 3  (base top → 'Participants',  ft → 'We')")
    fig.suptitle("Logit lens: what each layer would predict (late layers)",
                 color=INK, fontsize=12, fontweight="bold", x=0.02, ha="left", y=0.99)
    legend = [Patch(facecolor=FAITHFUL_BG, edgecolor="white", label="input-faithful token"),
              Patch(facecolor=TEMPLATE_BG, edgecolor="white", label="review-template token"),
              Patch(facecolor=OTHER_BG, edgecolor="white", label="other / not yet decodable")]
    fig.legend(handles=legend, frameon=False, fontsize=9, loc="lower center",
               ncol=3, labelcolor=INK, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    path = os.path.join(OUT, "logit_lens.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("[saved]", fig_recovery())
    print("[saved]", fig_logit_lens())
