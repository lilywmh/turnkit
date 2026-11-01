"""Draw docs/example.png from the sample conversation. Requires matplotlib."""
from pathlib import Path

import matplotlib.pyplot as plt

import turnkit as tk

root = Path(__file__).resolve().parents[1]
turns = tk.build_turns(tk.load_segments(root / "examples" / "sample_conversation.csv"))
switches = tk.transitions(turns)
windows = tk.windowed_features(turns, n_windows=3)

ink, soft, grid = "#1f2933", "#6b7280", "#e5e7eb"
colors = {"A": "#3b6ea8", "B": "#d4874a"}

fig = plt.figure(figsize=(9, 3.6), dpi=200)
gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.28)

ax = fig.add_subplot(gs[0])
for _, t in turns.iterrows():
    ax.barh(1 if t.speaker == "A" else 0, t.duration, left=t.start, height=0.55,
            color=colors[t.speaker], alpha=0.9, edgecolor="white", linewidth=0.8)
for _, s in switches.iterrows():
    ax.axvline(s.time, color="#c2410c" if s.gap < 0 else soft, lw=0.8, ls=(0, (2, 2)), zorder=0)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Speaker B", "Speaker A"], color=ink)
ax.set_xlabel("time (s)", color=soft); ax.set_xlim(0, turns.end.max() + 0.5); ax.set_ylim(-0.6, 1.6)
ax.set_title("Turns and speaker switches", loc="left", color=ink, fontsize=11, fontweight="bold")
ax.text(0.0, -0.32, "dashed lines = switches   orange = overlapping switch",
        transform=ax.transAxes, color=soft, fontsize=8)

ax2 = fig.add_subplot(gs[1])
ax2.plot(windows.window + 1, windows.switches_per_min, marker="o", color="#3b6ea8", lw=2)
ax2.set_xticks([1, 2, 3]); ax2.set_xlabel("third of conversation", color=soft)
ax2.set_title("Switches / min", loc="left", color=ink, fontsize=11, fontweight="bold")
ax2.set_ylim(0, windows.switches_per_min.max() * 1.25); ax2.grid(axis="y", color=grid, lw=0.8)

for a in (ax, ax2):
    for side in ("top", "right", "left"):
        a.spines[side].set_visible(False)
    a.spines["bottom"].set_color(grid)
    a.tick_params(colors=soft, length=0)

fig.savefig(root / "docs" / "example.png", bbox_inches="tight", facecolor="white")
print("Saved docs/example.png")
