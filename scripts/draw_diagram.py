"""Renders diagrams/graph.png — a static picture of the LangGraph wiring
in src/graph.py, for the submission requirement of an uploaded PNG/JPG."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath

fig, ax = plt.subplots(figsize=(13, 9))
ax.set_xlim(0, 13)
ax.set_ylim(0, 9)
ax.axis("off")

NODE_COLOR = "#dbe9ff"
DECISION_COLOR = "#fff2cc"
TERMINAL_COLOR = "#d9f2d9"
EDGE_COLOR = "#444444"


def box(x, y, w, h, text, color=NODE_COLOR, fontsize=10, style="round,pad=0.15"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=1.4,
                            edgecolor="#222222", facecolor=color)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return (x, y, w, h)


def center_top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def center_bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


def center_left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def center_right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def arrow(p1, p2, label=None, color=EDGE_COLOR, style="-", connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.3, color=color, linestyle=style,
                         connectionstyle=connectionstyle)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.12, label, fontsize=8, ha="center", color=color,
                style="italic", backgroundcolor="white")


start = box(5.6, 8.3, 1.8, 0.5, "START", color="#eeeeee", fontsize=10)
triage_intent = box(5.1, 7.2, 2.8, 0.7, "triage_intent\n(keyword heuristics)", DECISION_COLOR)
retrieval = box(5.1, 6.0, 2.8, 0.7, "retrieval\n(embedding search, top-k)", NODE_COLOR)
triage_finalize = box(5.1, 4.8, 2.8, 0.7, "triage_finalize\n(score -> answerable / clarify)", DECISION_COLOR)
generation = box(5.1, 3.6, 2.8, 0.7, "generation\n(local LLM, evidence-only prompt)", NODE_COLOR)
verification = box(5.1, 2.4, 2.8, 0.7, "verification\n(grounding / citation / schema checks)", DECISION_COLOR)
prepare_revision = box(9.0, 3.0, 2.7, 0.7, "prepare_revision\n(+feedback, revision_count += 1)\nmax 1 retry", "#ffe0cc", fontsize=9)
finalize = box(5.1, 1.0, 2.8, 0.7, "finalize\n(assemble output JSON)", NODE_COLOR)
end = box(5.6, 0.0, 1.8, 0.6, "END", color="#eeeeee")

# left shortcut lane for the three routes that bypass generation entirely
shortcuts = box(0.3, 4.5, 3.0, 3.4, "", color="#f7f7f7", style="round,pad=0.2")
ax.text(1.8, 7.65, "shortcut routes\n(never touch generation)", fontsize=8.5, ha="center",
        style="italic", color="#666666")

arrow(center_bottom(start), center_top(triage_intent))
arrow(center_bottom(triage_intent), center_top(retrieval), label="pending")
arrow(center_bottom(retrieval), center_top(triage_finalize))
arrow(center_bottom(triage_finalize), center_top(generation), label="answerable")
arrow(center_bottom(generation), center_top(verification))

# verification -> prepare_revision -> generation loop (right-hand lane)
arrow(center_right(verification), center_left(prepare_revision), label="fail, retry left",
      connectionstyle="arc3,rad=0.2")
arrow((prepare_revision[0] + prepare_revision[2] * 0.15, prepare_revision[1] + prepare_revision[3]),
      (generation[0] + generation[2], generation[1] + generation[3] * 0.6),
      label="regenerate", connectionstyle="arc3,rad=-0.3")

# verification -> finalize (pass) straight down
arrow(center_bottom(verification), center_top(finalize), label="pass")
# verification -> finalize (fail, no retries left / safe failure) via right lane
arrow((prepare_revision[0] + prepare_revision[2] * 0.5, prepare_revision[1]),
      (7.9, 1.35), label="fail, no retries left\n(safe failure)", connectionstyle="arc3,rad=0.25")
arrow((7.9, 1.35), center_right(finalize))

# out-of-scope / requires_escalation shortcut
arrow(center_left(triage_intent), (1.8, 7.6), label="out_of_scope /\nrequires_escalation",
      connectionstyle="arc3,rad=0.0")
# requires_clarification shortcut
arrow(center_left(triage_finalize), (1.8, 5.0), label="requires_clarification",
      connectionstyle="arc3,rad=0.0")

arrow((1.8, 6.9), (1.8, 5.4), color="#999999")  # visual connector inside shortcut lane
arrow((1.8, 5.4), (1.8, 4.6), color="#999999")
arrow((1.8, 4.6), center_left(finalize), label="finalize (safe response,\nno model call)", color="#999999")

arrow(center_bottom(finalize), center_top(end))

ax.text(0.2, 8.6, "Local-First Support Agent Network — LangGraph wiring", fontsize=14, weight="bold")
ax.text(0.2, 8.25, "Boxes: nodes  |  Yellow/orange: routing or bookkeeping  |  Grey: entry/exit",
        fontsize=9, color="#555555")

plt.tight_layout()
plt.savefig("diagrams/graph.png", dpi=170, bbox_inches="tight")
print("saved diagrams/graph.png")
