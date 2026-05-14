"""Animated visualisation of agents on the grid.

Renders the 5x5 grid, the A and B cells, and four colour-coded agents.
Filled circle = carrying.  Hollow ring = empty.  Stacked agents are
offset to the corners of a small ring so all four remain visible.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .agent import DQNAgent
from .env import ACTION_NAMES, EnvConfig, GridWorld

AGENT_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]


def roll_out(agent: DQNAgent, env_cfg: EnvConfig, max_steps: int = 30,
             seed: Optional[int] = None,
             central_clock: bool = True,
             force_focal_at_B: bool = True) -> List[Dict[str, Any]]:
    """Run one greedy episode and capture every micro-step.

    Args:
        agent:           The trained agent.
        env_cfg:         Environment config (provides n_agents and grid size).
        max_steps:       Max outer ticks (each tick = all agents move once).
        seed:            Scenario seed.  If ``None``, current RNG state is used.
        central_clock:   Round-robin update order.
        force_focal_at_B: Force agent 0 to start at B empty, so the rollout
                         always demonstrates a B→A→B round-trip.

    Returns:
        List of per-micro-step frame dicts ready for ``animate_rollout``.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = GridWorld(env_cfg)
    if force_focal_at_B:
        env.pos[0] = env.B
        env.carry[0] = 0
        env.start_pos[0] = env.B
        env.deliveries[0] = 0

    saved_eps = agent.eps
    agent.eps = 0.0

    frames = [dict(
        positions=list(env.pos), carries=list(env.carry),
        A=env.A, B=env.B, mover=-1, action=None,
        collided=False, step=0, focal_deliveries=env.deliveries[0],
    )]

    for outer in range(max_steps):
        order = (list(range(env.N)) if central_clock
                 else random.sample(range(env.N), env.N))
        for idx in order:
            s = env.state(idx)
            a = agent.select_action(s, greedy=True)
            _, collided = env.step_one(idx, a)
            frames.append(dict(
                positions=list(env.pos), carries=list(env.carry),
                A=env.A, B=env.B, mover=idx, action=a,
                collided=collided, step=outer + 1,
                focal_deliveries=env.deliveries[0],
            ))
        if env.deliveries[0] >= 1:
            break

    agent.eps = saved_eps
    return frames


def animate_rollout(frames: List[Dict[str, Any]], grid: int = 5,
                    title: str = "Agents", fps: int = 3) -> FuncAnimation:
    """Animate a rollout captured by ``roll_out``.

    Args:
        frames: Output of ``roll_out``.
        grid:   Grid size (used for axis limits).
        title:  Plot title.
        fps:    Frames per second.

    Returns:
        ``FuncAnimation``.  Use ``.to_jshtml()`` in Jupyter or
        ``.save(path, writer=PillowWriter(fps=fps))`` to write a GIF.
    """
    n_agents = len(frames[0]["positions"])
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    fig.subplots_adjust(top=0.85, bottom=0.08, left=0.08, right=0.72)
    ax.set_xlim(-0.5, grid - 0.5)
    ax.set_ylim(-0.5, grid - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(range(grid))
    ax.set_yticks(range(grid))
    ax.grid(True, color="lightgray", linewidth=0.8)

    A0 = frames[0]["A"]
    B0 = frames[0]["B"]
    ax.add_patch(mpatches.Rectangle((A0[1] - 0.45, A0[0] - 0.45), 0.9, 0.9,
                                    fc="#e8f6e9", ec="#27ae60",
                                    lw=2.5, alpha=0.85, zorder=1))
    ax.add_patch(mpatches.Rectangle((B0[1] - 0.45, B0[0] - 0.45), 0.9, 0.9,
                                    fc="#fdebd0", ec="#d68910",
                                    lw=2.5, alpha=0.85, zorder=1))
    ax.text(A0[1] - 0.38, A0[0] - 0.30, "A", ha="left", va="top",
            fontsize=14, fontweight="bold", color="#1e8449", zorder=2)
    ax.text(B0[1] - 0.38, B0[0] - 0.30, "B", ha="left", va="top",
            fontsize=14, fontweight="bold", color="#b9770e", zorder=2)

    scatters, labels = [], []
    for i in range(n_agents):
        s = ax.scatter([], [], s=380, c=AGENT_COLORS[i % len(AGENT_COLORS)],
                       ec="black", linewidths=1.5, zorder=5)
        t = ax.text(0, 0, str(i), ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white", zorder=6)
        scatters.append(s)
        labels.append(t)

    legend_elems = [
        mpatches.Patch(fc="#e8f6e9", ec="#27ae60", label="A (pickup)"),
        mpatches.Patch(fc="#fdebd0", ec="#d68910", label="B (drop-off)"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#666",
                   markeredgecolor="#666", markersize=12, label="carrying (A→B)"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="#666", markersize=12, label="empty (B→A)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left",
              bbox_to_anchor=(1.04, 1.0), fontsize=9,
              framealpha=0.9, title="Legend")

    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.40, y=0.94)
    info = fig.text(0.40, 0.89, "", ha="center", fontsize=10, family="monospace")
    flash = mpatches.Circle((0, 0), 0.42, fc="red", ec="darkred",
                            lw=2.5, alpha=0.0, zorder=4)
    ax.add_patch(flash)

    ring = [(-0.18, -0.18), (0.18, -0.18), (-0.18, 0.18), (0.18, 0.18)]

    def update(frame_i: int):
        f = frames[frame_i]
        bucket = defaultdict(list)
        for i in range(n_agents):
            bucket[f["positions"][i]].append(i)
        offsets = [(0.0, 0.0)] * n_agents
        for cell, ids in bucket.items():
            if len(ids) > 1:
                for k, i in enumerate(ids):
                    offsets[i] = ring[k % 4]

        for i in range(n_agents):
            x, y = f["positions"][i]
            ox, oy = offsets[i]
            scatters[i].set_offsets([[y + oy, x + ox]])
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            if f["carries"][i] == 1:
                scatters[i].set_facecolor(color)
                labels[i].set_color("white")
            else:
                scatters[i].set_facecolor("white")
                labels[i].set_color(color)
            scatters[i].set_edgecolor(color)
            labels[i].set_position((y + oy, x + ox))

        mover = f["mover"]
        action_name = ACTION_NAMES[f["action"]] if f["action"] is not None else "-"
        info.set_text(f"tick={f['step']:>2}   "
                      f"mover={'-' if mover < 0 else mover}   "
                      f"action={action_name}   "
                      f"focal_deliveries={f['focal_deliveries']}")

        if f["collided"] and mover >= 0:
            x, y = f["positions"][mover]
            flash.center = (y, x)
            flash.set_alpha(0.7)
        else:
            flash.set_alpha(0.0)

        return [*scatters, *labels, info, flash]

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 / fps, blit=False, repeat=True)
    plt.close(fig)
    return anim
