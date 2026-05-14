"""GridWorld — multi-agent shuttle environment.

Multiple agents shuttle indefinitely between cells A and B on a square grid.
The task is non-episodic: agents alternate roles (A→B then B→A) as they pick
up and drop off items, and learn to avoid head-on collisions.

A head-on collision is defined as one agent moving onto a cell occupied by
another agent travelling in the opposite role (one carrying, one empty).
Collisions on cells A and B themselves are excluded by spec.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# Four discrete actions: North, South, East, West.  No wait action.
ACTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, 1), (0, -1)]
ACTION_NAMES: List[str] = ["N", "S", "E", "W"]
STATE_DIM: int = 9   # 4 vector components + carry flag + 4 sensor bits


@dataclass
class EnvConfig:
    """Reward and shaping coefficients for the environment."""

    grid: int = 5
    n_agents: int = 4
    collision_penalty: float = -10.0
    step_penalty: float = -0.05
    progress_coef: float = 0.5
    pickup_reward: float = 5.0
    delivery_reward: float = 10.0
    near_opposite_penalty: float = -0.2
    oscillation_penalty: float = -0.3
    use_opposite_sensor: bool = True


class GridWorld:
    """Multi-agent grid world for the shuttle coordination task.

    Each agent has a binary role determined by ``carry`` flag:
      - ``carry == 1`` (A2B):  carrying an item, target cell is B.
      - ``carry == 0`` (B2A):  empty, target cell is A.
    The role flips automatically when the agent reaches its target.

    Args:
        cfg: Environment configuration.  If ``None``, uses defaults.

    Attributes:
        A:           Tuple ``(row, col)`` of the pickup cell.
        B:           Tuple ``(row, col)`` of the drop-off cell.
        pos:         List of agent positions.
        carry:       List of agent carry flags (0 or 1).
        collisions:  Cumulative head-on collision count since instantiation.
        deliveries:  Per-agent count of completed round-trips.
    """

    def __init__(self, cfg: Optional[EnvConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else EnvConfig()
        self.G: int = self.cfg.grid
        self.N: int = self.cfg.n_agents
        self.A: Tuple[int, int] = (0, 0)
        self.B: Tuple[int, int] = (0, 0)
        self.pos: List[Tuple[int, int]] = []
        self.carry: List[int] = []
        self.prev_pos: List[Optional[Tuple[int, int]]] = []
        self.start_pos: List[Tuple[int, int]] = []
        self.deliveries: List[int] = []
        self.collisions: int = 0
        self.reset_layout()

    # ---------- internal helpers ----------
    def _rnd_cell(self) -> Tuple[int, int]:
        return random.randrange(self.G), random.randrange(self.G)

    # ---------- public API ----------
    def reset_layout(self) -> None:
        """Randomise A, B, and each agent's starting cell and role.

        Roles are independent per agent.  An agent starting at A always
        carries, one starting at B is always empty.
        """
        self.A = self._rnd_cell()
        self.B = self._rnd_cell()
        while self.B == self.A:
            self.B = self._rnd_cell()
        self.carry = [random.choice([0, 1]) for _ in range(self.N)]
        self.pos = [self.A if c == 1 else self.B for c in self.carry]
        self.prev_pos = [None] * self.N
        self.start_pos = list(self.pos)
        self.deliveries = [0] * self.N

    def state(self, i: int) -> np.ndarray:
        """Compute the observation for agent ``i``.

        Returns a 9-dimensional ``float32`` array containing, in order:
          - dx, dy to A (normalised by grid size)
          - dx, dy to B (normalised by grid size)
          - carry flag (1.0 if carrying)
          - 4 opposite-direction-agent sensors for N/S/E/W neighbours,
            each 1.0 if an opposite-role agent currently occupies that cell.

        Args:
            i: Index of the agent whose observation to compute.

        Returns:
            ``np.ndarray`` of shape ``(9,)`` and dtype ``float32``.
        """
        x, y = self.pos[i]
        G = self.G
        dxA, dyA = (self.A[0] - x) / G, (self.A[1] - y) / G
        dxB, dyB = (self.B[0] - x) / G, (self.B[1] - y) / G
        c = float(self.carry[i])

        sens = [0.0, 0.0, 0.0, 0.0]
        if self.cfg.use_opposite_sensor:
            my_dir = 1 if self.carry[i] == 1 else -1
            for k, (dx, dy) in enumerate(ACTIONS):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < G and 0 <= ny < G):
                    continue
                for j in range(self.N):
                    if j == i:
                        continue
                    if self.pos[j] == (nx, ny):
                        their_dir = 1 if self.carry[j] == 1 else -1
                        if their_dir == -my_dir:
                            sens[k] = 1.0
                            break
        return np.array([dxA, dyA, dxB, dyB, c, *sens], dtype=np.float32)

    def step_one(self, i: int, a: int) -> Tuple[float, bool]:
        """Apply action ``a`` for agent ``i`` and return ``(reward, collided)``.

        Sequential update: callers should iterate over agents in the desired
        order (random or round-robin), calling ``step_one`` once per agent
        per outer tick.

        On collision the agent does not actually move and is penalised.

        Args:
            i: Index of the moving agent.
            a: Action index in ``[0, 4)``.

        Returns:
            Tuple of ``(reward, collided_flag)``.
        """
        cfg = self.cfg
        dx, dy = ACTIONS[a]
        x, y = self.pos[i]
        nx = max(0, min(self.G - 1, x + dx))
        ny = max(0, min(self.G - 1, y + dy))
        nxt = (nx, ny)

        # Head-on collision check (spec: ignored on A and B cells).
        collided = False
        if nxt != self.A and nxt != self.B:
            my_dir = 1 if self.carry[i] == 1 else -1
            for j in range(self.N):
                if j == i:
                    continue
                their_dir = 1 if self.carry[j] == 1 else -1
                if their_dir != -my_dir:
                    continue
                if self.pos[j] == nxt:
                    collided = True
                    break

        r = cfg.step_penalty
        if collided:
            r += cfg.collision_penalty
            self.collisions += 1
            return r, True

        # Manhattan progress shaping toward current goal.
        goal = self.B if self.carry[i] == 1 else self.A
        old_d = abs(x - goal[0]) + abs(y - goal[1])
        new_d = abs(nx - goal[0]) + abs(ny - goal[1])
        r += cfg.progress_coef * (old_d - new_d)

        self.pos[i] = nxt

        # Pickup / delivery → role flips.
        if self.carry[i] == 1 and nxt == self.B:
            self.carry[i] = 0
            r += cfg.delivery_reward
            if self.start_pos[i] == self.B:
                self.deliveries[i] += 1
                self.start_pos[i] = self.B
        elif self.carry[i] == 0 and nxt == self.A:
            self.carry[i] = 1
            r += cfg.pickup_reward
            if self.start_pos[i] == self.A:
                self.deliveries[i] += 1
                self.start_pos[i] = self.A

        # Mild "adjacent to opposite-role agent" penalty.
        for j in range(self.N):
            if j == i:
                continue
            px, py = self.pos[j]
            if abs(nx - px) + abs(ny - py) == 1:
                dir_i = 1 if self.carry[i] == 1 else -1
                dir_j = 1 if self.carry[j] == 1 else -1
                if dir_i != dir_j:
                    r += cfg.near_opposite_penalty

        # Oscillation penalty for stepping straight back.
        if self.prev_pos[i] == nxt:
            r += cfg.oscillation_penalty
        self.prev_pos[i] = (x, y)

        return r, False

    @staticmethod
    def state_dim() -> int:
        """Return the observation-space dimension."""
        return STATE_DIM
