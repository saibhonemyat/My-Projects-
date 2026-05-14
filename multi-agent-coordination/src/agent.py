"""DQN network and Double-DQN agent.

A shared Double-DQN policy with Huber loss and gradient clipping.  All agents
in the environment sample actions from the same network and write into the
same replay buffer.  This is the simplest form of multi-agent learning and
works well when agents have symmetric roles.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


@dataclass
class AgentConfig:
    """Hyperparameters for the DQN agent."""

    state_dim: int = 9
    action_dim: int = 4
    hidden_dim: int = 128
    lr: float = 5e-4
    gamma: float = 0.95
    buffer_size: int = 80_000
    batch_size: int = 64
    target_update_every: int = 1000
    train_every: int = 4
    learn_after: int = 2000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 150_000


class DQN(nn.Module):
    """Two-layer MLP that maps a state vector to action-value estimates."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Shared-policy Double-DQN with Huber loss.

    Implements:
      - Double-DQN: action argmax from the online network, target value from
        the target network.  Reduces value-overestimation bias.
      - Huber (smooth L1) loss for stability against outlier TD errors.
      - Gradient norm clipping at 5.0.
      - Periodic target-network sync.

    The same agent object is shared by all grid agents; they call
    ``select_action`` and ``store`` independently and the underlying network
    serves them all.

    Args:
        cfg: Agent hyperparameters.  If ``None``, uses defaults.
    """

    def __init__(self, cfg: "AgentConfig | None" = None) -> None:
        self.cfg = cfg if cfg is not None else AgentConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = DQN(self.cfg.state_dim, self.cfg.action_dim,
                         self.cfg.hidden_dim).to(self.device)
        self.target_net = DQN(self.cfg.state_dim, self.cfg.action_dim,
                              self.cfg.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optim = optim.Adam(self.q_net.parameters(), lr=self.cfg.lr)

        self.memory: Deque[Transition] = deque(maxlen=self.cfg.buffer_size)
        self.action_dim = self.cfg.action_dim
        self.batch_size = self.cfg.batch_size
        self.gamma = self.cfg.gamma

        self.eps_start = self.cfg.eps_start
        self.eps_end = self.cfg.eps_end
        self.eps_decay_steps = self.cfg.eps_decay_steps
        self.eps: float = self.cfg.eps_start

        self.train_steps = 0
        self.experience_count = 0

    # ---------- exploration ----------
    def update_epsilon(self, step: int) -> None:
        """Linear epsilon schedule from ``eps_start`` to ``eps_end``."""
        frac = min(1.0, step / max(1, self.eps_decay_steps))
        self.eps = self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, s: np.ndarray, greedy: bool = False) -> int:
        """Pick an action for state ``s``.

        Args:
            s:      State vector.
            greedy: If True, always pick the argmax.  Otherwise, ε-greedy.
        """
        if not greedy and random.random() < self.eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            t = torch.from_numpy(s).unsqueeze(0).to(self.device)
            q = self.q_net(t)
        return int(torch.argmax(q, dim=1).item())

    # ---------- replay ----------
    def store(self, s: np.ndarray, a: int, r: float,
              s2: np.ndarray, done: bool) -> None:
        """Add a transition to the replay buffer."""
        self.memory.append((s, a, r, s2, done))
        self.experience_count += 1

    def train_step(self) -> None:
        """Run one Double-DQN gradient update if conditions are met.

        Conditions:
          - At least ``learn_after`` experiences stored.
          - This call falls on a ``train_every`` boundary.
          - The buffer has at least ``batch_size`` samples.
        """
        if self.experience_count < self.cfg.learn_after:
            return
        if self.experience_count % self.cfg.train_every != 0:
            return
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, done = zip(*batch)
        s = torch.from_numpy(np.asarray(s, dtype=np.float32)).to(self.device)
        s2 = torch.from_numpy(np.asarray(s2, dtype=np.float32)).to(self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        d = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_pred = self.q_net(s).gather(1, a)
        with torch.no_grad():
            # Double-DQN: argmax from online net, value from target net.
            best_next_a = self.q_net(s2).argmax(dim=1, keepdim=True)
            q_next = self.target_net(s2).gather(1, best_next_a)
            target = r + self.gamma * q_next * (1.0 - d)

        loss = F.smooth_l1_loss(q_pred, target)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optim.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    # ---------- persistence ----------
    def save(self, path: str) -> None:
        """Save the Q-network weights to ``path``."""
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "cfg": self.cfg.__dict__,
        }, path)

    def load(self, path: str) -> None:
        """Load Q-network weights from ``path``."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
