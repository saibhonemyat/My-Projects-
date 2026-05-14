"""Training loop with curriculum learning and early stopping.

The curriculum runs three phases on the required 5x5 grid:

  1. Single-agent navigation warmup (zero possible collisions).
  2. Two-agent sensor warmup (cheap collision exposure).
  3. Full four-agent coordination with the harshest collision penalty.

A mini-evaluation runs every ``mini_eval_every`` steps during the final
phase; training stops early when success rate clears ``early_stop_rate``.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .agent import AgentConfig, DQNAgent
from .env import EnvConfig, GridWorld

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Hyperparameters for the training loop itself."""

    total_step_cap: int = 1_500_000
    collision_cap: int = 4_000
    walltime_cap: int = 600
    layout_reset_every: int = 200
    mini_eval_every: int = 20_000
    early_stop_rate: float = 0.93
    seed: int = 0
    central_clock: bool = True
    phases: List[Dict[str, Any]] = field(default_factory=list)


def _mini_eval_silent(agent: DQNAgent, env_cfg: EnvConfig,
                      trials: int = 200, max_steps: int = 20,
                      seed: int = 99999, central_clock: bool = True) -> float:
    """Quick greedy success-rate estimate, used for early stopping."""
    rng = random.Random(seed)
    saved_eps = agent.eps
    agent.eps = 0.0
    successes = 0
    for _ in range(trials):
        scen_seed = rng.randint(0, 10**9)
        random.seed(scen_seed)
        np.random.seed(scen_seed)
        env = GridWorld(env_cfg)
        env.pos[0] = env.B
        env.carry[0] = 0
        env.start_pos[0] = env.B
        env.deliveries[0] = 0
        succeeded, collided = False, False
        for _ in range(max_steps):
            order = (list(range(env.N)) if central_clock
                     else random.sample(range(env.N), env.N))
            for idx in order:
                s = env.state(idx)
                a = agent.select_action(s, greedy=True)
                _, c = env.step_one(idx, a)
                if idx == 0 and c:
                    collided = True
            if env.deliveries[0] >= 1:
                succeeded = True
                break
            if collided:
                break
        if succeeded and not collided:
            successes += 1
    agent.eps = saved_eps
    return successes / trials


def train(env_cfg: EnvConfig, agent_cfg: AgentConfig, train_cfg: TrainConfig
          ) -> Tuple[DQNAgent, Dict[str, list], int]:
    """Train a shared-policy Double-DQN agent.

    Args:
        env_cfg:   Base environment config.  Phase params override fields
                   on a copy.
        agent_cfg: Agent hyperparameters.
        train_cfg: Training loop hyperparameters and phase list.

    Returns:
        ``(agent, history, total_collisions)`` where ``history`` is a dict of
        per-step series for plotting.
    """
    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    agent = DQNAgent(agent_cfg)
    history: Dict[str, list] = {"step": [], "collisions": [],
                                "deliveries": [], "eps": []}

    start = time.time()
    total_steps = 0
    cumulative_collisions = 0
    last_log = 0

    for ph_idx, ph in enumerate(train_cfg.phases):
        n_ag = int(ph["n_agents"])
        # Build a per-phase env config from a shallow copy of the base.
        ph_env_cfg = EnvConfig(**env_cfg.__dict__)
        ph_env_cfg.n_agents = n_ag
        for key in ("collision_penalty", "near_opposite_penalty", "progress_coef"):
            if key in ph:
                setattr(ph_env_cfg, key, ph[key])
        env = GridWorld(ph_env_cfg)

        # Per-phase epsilon schedule (restart from current eps each phase).
        agent.eps_start = agent.eps if ph_idx > 0 else agent.cfg.eps_start
        agent.eps_end = float(ph["eps_target"])
        agent.eps_decay_steps = int(ph["eps_decay_steps"])
        eps_phase_start_step = total_steps

        logger.info("phase %d/%d: %s (n_agents=%d, budget=%d)",
                    ph_idx + 1, len(train_cfg.phases), ph["name"],
                    n_ag, ph["budget"])

        ph_steps = 0
        idx_clock = 0
        layout_counter = 0
        mini_eval_check_at = total_steps
        budget = int(ph["budget"])

        while ph_steps < budget and total_steps < train_cfg.total_step_cap:
            if time.time() - start > train_cfg.walltime_cap:
                logger.warning("walltime cap reached")
                break

            if train_cfg.central_clock:
                order = [idx_clock]
                idx_clock = (idx_clock + 1) % n_ag
            else:
                order = random.sample(range(n_ag), n_ag)

            for idx in order:
                s = env.state(idx)
                a = agent.select_action(s)
                r, _ = env.step_one(idx, a)
                s2 = env.state(idx)
                agent.store(s, a, r, s2, False)
                agent.train_step()
                ph_steps += 1
                total_steps += 1
                agent.update_epsilon(total_steps - eps_phase_start_step)
                if total_steps >= train_cfg.total_step_cap:
                    break

            layout_counter += 1
            if layout_counter >= train_cfg.layout_reset_every:
                env.reset_layout()
                layout_counter = 0

            if total_steps - last_log >= 10_000:
                last_log = total_steps
                cur_global_col = cumulative_collisions + env.collisions
                history["step"].append(total_steps)
                history["collisions"].append(cur_global_col)
                history["deliveries"].append(sum(env.deliveries))
                history["eps"].append(agent.eps)
                if total_steps % 50_000 < 1000:
                    elapsed = time.time() - start
                    logger.info("step %d eps=%.3f col=%d del=%d (%.0fs)",
                                total_steps, agent.eps, cur_global_col,
                                sum(env.deliveries), elapsed)

            # Mini-eval early stopping, final phase only.
            if (ph_idx == len(train_cfg.phases) - 1
                    and total_steps - mini_eval_check_at >= train_cfg.mini_eval_every
                    and ph_steps >= 40_000):
                mini_eval_check_at = total_steps
                rate = _mini_eval_silent(agent, env_cfg, trials=200,
                                         max_steps=20,
                                         central_clock=train_cfg.central_clock)
                logger.info("mini-eval at step %d: %.1f%%", total_steps, rate * 100)
                if rate >= train_cfg.early_stop_rate:
                    logger.info("early stop: mini-eval %.1f%% >= %.0f%%",
                                rate * 100, train_cfg.early_stop_rate * 100)
                    cumulative_collisions += env.collisions
                    return agent, history, cumulative_collisions

            if cumulative_collisions + env.collisions >= train_cfg.collision_cap:
                logger.warning("collision cap reached")
                break

        cumulative_collisions += env.collisions
        logger.info("phase done: steps=%d phase_col=%d total_col=%d",
                    ph_steps, env.collisions, cumulative_collisions)

    elapsed = time.time() - start
    logger.info("training done: steps=%d total_col=%d walltime=%.0fs",
                total_steps, cumulative_collisions, elapsed)
    return agent, history, cumulative_collisions
