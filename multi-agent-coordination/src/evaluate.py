"""Evaluation routines for the trained agent.

Two functions:
  - ``evaluate_round_trip``: aggregate success rate over many random trials.
  - ``evaluate_detailed``:   per-trial outcome breakdown for failure analysis.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .agent import DQNAgent
from .env import EnvConfig, GridWorld


@dataclass
class TrialResult:
    """Outcome of a single B → A → B test."""

    trial: int
    scenario_seed: int
    A: Tuple[int, int]
    B: Tuple[int, int]
    optimal_steps: int
    picked_up_at_step: Optional[int]
    delivered_at_step: Optional[int]
    collided_at_step: Optional[int]
    steps_taken: int
    outcome: str


@dataclass
class EvalSummary:
    """Aggregate result of an evaluation run."""

    success_rate: float
    avg_steps_on_success: float
    collisions: int
    n_trials: int


def evaluate_round_trip(agent: DQNAgent, env_cfg: EnvConfig,
                        trials: int = 1000, seed: int = 12345,
                        max_steps: int = 20,
                        central_clock: bool = True,
                        verbose: bool = True) -> EvalSummary:
    """Spec-aligned aggregate evaluation.

    For each trial:
      - Random A, B, and other-agent placement.
      - Focal agent (index 0) forced to start at B empty.
      - Success = full B→A→B round-trip in ≤ ``max_steps`` ticks,
        collision-free.

    Args:
        agent:         The trained agent.  Acts greedily (``eps = 0``).
        env_cfg:       Base environment config (the n_agents from here is used).
        trials:        Number of independent trials.
        seed:          Master RNG seed for scenario generation.
        max_steps:     Per-trial horizon.
        central_clock: Round-robin agent update order.
        verbose:       Print summary line.

    Returns:
        ``EvalSummary``.
    """
    rng = random.Random(seed)
    saved_eps = agent.eps
    agent.eps = 0.0

    successes, total_steps_on_success, collisions = 0, 0, 0
    for _ in range(trials):
        scen_seed = rng.randint(0, 10**9)
        random.seed(scen_seed)
        np.random.seed(scen_seed)
        env = GridWorld(env_cfg)
        env.pos[0] = env.B
        env.carry[0] = 0
        env.start_pos[0] = env.B
        env.deliveries[0] = 0

        succeeded = False
        collided = False
        steps_taken = 0
        for step in range(max_steps):
            order = (list(range(env.N)) if central_clock
                     else random.sample(range(env.N), env.N))
            for idx in order:
                s = env.state(idx)
                a = agent.select_action(s, greedy=True)
                _, c = env.step_one(idx, a)
                if idx == 0 and c:
                    collided = True
            steps_taken = step + 1
            if env.deliveries[0] >= 1:
                succeeded = True
                break
            if collided:
                break
        if succeeded and not collided:
            successes += 1
            total_steps_on_success += steps_taken
        if collided:
            collisions += 1

    agent.eps = saved_eps
    rate = successes / trials
    avg_steps = (total_steps_on_success / successes) if successes else float("nan")

    if verbose:
        print(f"EVAL (B→A→B, ≤{max_steps} steps): {rate * 100:5.2f}%  "
              f"({successes}/{trials})   avg-steps-on-success={avg_steps:.2f}   "
              f"collisions={collisions}")
    return EvalSummary(success_rate=rate, avg_steps_on_success=avg_steps,
                       collisions=collisions, n_trials=trials)


def evaluate_detailed(agent: DQNAgent, env_cfg: EnvConfig,
                      n_trials: int = 20, max_steps: int = 25,
                      seed: int = 2024,
                      central_clock: bool = True) -> List[TrialResult]:
    """Per-trial diagnostic evaluation with outcome categories.

    Each trial classifies into one of:
      - ``SUCCESS``                  — focal completed B→A→B without collision.
      - ``COLLISION``                — focal collided.
      - ``PICKED_UP_NOT_DELIVERED``  — got to A but didn't make it back to B.
      - ``NEVER_REACHED_A``          — never picked up at all.

    Useful for failure-mode inspection.  Each result includes the scenario
    seed so the same trial can be deterministically replayed (e.g. for the
    visualisation module).
    """
    saved_eps = agent.eps
    agent.eps = 0.0
    rng = random.Random(seed)
    results: List[TrialResult] = []

    for trial in range(n_trials):
        scen_seed = rng.randint(0, 10**9)
        random.seed(scen_seed)
        np.random.seed(scen_seed)
        env = GridWorld(env_cfg)
        env.pos[0] = env.B
        env.carry[0] = 0
        env.start_pos[0] = env.B
        env.deliveries[0] = 0

        A_loc, B_loc = env.A, env.B
        optimal = 2 * (abs(A_loc[0] - B_loc[0]) + abs(A_loc[1] - B_loc[1]))

        picked_up_at = delivered_at = collided_at = None
        steps_taken = 0

        for step in range(max_steps):
            order = (list(range(env.N)) if central_clock
                     else random.sample(range(env.N), env.N))
            for idx in order:
                s = env.state(idx)
                a = agent.select_action(s, greedy=True)
                _, c = env.step_one(idx, a)
                if idx == 0:
                    if c and collided_at is None:
                        collided_at = step + 1
                    if env.carry[0] == 1 and picked_up_at is None:
                        picked_up_at = step + 1
                    if env.deliveries[0] >= 1 and delivered_at is None:
                        delivered_at = step + 1
            steps_taken = step + 1
            if env.deliveries[0] >= 1 or collided_at is not None:
                break

        if delivered_at is not None and collided_at is None:
            outcome = "SUCCESS"
        elif collided_at is not None:
            outcome = "COLLISION"
        elif picked_up_at is not None:
            outcome = "PICKED_UP_NOT_DELIVERED"
        else:
            outcome = "NEVER_REACHED_A"

        results.append(TrialResult(
            trial=trial, scenario_seed=scen_seed,
            A=A_loc, B=B_loc, optimal_steps=optimal,
            picked_up_at_step=picked_up_at,
            delivered_at_step=delivered_at,
            collided_at_step=collided_at,
            steps_taken=steps_taken, outcome=outcome,
        ))

    agent.eps = saved_eps
    return results
