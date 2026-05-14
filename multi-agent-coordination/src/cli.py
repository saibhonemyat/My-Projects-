"""Command-line entry points: train, evaluate, demo.

Usage:
    python -m src.cli train  --config configs/default.yaml --output models/best.pt
    python -m src.cli eval   --checkpoint models/best.pt
    python -m src.cli demo   --checkpoint models/best.pt --output assets/rollout.gif
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
from matplotlib.animation import PillowWriter

from .agent import AgentConfig, DQNAgent
from .env import EnvConfig
from .evaluate import evaluate_round_trip
from .train import TrainConfig, train
from .visualize import animate_rollout, roll_out


def _load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_configs(cfg: Dict[str, Any]
                   ) -> tuple[EnvConfig, AgentConfig, TrainConfig]:
    env_cfg = EnvConfig(**cfg["env"],
                        use_opposite_sensor=cfg["options"]["opposite_sensor"])
    agent_cfg = AgentConfig(**cfg["agent"])
    train_cfg = TrainConfig(
        total_step_cap=cfg["training"]["total_step_cap"],
        collision_cap=cfg["training"]["collision_cap"],
        walltime_cap=cfg["training"]["walltime_cap"],
        layout_reset_every=cfg["training"]["layout_reset_every"],
        mini_eval_every=cfg["training"]["mini_eval_every"],
        early_stop_rate=cfg["training"]["early_stop_rate"],
        seed=cfg["training"]["seed"],
        central_clock=cfg["options"]["central_clock"],
        phases=cfg["phases"] if cfg["options"]["staged_training"] else [
            {"name": "single", "n_agents": cfg["env"]["n_agents"],
             "budget": cfg["training"]["total_step_cap"],
             "collision_penalty": -10.0, "near_opposite_penalty": -0.2,
             "progress_coef": 0.5, "eps_target": 0.05,
             "eps_decay_steps": 200_000}],
    )
    return env_cfg, agent_cfg, train_cfg


def cmd_train(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    env_cfg, agent_cfg, train_cfg = _build_configs(cfg)

    agent, history, total_collisions = train(env_cfg, agent_cfg, train_cfg)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.output)
    print(f"saved checkpoint: {args.output}")
    print(f"training collisions: {total_collisions}")
    print(f"final step: {history['step'][-1] if history['step'] else 0:,}")

    eval_cfg = cfg["evaluation"]
    env_cfg_eval = EnvConfig(**cfg["env"],
                             use_opposite_sensor=cfg["options"]["opposite_sensor"])
    print("\n=== Quick post-training eval ===")
    evaluate_round_trip(agent, env_cfg_eval,
                        trials=eval_cfg["trials"],
                        max_steps=eval_cfg["max_steps_perf"],
                        central_clock=cfg["options"]["central_clock"])
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    env_cfg, agent_cfg, _ = _build_configs(cfg)
    agent = DQNAgent(agent_cfg)
    agent.load(args.checkpoint)
    print(f"loaded checkpoint: {args.checkpoint}")

    eval_cfg = cfg["evaluation"]
    print(f"\n=== Spec minimum-grade horizon (≤ {eval_cfg['max_steps_min']} steps) ===")
    summary_25 = evaluate_round_trip(agent, env_cfg,
                                     trials=eval_cfg["trials"],
                                     max_steps=eval_cfg["max_steps_min"],
                                     central_clock=cfg["options"]["central_clock"])

    print(f"\n=== Performance-point horizon (≤ {eval_cfg['max_steps_perf']} steps) ===")
    summary_20 = evaluate_round_trip(agent, env_cfg,
                                     trials=eval_cfg["trials"],
                                     max_steps=eval_cfg["max_steps_perf"],
                                     central_clock=cfg["options"]["central_clock"])

    print()
    print(f"≥75% at ≤{eval_cfg['max_steps_min']} steps: "
          f"{'PASS' if summary_25.success_rate >= 0.75 else 'FAIL'} "
          f"({summary_25.success_rate * 100:.2f}%)")
    print(f"≥85% at ≤{eval_cfg['max_steps_perf']} steps: "
          f"{'PASS' if summary_20.success_rate >= 0.85 else 'FAIL'} "
          f"({summary_20.success_rate * 100:.2f}%)")
    print(f"≥95% at ≤{eval_cfg['max_steps_perf']} steps: "
          f"{'PASS' if summary_20.success_rate >= 0.95 else 'FAIL'} "
          f"({summary_20.success_rate * 100:.2f}%)")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    env_cfg, agent_cfg, _ = _build_configs(cfg)
    agent = DQNAgent(agent_cfg)
    agent.load(args.checkpoint)
    frames = roll_out(agent, env_cfg, max_steps=25, seed=args.seed,
                      central_clock=cfg["options"]["central_clock"])
    anim = animate_rollout(frames, grid=cfg["env"]["grid"],
                           title=f"Trained agents (seed={args.seed})", fps=3)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.output, writer=PillowWriter(fps=3))
    print(f"saved rollout: {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="multi-agent-coord")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--output", default="models/best.pt")
    p_train.add_argument("--seed", type=int, default=None)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--checkpoint", default="models/best.pt")
    p_eval.set_defaults(func=cmd_eval)

    p_demo = sub.add_parser("demo")
    p_demo.add_argument("--checkpoint", default="models/best.pt")
    p_demo.add_argument("--output", default="assets/rollout.gif")
    p_demo.add_argument("--seed", type=int, default=7)
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
