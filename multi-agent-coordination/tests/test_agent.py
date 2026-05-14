"""Unit tests for the DQN agent.

Covers:
  - Network output shape.
  - ε-greedy schedule moves toward eps_end.
  - Replay store and gradient update don't raise.
  - Save → load round-trip preserves Q values.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from src.agent import AgentConfig, DQNAgent


def test_select_action_returns_valid_index() -> None:
    agent = DQNAgent(AgentConfig())
    agent.eps = 0.0
    s = np.zeros(agent.cfg.state_dim, dtype=np.float32)
    for _ in range(20):
        a = agent.select_action(s)
        assert 0 <= a < agent.action_dim


def test_epsilon_schedule_decays() -> None:
    cfg = AgentConfig(eps_start=1.0, eps_end=0.05, eps_decay_steps=100)
    agent = DQNAgent(cfg)
    assert agent.eps == 1.0
    agent.update_epsilon(50)
    assert 0.5 < agent.eps < 0.6
    agent.update_epsilon(100)
    assert abs(agent.eps - 0.05) < 1e-6
    agent.update_epsilon(10000)  # never goes below eps_end
    assert abs(agent.eps - 0.05) < 1e-6


def test_train_step_does_not_raise() -> None:
    cfg = AgentConfig(batch_size=4, learn_after=4, train_every=1,
                      buffer_size=100)
    agent = DQNAgent(cfg)
    dim = cfg.state_dim
    for _ in range(50):
        s = np.random.rand(dim).astype(np.float32)
        s2 = np.random.rand(dim).astype(np.float32)
        agent.store(s, 0, 1.0, s2, False)
        agent.train_step()
    # No exception → pass.


def test_save_load_round_trip() -> None:
    agent = DQNAgent(AgentConfig())
    s = np.random.rand(agent.cfg.state_dim).astype(np.float32)
    with torch.no_grad():
        t = torch.from_numpy(s).unsqueeze(0).to(agent.device)
        q_before = agent.q_net(t).cpu().numpy()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pt")
        agent.save(path)

        agent2 = DQNAgent(AgentConfig())
        agent2.load(path)
        with torch.no_grad():
            t2 = torch.from_numpy(s).unsqueeze(0).to(agent2.device)
            q_after = agent2.q_net(t2).cpu().numpy()

    np.testing.assert_allclose(q_before, q_after, rtol=0, atol=1e-7)
