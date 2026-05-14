"""Unit tests for the GridWorld environment.

Covers the rules that are easy to get subtly wrong:
  - Head-on collision detection between opposite-role agents.
  - Same-direction agents must be able to share a cell.
  - Collisions on A and B themselves are not counted.
  - Pickup at A and drop-off at B flip the carry flag correctly.
  - Movement is clipped to the grid (agents bounce off walls).
  - The observation vector has the expected shape and bounded values.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.env import ACTIONS, EnvConfig, GridWorld, STATE_DIM


# ----- helpers -------------------------------------------------------------

def _make_env(n_agents: int = 4) -> GridWorld:
    """Build a deterministic 5x5 env with fixed A and B."""
    random.seed(0)
    np.random.seed(0)
    cfg = EnvConfig(n_agents=n_agents)
    env = GridWorld(cfg)
    # Force a known layout for predictable tests.
    env.A = (0, 0)
    env.B = (4, 4)
    return env


# ----- shape and state-vector tests ----------------------------------------

def test_state_dim_matches_constant() -> None:
    env = _make_env()
    s = env.state(0)
    assert s.shape == (STATE_DIM,)
    assert s.dtype == np.float32


def test_state_values_are_bounded() -> None:
    """Normalised distances live in [-1, 1] for any grid layout."""
    env = _make_env()
    for i in range(env.N):
        s = env.state(i)
        # First 4 entries are dxA, dyA, dxB, dyB; bounded by grid normalisation.
        assert (s[:4] >= -1.0).all()
        assert (s[:4] <= 1.0).all()
        # Carry flag is exactly 0 or 1.
        assert s[4] in (0.0, 1.0)
        # Sensor bits are exactly 0 or 1.
        for b in s[5:]:
            assert b in (0.0, 1.0)


# ----- collision rules -----------------------------------------------------

def test_opposite_role_collision_is_counted() -> None:
    """Carrying agent moving onto empty agent's cell → collision."""
    env = _make_env(n_agents=2)
    # agent 0: carrying, at (2, 2); agent 1: empty, at (2, 3).
    env.pos = [(2, 2), (2, 3)]
    env.carry = [1, 0]
    env.collisions = 0
    # Action 2 is East (+col) for agent 0 — moves onto agent 1.
    east = ACTIONS.index((0, 1))
    _, collided = env.step_one(0, east)
    assert collided is True
    assert env.collisions == 1
    # Position must not have changed on collision.
    assert env.pos[0] == (2, 2)


def test_same_role_agents_can_share_cell() -> None:
    """Two carrying agents on the same cell should NOT collide."""
    env = _make_env(n_agents=2)
    env.pos = [(2, 2), (2, 3)]
    env.carry = [1, 1]  # both carrying — same role
    env.collisions = 0
    east = ACTIONS.index((0, 1))
    _, collided = env.step_one(0, east)
    assert collided is False
    assert env.collisions == 0
    assert env.pos[0] == (2, 3)  # actually moved


def test_collision_on_cell_A_is_disregarded() -> None:
    """Spec: 'Collisions in locations A and B are disregarded.'"""
    env = _make_env(n_agents=2)
    env.A = (1, 1)
    env.B = (4, 4)
    # agent 0 (empty, going to A) at (1, 2); agent 1 (carrying, going to B) at (1, 1)=A.
    env.pos = [(1, 2), (1, 1)]
    env.carry = [0, 1]
    env.collisions = 0
    west = ACTIONS.index((0, -1))
    _, collided = env.step_one(0, west)  # moves onto A, where agent 1 is
    assert collided is False
    assert env.collisions == 0


def test_collision_on_cell_B_is_disregarded() -> None:
    env = _make_env(n_agents=2)
    env.A = (0, 0)
    env.B = (3, 3)
    # agent 0 carrying, just left of B; agent 1 empty, sitting on B.
    env.pos = [(3, 2), (3, 3)]
    env.carry = [1, 0]
    env.collisions = 0
    east = ACTIONS.index((0, 1))
    _, collided = env.step_one(0, east)
    assert collided is False
    assert env.collisions == 0


# ----- pickup / drop-off ---------------------------------------------------

def test_pickup_flips_carry_flag() -> None:
    env = _make_env(n_agents=1)
    env.A = (2, 2)
    env.B = (4, 4)
    env.pos = [(2, 3)]
    env.carry = [0]  # empty, target = A
    west = ACTIONS.index((0, -1))
    _, collided = env.step_one(0, west)
    assert collided is False
    assert env.pos[0] == (2, 2)
    assert env.carry[0] == 1  # now carrying


def test_dropoff_flips_carry_flag() -> None:
    env = _make_env(n_agents=1)
    env.A = (0, 0)
    env.B = (2, 2)
    env.pos = [(2, 1)]
    env.carry = [1]  # carrying, target = B
    east = ACTIONS.index((0, 1))
    _, collided = env.step_one(0, east)
    assert collided is False
    assert env.pos[0] == (2, 2)
    assert env.carry[0] == 0  # delivered


# ----- bounds --------------------------------------------------------------

def test_movement_is_clipped_to_grid() -> None:
    """Moving north from row 0 should keep the agent on row 0."""
    env = _make_env(n_agents=1)
    env.pos = [(0, 2)]
    env.carry = [1]
    north = ACTIONS.index((-1, 0))
    _, collided = env.step_one(0, north)
    assert collided is False
    assert env.pos[0] == (0, 2)  # didn't move off grid


# ----- layout reset --------------------------------------------------------

def test_reset_layout_places_A_and_B_in_different_cells() -> None:
    cfg = EnvConfig()
    for s in range(20):
        random.seed(s)
        env = GridWorld(cfg)
        assert env.A != env.B


def test_reset_layout_places_agents_at_A_or_B() -> None:
    cfg = EnvConfig(n_agents=4)
    random.seed(0)
    env = GridWorld(cfg)
    for i in range(env.N):
        if env.carry[i] == 1:
            assert env.pos[i] == env.A
        else:
            assert env.pos[i] == env.B


# ----- sensor --------------------------------------------------------------

def test_sensor_fires_only_for_opposite_role_neighbour() -> None:
    """The opposite-direction sensor must light up only when an adjacent
    cell contains an agent moving the opposite way."""
    env = _make_env(n_agents=2)
    env.pos = [(2, 2), (2, 3)]
    env.carry = [1, 0]  # opposite roles, adjacent
    s = env.state(0)
    # Sensor entries are positions 5..9 in N/S/E/W order.
    east_idx = 5 + ACTIONS.index((0, 1))
    assert s[east_idx] == 1.0

    env.carry = [1, 1]  # same role now → sensor should be silent
    s = env.state(0)
    assert s[east_idx] == 0.0
