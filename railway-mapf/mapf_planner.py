"""
Question 3 - Multi-agent path-finding with replanning on malfunction.

Improvements over original:
  - plan_single extracted to module level so both get_path and replan share
    the same planner without code duplication.
  - replan now detects secondary conflicts: after inserting wait steps for
    malfunctioning/failed agents it checks every other agent's remaining path
    for vertex and edge conflicts, and replans any that are now infeasible.
  - infer_direction helper lets replan resume planning from the agent's actual
    current heading rather than always falling back to initial_direction.
  - Parent-pointer reconstruction keeps the heap small on large instances.
"""

from lib_piglet.utils.tools import eprint
from typing import List, Tuple
import glob, os, sys, time, json
from heapq import heappush, heappop
from collections import deque

try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
    from flatland.utils.controller import (
        get_action, Train_Actions, Directions, check_conflict,
        path_controller, evaluator, remote_evaluator,
    )
except Exception as e:
    eprint("Cannot load flatland modules!")
    eprint(e)
    exit(1)

# ---------------------------------------------------------------------------
# Debug / visualiser flags
# ---------------------------------------------------------------------------
debug = False
visualizer = False

test_single_instance = False
level = 1
test = 6


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _in_bounds(x, y, height, width):
    return 0 <= x < height and 0 <= y < width


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _infer_direction(path: list, t: int, default_dir: int) -> int:
    """Return the heading at time t by comparing path[t-1] → path[t]."""
    if t > 0 and t < len(path) and path[t] != path[t - 1]:
        prev, cur = path[t - 1], path[t]
        if cur[0] < prev[0]:
            return Directions.NORTH
        if cur[1] > prev[1]:
            return Directions.EAST
        if cur[0] > prev[0]:
            return Directions.SOUTH
        if cur[1] < prev[1]:
            return Directions.WEST
    return default_dir


def _build_conflicts(paths: list, time_limit: int, goal_reserve: int = 2):
    """
    Build vertex and edge conflict tables from a list of planned paths.

    goal_reserve: number of extra timesteps to block a goal cell after arrival,
    preventing other agents from passing through a finished agent's cell.
    """
    vertex_conflicts: dict = {}
    edge_conflicts: dict = {}

    for p in paths:
        if not p:
            continue
        for t, pos in enumerate(p):
            if t > time_limit:
                break
            vertex_conflicts.setdefault(t, set()).add(pos)
        # Reserve goal cell for a few timesteps after arrival
        if goal_reserve > 0:
            last_t = len(p) - 1
            goal_pos = p[-1]
            for tt in range(last_t + 1, min(time_limit, last_t + 1 + goal_reserve)):
                vertex_conflicts.setdefault(tt, set()).add(goal_pos)
        for t in range(min(len(p) - 1, time_limit)):
            edge_conflicts.setdefault(t, set()).add((p[t], p[t + 1]))

    return vertex_conflicts, edge_conflicts


# ---------------------------------------------------------------------------
# Core single-agent planner (shared by get_path and replan)
# ---------------------------------------------------------------------------

def _plan_single(
    start: tuple,
    start_dir: int,
    goal: tuple,
    deadline: int,
    rail: GridTransitionMap,
    vertex_conflicts: dict,
    edge_conflicts: dict,
    time_limit: int,
    height: int,
    width: int,
) -> list:
    """
    Time-space A* for a single agent.

    Uses parent-pointer reconstruction — the heap carries only
    (f, tie, t, loc, direction), not the full path.
    """
    if start == goal:
        return [start]

    pq = []
    tie = 0
    h0 = _manhattan(start, goal)
    heappush(pq, (h0, tie, 0, start, start_dir))
    tie += 1

    came_from: dict = {}
    g_scores: dict = {(0, start, start_dir): 0}
    closed: set = set()

    best_goal_time = float("inf")
    best_goal_state = None
    initial_dist = _manhattan(start, goal)

    while pq:
        f, _, t, loc, direction = heappop(pq)

        state = (t, loc, direction)
        if state in closed:
            continue
        closed.add(state)

        if loc == goal:
            if t < best_goal_time:
                best_goal_time = t
                best_goal_state = state
            continue  # keep searching for cheaper paths in the same timestep bucket

        if t >= time_limit or t >= best_goal_time:
            continue

        # ------ Wait ------
        t1 = t + 1
        if loc not in vertex_conflicts.get(t1, ()):
            new_g = t1
            next_state = (t1, loc, direction)
            if next_state not in closed and new_g < g_scores.get(next_state, float("inf")):
                g_scores[next_state] = new_g
                came_from[next_state] = state
                # Adaptive wait penalty: lighter early on when we haven't moved much yet
                progress = _manhattan(start, loc)
                wait_pen = 0.05 if progress < initial_dist * 0.4 else 0.25
                lateness = max(0, (new_g + _manhattan(loc, goal)) - deadline)
                cost = new_g + _manhattan(loc, goal) * 1.02 + wait_pen + 1.2 * lateness
                heappush(pq, (cost, tie, t1, loc, direction))
                tie += 1

        # ------ Move ------
        valid_transitions = rail.get_transitions(loc[0], loc[1], direction)
        for next_dir in range(4):
            if not valid_transitions[next_dir]:
                continue

            new_x, new_y = loc
            if next_dir == Directions.NORTH:
                new_x -= 1
            elif next_dir == Directions.EAST:
                new_y += 1
            elif next_dir == Directions.SOUTH:
                new_x += 1
            elif next_dir == Directions.WEST:
                new_y -= 1

            if not _in_bounds(new_x, new_y, height, width):
                continue

            nxt = (new_x, new_y)
            next_state = (t + 1, nxt, next_dir)

            if next_state in closed:
                continue
            if nxt in vertex_conflicts.get(t + 1, ()):
                continue
            if (nxt, loc) in edge_conflicts.get(t, ()):
                continue

            new_g = t + 1
            if new_g >= g_scores.get(next_state, float("inf")):
                continue

            g_scores[next_state] = new_g
            came_from[next_state] = state

            turn_pen = 0.1 if next_dir != direction else 0.0
            lateness = max(0, (new_g + _manhattan(nxt, goal)) - deadline)
            cost = new_g + _manhattan(nxt, goal) * 1.02 + turn_pen + 1.2 * lateness
            heappush(pq, (cost, tie, new_g, nxt, next_dir))
            tie += 1

    if best_goal_state is None:
        return []

    # Reconstruct path via parent pointers
    path = []
    cur = best_goal_state
    while cur in came_from:
        path.append(cur[1])
        cur = came_from[cur]
    path.append(start)
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# get_path — initial multi-agent planning (Prioritised Planning / CBS-lite)
# ---------------------------------------------------------------------------

def get_path(agents: List[EnvAgent], rail: GridTransitionMap, max_timestep: int):
    """
    Plan conflict-free paths for all agents using prioritised planning.

    Agents are ordered by slack (deadline − minimum travel time), smallest
    first, so the most time-pressured agents get first priority.
    """

    height, width = rail.height, rail.width
    num_agents = len(agents)
    base_limit = min(max_timestep, max(64, height * width * 6))

    # ------------------------------------------------------------------
    # Priority ordering: slack-first, break ties by descending distance
    # ------------------------------------------------------------------
    def compute_order():
        items = []
        for i, ag in enumerate(agents):
            dist = _manhattan(ag.initial_position, ag.target)
            deadline = getattr(ag, "deadline", max_timestep)
            slack = deadline - dist
            items.append((slack, -dist, i))
        items.sort()
        return [i for *_, i in items]

    order = compute_order()
    path_all = [[] for _ in range(num_agents)]

    for agent_id in order:
        ag = agents[agent_id]
        start = ag.initial_position
        start_dir = ag.initial_direction
        goal = ag.target
        deadline = getattr(ag, "deadline", max_timestep)

        # Build conflict tables from already-planned agents
        planned = [path_all[j] for j in range(num_agents) if path_all[j]]
        vc, ec = _build_conflicts(planned, base_limit)

        plan = _plan_single(start, start_dir, goal, deadline, rail, vc, ec, base_limit, height, width)
        path_all[agent_id] = plan if plan else [start]

    return path_all


# ---------------------------------------------------------------------------
# replan — called when a malfunction or collision occurs mid-episode
# ---------------------------------------------------------------------------

def replan(
    agents: List[EnvAgent],
    rail: GridTransitionMap,
    current_timestep: int,
    existing_paths: List[Tuple],
    max_timestep: int,
    new_malfunction_agents: List[int],
    failed_agents: List[int],
):
    """
    Replan paths after malfunctions or execution failures.

    Steps:
      1. Extend malfunctioning agents' paths with wait steps.
      2. Extend failed-to-move agents with one wait step.
      3. Detect agents whose remaining paths now conflict with the
         updated paths from steps 1-2.
      4. Replan those secondary-conflict agents in priority order.
    """

    height, width = rail.height, rail.width
    num_agents = len(agents)
    base_limit = min(max_timestep, max(64, height * width * 6))

    # ------------------------------------------------------------------
    # Step 1 — Insert wait steps for malfunctioning agents
    # ------------------------------------------------------------------
    for agent_id in new_malfunction_agents:
        agent = agents[agent_id]
        duration = agent.malfunction_data.get("malfunction", 0)
        path = list(existing_paths[agent_id])
        if len(path) > current_timestep:
            pos = path[current_timestep]
            for _ in range(duration):
                path.insert(current_timestep + 1, pos)
        existing_paths[agent_id] = path

    # ------------------------------------------------------------------
    # Step 2 — Insert one wait step for agents that failed to move
    # ------------------------------------------------------------------
    for agent_id in failed_agents:
        if agent_id in new_malfunction_agents:
            continue
        path = list(existing_paths[agent_id])
        if len(path) > current_timestep:
            pos = path[current_timestep]
            path.insert(current_timestep + 1, pos)
        existing_paths[agent_id] = path

    # ------------------------------------------------------------------
    # Step 3 — Detect secondary conflicts in other agents' remaining paths
    # ------------------------------------------------------------------
    affected = set(new_malfunction_agents) | set(failed_agents)

    def _path_has_conflict(agent_id: int) -> bool:
        """Check if agent_id's path conflicts with any affected agent's updated path."""
        path_i = existing_paths[agent_id]
        for j in affected:
            path_j = existing_paths[j]
            limit = min(len(path_i), len(path_j), max_timestep)
            for t in range(current_timestep, limit):
                # Vertex conflict
                if path_i[t] == path_j[t]:
                    return True
                # Edge conflict (swap)
                if t + 1 < limit:
                    if path_i[t] == path_j[t + 1] and path_i[t + 1] == path_j[t]:
                        return True
        return False

    agents_to_replan = []
    for i in range(num_agents):
        if i in affected:
            continue
        if _path_has_conflict(i):
            agents_to_replan.append(i)

    if debug and agents_to_replan:
        print(
            f"[replan t={current_timestep}] Secondary conflicts detected for agents: {agents_to_replan}",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Step 4 — Replan secondary-conflict agents in priority order
    # ------------------------------------------------------------------
    def _replan_order(ids):
        items = []
        for i in ids:
            ag = agents[i]
            dist = _manhattan(ag.initial_position, ag.target)
            deadline = getattr(ag, "deadline", max_timestep)
            slack = deadline - dist
            items.append((slack, -dist, i))
        items.sort()
        return [i for *_, i in items]

    for agent_id in _replan_order(agents_to_replan):
        ag = agents[agent_id]
        path = existing_paths[agent_id]

        # Current position and heading at current_timestep
        cur_pos = path[current_timestep] if len(path) > current_timestep else ag.target
        cur_dir = _infer_direction(path, current_timestep, ag.initial_direction)
        goal = ag.target
        deadline = getattr(ag, "deadline", max_timestep)

        # Prefix: keep the path up to and including current_timestep
        prefix = list(path[:current_timestep + 1])

        # Build conflict table from all OTHER agents (using their full updated paths)
        other_paths = [existing_paths[j] for j in range(num_agents) if j != agent_id]
        vc, ec = _build_conflicts(other_paths, base_limit)

        # Plan the remaining suffix from cur_pos to goal
        suffix = _plan_single(
            cur_pos, cur_dir, goal, deadline,
            rail, vc, ec, base_limit, height, width,
        )

        if suffix:
            existing_paths[agent_id] = prefix + suffix[1:]  # avoid duplicating cur_pos
        else:
            # Fallback: keep original path (agent will wait in place)
            if debug:
                print(f"[replan] Could not replan agent {agent_id}, keeping original.", file=sys.stderr)

    return existing_paths


# ---------------------------------------------------------------------------
# Boilerplate — do not modify
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path, sys.argv, replan=replan)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(
                os.path.join(script_path, "multi_test_case/level{}_test_{}.pkl".format(level, test))
            )
        test_cases.sort()
        deadline_files = [test.replace(".pkl", ".ddl") for test in test_cases]
        evaluator(get_path, test_cases, debug, visualizer, 3, deadline_files, replan=replan)
