"""
Question 2 - Single agent path-finding with conflict avoidance.

Improvements over original:
  - Parent-pointer reconstruction replaces storing full path in the heap,
    significantly reducing memory usage on large grids.
  - Proper A* closed set: a state is sealed after it is popped, eliminating
    redundant re-expansion that the original visited/g_scores hybrid allowed.
  - Removed the path[-2] anti-backtracking guard which could incorrectly
    prune valid routes in constrained rail layouts.
  - Simplified wait-penalty logic (flat 0.3) for cleaner cost function.
"""

from lib_piglet.utils.tools import eprint
import glob, os, sys
from heapq import heappush, heappop

try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.utils.controller import (
        get_action, Train_Actions, Directions, check_conflict,
        path_controller, evaluator, remote_evaluator,
    )
except Exception as e:
    eprint("Cannot load flatland modules!", e)
    exit(1)

# ---------------------------------------------------------------------------
# Debug / visualiser flags
# ---------------------------------------------------------------------------
debug = False
visualizer = False

test_single_instance = False
level = 1
test = 5


# ---------------------------------------------------------------------------
# get_path
# ---------------------------------------------------------------------------

def get_path(
    start: tuple,
    start_direction: int,
    goal: tuple,
    rail: GridTransitionMap,
    agent_id: int,
    existing_paths: list,
    max_timestep: int,
):
    """
    Time-space A* with conflict avoidance.

    Returns a list of (x, y) tuples representing the agent's plan.
    """

    if start == goal:
        return [start]

    height, width = rail.height, rail.width
    max_search_time = max(max_timestep, height * width * 4)

    # ------------------------------------------------------------------
    # Pre-compute conflict tables from already-planned agents.
    # vertex_conflicts[t]  -> set of positions occupied at time t
    # edge_conflicts[t]    -> set of (pos_a, pos_b) edges traversed at time t
    # ------------------------------------------------------------------
    vertex_conflicts: dict = {}
    edge_conflicts: dict = {}

    for p in (existing_paths or []):
        L = len(p)
        for t in range(L):
            vertex_conflicts.setdefault(t, set()).add(p[t])
        for t in range(L - 1):
            edge_conflicts.setdefault(t, set()).add((p[t], p[t + 1]))

    def in_bounds(x, y):
        return 0 <= x < height and 0 <= y < width

    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ------------------------------------------------------------------
    # A* — heap entries: (f, tie, t, loc, direction)
    # Paths are reconstructed via came_from, NOT stored in the heap.
    # ------------------------------------------------------------------
    pq = []
    tie = 0
    heappush(pq, (manhattan(start, goal), tie, 0, start, start_direction))
    tie += 1

    # came_from[(t, loc, dir)] = parent state (t-1, prev_loc, prev_dir)
    came_from: dict = {}

    # Best known g-score for each state
    g_scores: dict = {(0, start, start_direction): 0}

    # Closed set — states we have already expanded (no need to revisit)
    closed: set = set()

    while pq:
        f, _, t, loc, direction = heappop(pq)

        state = (t, loc, direction)

        # Skip if already expanded (a cheaper path was found earlier)
        if state in closed:
            continue
        closed.add(state)

        # ------ Goal reached — reconstruct and return path ------
        if loc == goal:
            path = []
            cur = state
            while cur in came_from:
                path.append(cur[1])
                cur = came_from[cur]
            path.append(start)
            path.reverse()
            return path

        if t >= max_search_time:
            continue

        # ------ Expand moves ------
        valid_transitions = rail.get_transitions(loc[0], loc[1], direction)

        # Prefer straight, then right, then left, then reverse
        dir_order = [direction, (direction + 1) % 4, (direction + 3) % 4, (direction + 2) % 4]

        for next_dir in dir_order:
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

            if not in_bounds(new_x, new_y):
                continue

            next_loc = (new_x, new_y)
            next_state = (t + 1, next_loc, next_dir)

            if next_state in closed:
                continue

            # Vertex conflict — another agent occupies next_loc at t+1
            if next_loc in vertex_conflicts.get(t + 1, ()):
                continue

            # Edge conflict — swapping positions with another agent
            if (next_loc, loc) in edge_conflicts.get(t, ()):
                continue

            new_g = t + 1
            if new_g >= g_scores.get(next_state, float("inf")):
                continue  # Already found an equally good or better path

            g_scores[next_state] = new_g
            came_from[next_state] = state

            turn_pen = 0.1 if next_dir != direction else 0.0
            new_f = new_g + manhattan(next_loc, goal) + turn_pen
            heappush(pq, (new_f, tie, t + 1, next_loc, next_dir))
            tie += 1

        # ------ Expand wait ------
        next_state_wait = (t + 1, loc, direction)
        if next_state_wait not in closed:
            if loc not in vertex_conflicts.get(t + 1, ()):
                new_g = t + 1
                if new_g < g_scores.get(next_state_wait, float("inf")):
                    g_scores[next_state_wait] = new_g
                    came_from[next_state_wait] = state

                    wait_pen = 0.3
                    new_f = new_g + manhattan(loc, goal) + wait_pen
                    heappush(pq, (new_f, tie, t + 1, loc, direction))
                    tie += 1

    # No path found within time limit
    return []


# ---------------------------------------------------------------------------
# Boilerplate — do not modify
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path, sys.argv)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(
                os.path.join(script_path, "multi_test_case/level{}_test_{}.pkl".format(level, test))
            )
        test_cases.sort()
        evaluator(get_path, test_cases, debug, visualizer, 2)
