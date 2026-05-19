# myTeam.py  —  PDDL-lite hybrid
#
# PDDL role: decide HIGH-LEVEL MODE only (ATTACK / RETURN / DEFEND / HUNT)
#            called ONLY when game state changes significantly (~5-15x per game)
#
# Python role: everything tactical
#              food target scoring, A* pathfinding, ghost avoidance,
#              cut-off intercept, scared ghost hunting, oscillation escape
#
# Why: PDDL solver is fine when called rarely.
#      Running it every tick costs 50-200ms per move — way too slow.

from captureAgents import CaptureAgent
from game import Directions, Actions
from util import nearestPoint, PriorityQueue
import random, os
from collections import deque, Counter
from typing import Tuple, Dict, Optional, Set

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

from lib_piglet.utils.pddl_solver import pddl_solver

SCARED_MIN = 4   # scaredTimer threshold to treat ghost as huntable

def _action_to_mode(name: str) -> str:
    """Map PDDL action name → Python mode string."""
    n = name.lower()
    if 'hunt'    in n:                                              return 'HUNT'
    if 'return'  in n or 'urgent' in n:                            return 'RETURN'
    if any(x in n for x in ('intercept','patrol','defence','defend','support')): return 'DEFEND'
    if 'capsule' in n:                                             return 'CAPSULE'
    return 'ATTACK'


def createTeam(firstIndex, secondIndex, isRed,
               first='SmartAgent', second='SmartAgent'):
    return [SmartAgent(firstIndex), SmartAgent(secondIndex)]


class _Shared:
    """Class-level dict so both agents read each other's state."""
    info: Dict[int, Dict] = {}


class SmartAgent(CaptureAgent):

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self._solver = pddl_solver(os.path.join(BASE_FOLDER, 'myTeam.pddl'))

        self.walls    = gameState.getWalls()
        self.red      = gameState.isOnRedTeam(self.index)
        mid           = self.walls.width // 2
        self.home_x   = (mid - 1) if self.red else mid
        self.start    = gameState.getAgentPosition(self.index)
        team          = self.getTeam(gameState)
        self.ally_idx = [i for i in team if i != self.index][0]

        self.home_boundary = [
            (self.home_x, y) for y in range(self.walls.height)
            if not self.walls[self.home_x][y]
        ] or [self._ipos(self.start)]

        other_x = self.home_x + 1 if self.red else self.home_x - 1
        self.entrances = [
            (self.home_x, y) for y in range(self.walls.height)
            if not self.walls[self.home_x][y]
               and 0 <= other_x < self.walls.width
               and not self.walls[other_x][y]
        ] or self.home_boundary

        self._compute_topology()

        cy = self.walls.height // 2
        self.patrol_pts = sorted(self.entrances,
                                 key=lambda p: abs(p[1] - cy))[:6] or self.home_boundary[:6]
        self.patrol_idx = 0 if self.index == min(team) else len(self.patrol_pts) // 2

        self._pos_hist       = deque(maxlen=8)
        self._last_pos       = self._ipos(self.start)
        self._stuck          = 0
        self._ghost_prev     : Dict[int, Tuple] = {}
        self._def_food_prev  = self._food_set(self.getFoodYouAreDefending(gameState))
        self._last_eaten_pos = None
        self._last_eaten_tick = -999

        # PDDL result cache
        self._pddl_mode        = 'ATTACK'
        self._pddl_fingerprint = None

        self.tick        = 0
        self.is_attacker = True
        _Shared.info[self.index] = {
            'pos': self._ipos(self.start), 'carrying': 0,
            'mode': 'ATTACK', 'is_pacman': False,
        }

    def _compute_topology(self):
        self.dead_depth: Dict[Tuple, int] = {}
        q = deque()
        for x in range(self.walls.width):
            for y in range(self.walls.height):
                if not self.walls[x][y] and sum(1 for _ in self._nb(x, y)) != 2:
                    self.dead_depth[(x, y)] = 0
                    q.append((x, y))
        while q:
            cur = q.popleft()
            for nb in self._nb(*cur):
                if nb not in self.dead_depth:
                    self.dead_depth[nb] = self.dead_depth[cur] + 1
                    q.append(nb)

    def _nb(self, x, y):
        for a in (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST):
            dx, dy = Actions.directionToVector(a)
            nx, ny = int(x + dx), int(y + dy)
            if 0 <= nx < self.walls.width and 0 <= ny < self.walls.height \
               and not self.walls[nx][ny]:
                yield (nx, ny)

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #

    def chooseAction(self, gameState):
        self.tick += 1
        self._update_memory(gameState)
        self._update_shared(gameState)
        self._assign_roles(gameState)

        my_pos = self._pos(gameState)
        legal  = [a for a in gameState.getLegalActions(self.index)
                  if a != Directions.STOP]
        if not legal:
            return Directions.STOP

        self._pos_hist.append(my_pos)
        self._stuck = self._stuck + 1 if my_pos == self._last_pos else 0
        self._last_pos = my_pos
        if self._stuck >= 3 or self._oscillating():
            self._stuck = 0
            return self._escape(gameState, legal)

        mode = self._get_pddl_mode(gameState)
        _Shared.info[self.index]['mode'] = mode

        my_state = gameState.getAgentState(self.index)
        ghost_d  = self._ghost_dist(gameState, my_pos)

        # immediate danger — override any PDDL mode
        if my_state.isPacman and ghost_d <= 2 and not self._all_ghosts_scared(gameState):
            return self._escape(gameState, legal)

        if mode == 'RETURN':
            target = self._nearest(my_pos, self.home_boundary)
            return self._astar_step(gameState, my_pos, target, avoid=True, legal=legal)

        if mode == 'HUNT':
            target = self._nearest_scared(gameState, my_pos)
            if target:
                return self._astar_step(gameState, my_pos, target, avoid=False, legal=legal)
            mode = 'ATTACK'

        if mode == 'CAPSULE':
            cap = self._nearest_capsule(gameState, my_pos)
            if cap:
                return self._astar_step(gameState, my_pos, cap, avoid=True, legal=legal)
            mode = 'ATTACK'

        if mode == 'ATTACK':
            target = self._best_food(gameState, my_pos)
            if not target:
                target = self._nearest(my_pos, self.home_boundary)
            return self._astar_step(gameState, my_pos, target, avoid=True, legal=legal)

        if mode == 'DEFEND':
            target = self._defend_target(gameState, my_pos)
            return self._astar_step(gameState, my_pos, target, avoid=False, legal=legal)

        return random.choice(legal)

    # ------------------------------------------------------------------ #
    #  PDDL mode — cached, called only on state-change fingerprint        #
    # ------------------------------------------------------------------ #

    def _get_pddl_mode(self, gameState) -> str:
        fp = self._fingerprint(gameState)
        if fp == self._pddl_fingerprint:
            return self._pddl_mode
        self._pddl_fingerprint = fp
        self._pddl_mode = self._run_pddl(gameState)
        return self._pddl_mode

    def _fingerprint(self, gameState) -> tuple:
        """Compact state hash. PDDL only re-runs when this changes."""
        my_state   = gameState.getAgentState(self.index)
        carry      = getattr(my_state, 'numCarrying', 0)
        my_pos     = self._pos(gameState)
        ghost_d    = self._ghost_dist(gameState, my_pos)
        invader    = any(
            gameState.getAgentState(i) and gameState.getAgentState(i).isPacman
            and gameState.getAgentState(i).getPosition() is not None
            for i in self.getOpponents(gameState))
        all_scared  = self._all_ghosts_scared(gameState)
        time_left   = getattr(gameState.data, 'timeleft', 1200)
        def_left    = len(self.getFoodYouAreDefending(gameState).asList())
        return (
            my_state.isPacman,
            min(carry, 10),
            ghost_d <= 5,
            ghost_d <= 2,
            all_scared,
            invader,
            time_left <= 160,
            def_left <= 6,
            self.is_attacker,
            self._nearest_scared(gameState, my_pos) is not None,
        )

    def _run_pddl(self, gameState) -> str:
        try:
            objects, init = self._pddl_state(gameState)
            pos_goal, neg_goal = self._pddl_goals(objects, init, gameState)
            self._solver.parser_.reset_problem()
            self._solver.parser_.set_objects(objects)
            self._solver.parser_.set_state(init)
            self._solver.parser_.set_positive_goals(pos_goal)
            self._solver.parser_.set_negative_goals(neg_goal)
            plan = self._solver.solve()
            if plan:
                return _action_to_mode(plan[0][0].name)
        except Exception as e:
            print(f'[PDDL] agent {self.index}: {e}')
        return self._python_mode(gameState)

    # ------------------------------------------------------------------ #
    #  PDDL state builder                                                  #
    # ------------------------------------------------------------------ #

    def _pddl_state(self, gameState):
        states, objects = [], []
        myObj   = f'a{self.index}'
        allyObj = f'a{self.ally_idx}'

        for idx in self.getTeam(gameState):
            objects.append((f'a{idx}', 'current_agent' if idx == self.index else 'ally'))
        enemies = self.getOpponents(gameState)[:2]
        for k, eidx in enumerate(enemies, 1):
            objects.append((f'e{eidx}', f'enemy{k}'))

        my_state = gameState.getAgentState(self.index)
        my_pos   = self._pos(gameState)

        if my_state.isPacman: states.append(('is_pacman', myObj))
        carry = getattr(my_state, 'numCarrying', 0)
        if carry > 0:  states += [('food_in_backpack', myObj), ('carry_some', myObj)]
        if carry >= 3: states.append(('3_food_in_backpack', myObj))
        if carry >= 5: states.append(('5_food_in_backpack', myObj))
        if carry >= 10:states.append(('10_food_in_backpack', myObj))

        ally_info = _Shared.info.get(self.ally_idx, {})
        ally_pos  = ally_info.get('pos')
        if ally_pos and self.getMazeDistance(my_pos, ally_pos) <= 4:
            states.append(('near_ally', myObj))
        ally_mode = ally_info.get('mode', '')
        if ally_mode in ('DEFEND', 'PATROL'):
            states += [('ally_defending', allyObj), ('ally_cover', allyObj)]
        elif ally_mode == 'RETURN':
            states += [('ally_returning', allyObj), ('ally_cover', allyObj)]
        elif ally_mode in ('ATTACK', 'HUNT'):
            states.append(('ally_attacking', allyObj))

        safe_to_collect = True
        for k, eidx in enumerate(enemies, 1):
            eObj = f'e{eidx}'
            est  = gameState.getAgentState(eidx)
            if est and est.getPosition() is not None:
                epos   = self._ipos(est.getPosition())
                dist   = self.getMazeDistance(my_pos, epos)
                scared = getattr(est, 'scaredTimer', 0) > 0
                states.append(('enemy_visible', eObj))
                if dist <= 4:
                    states.append(('enemy_around', eObj, myObj))
                    if not scared: safe_to_collect = False
                if dist <= 5:
                    states.append(('enemy_short_distance', eObj, myObj))
                    if not scared: safe_to_collect = False
                elif dist <= 15: states.append(('enemy_medium_distance', eObj, myObj))
                else:            states.append(('enemy_long_distance',   eObj, myObj))
                if est.isPacman: states += [('enemy_is_pacman', eObj), ('invader_visible', eObj)]
                if scared:       states.append(('is_scared', eObj))

        if safe_to_collect: states.append(('safe_to_collect', myObj))
        if not safe_to_collect or _Shared.info[self.index].get('in_danger', False):
            states.append(('threatened', myObj))

        food_list = self.getFood(gameState).asList()
        if food_list:
            states.append(('food_available',))
            if any(self.getMazeDistance(my_pos, self._ipos(f)) <= 4 for f in food_list[:10]):
                states.append(('near_food', myObj))
        caps = list(self.getCapsules(gameState))
        if caps:
            states.append(('capsule_available',))
            if any(self.getMazeDistance(my_pos, self._ipos(c)) <= 4 for c in caps):
                states.append(('near_capsule', myObj))

        score = self.getScore(gameState)
        if score > 0:   states.append(('winning',))
        if score >= 3:  states.append(('winning_gt3',))
        if score >= 5:  states.append(('winning_gt5',))
        if score >= 10: states.append(('winning_gt10',))
        if score >= 20: states.append(('winning_gt20',))
        if getattr(gameState.data, 'timeleft', 1200) <= 160:
            states.append(('endgame',))

        return objects, states

    def _pddl_goals(self, objects, init, gameState):
        S        = set(init)
        myObj    = f'a{self.index}'
        my_state = gameState.getAgentState(self.index)
        my_pos   = self._pos(gameState)
        carry    = getattr(my_state, 'numCarrying', 0)
        time_left = getattr(gameState.data, 'timeleft', 1200)
        enemy_objs = [o[0] for o in objects if o[1] in ('enemy1', 'enemy2')]

        if time_left <= 50 and self.getScore(gameState) > 0:
            return [('defend_foods',)], []
        if len(self.getFoodYouAreDefending(gameState).asList()) <= 6 \
           or (self.tick - self._last_eaten_tick) <= 5:
            return [('defend_foods',)], []
        if not self.is_attacker:
            return [('defend_foods',)], []
        if my_state.isPacman and self._nearest_scared(gameState, my_pos):
            return [('safe_collection',)], []
        if any(s[0] == 'invader_visible' for s in S if s):
            return [('defend_foods',)], [('enemy_is_pacman', e) for e in enemy_objs]
        if ('capsule_available',) in S and ('near_capsule', myObj) in S:
            if self._ghost_dist(gameState, my_pos) <= 6:
                return [('eat_capsule',)], [('capsule_available',)]

        all_scared = self._all_ghosts_scared(gameState)
        ghost_d    = self._ghost_dist(gameState, my_pos)
        thresh     = self._return_thresh(gameState, ghost_d, all_scared)
        home_d     = self.getMazeDistance(my_pos, self._nearest(my_pos, self.home_boundary))
        if my_state.isPacman and (
            carry >= thresh
            or (carry > 0 and ghost_d <= 3 and not all_scared)
            or (carry > 0 and time_left <= home_d + 20)
        ):
            return [], [('is_pacman', myObj), ('food_in_backpack', myObj)]

        return [('safe_collection',)], []

    # ------------------------------------------------------------------ #
    #  Python mode fallback (when PDDL errors)                            #
    # ------------------------------------------------------------------ #

    def _python_mode(self, gameState) -> str:
        my_state   = gameState.getAgentState(self.index)
        my_pos     = self._pos(gameState)
        carry      = getattr(my_state, 'numCarrying', 0)
        ghost_d    = self._ghost_dist(gameState, my_pos)
        all_scared = self._all_ghosts_scared(gameState)
        time_left  = getattr(gameState.data, 'timeleft', 1200)

        if not self.is_attacker:                                          return 'DEFEND'
        if my_state.isPacman and ghost_d <= 2 and not all_scared:        return 'ESCAPE'
        if my_state.isPacman and self._nearest_scared(gameState, my_pos): return 'HUNT'

        thresh = self._return_thresh(gameState, ghost_d, all_scared)
        home_d = self.getMazeDistance(my_pos, self._nearest(my_pos, self.home_boundary))
        if my_state.isPacman and (
            carry >= thresh
            or (carry > 0 and ghost_d <= 3 and not all_scared)
            or (carry > 0 and time_left <= home_d + 20)
            or (carry > 0 and time_left <= 200)
        ):
            return 'RETURN'
        return 'ATTACK'

    # ------------------------------------------------------------------ #
    #  Return threshold                                                    #
    # ------------------------------------------------------------------ #

    def _return_thresh(self, gameState, ghost_d, all_scared) -> int:
        remaining = len(self.getFood(gameState).asList())
        lead      = self.getScore(gameState)
        base      = 7 if remaining >= 10 else (5 if remaining >= 5 else 3)
        if all_scared: return min(15, base + 5)
        if self.tick <= 150:   base = min(base, 3)
        elif self.tick <= 300: base = min(base, 5)
        if ghost_d <= 4: base = min(base, 3)
        if ghost_d <= 2: base = min(base, 1)
        if lead >= 6:    base = min(base, 4)
        if lead >= 10:   base = min(base, 3)
        return max(1, base)

    # ------------------------------------------------------------------ #
    #  Role assignment                                                     #
    # ------------------------------------------------------------------ #

    def _assign_roles(self, gameState):
        if self.getScore(gameState) >= 10 or len(self.getFood(gameState).asList()) <= 3:
            self.is_attacker = True; return

        team   = sorted(self.getTeam(gameState))
        states = [(i, gameState.getAgentState(i)) for i in team]
        attacker = None
        for i, st in states:
            if getattr(st, 'numCarrying', 0) > 0: attacker = i; break
        if attacker is None:
            for i, st in states:
                if st.isPacman: attacker = i; break
        if attacker is None:
            mid  = self.walls.width // 2
            # enemy boundary: red attacks right side (mid), blue attacks left side (mid-1)
            ex   = mid if self.red else mid - 1
            ebnd = [(ex, y) for y in range(self.walls.height)
                    if not self.walls[ex][y]] or [self._ipos(self.start)]
            # closest to enemy boundary becomes attacker
            attacker = min(team, key=lambda i: self.getMazeDistance(
                self._ipos(gameState.getAgentPosition(i)),
                self._nearest(self._ipos(gameState.getAgentPosition(i)), ebnd)))
        self.is_attacker = (self.index == attacker)

    # ------------------------------------------------------------------ #
    #  Food / capsule / ghost targets                                      #
    # ------------------------------------------------------------------ #

    def _best_food(self, gameState, my_pos):
        foods = [self._ipos(f) for f in self.getFood(gameState).asList()]
        if not foods: return None
        carry    = getattr(gameState.getAgentState(self.index), 'numCarrying', 0)
        home_tgt = self._nearest(my_pos, self.home_boundary)
        ally_pos = _Shared.info.get(self.ally_idx, {}).get('pos')
        scared   = self._all_ghosts_scared(gameState)
        danger   = self._danger_tiles(gameState)
        candidates = sorted(foods, key=lambda f: self.getMazeDistance(my_pos, f))[:20]
        best, bs = None, float('inf')
        for f in candidates:
            d_me    = self.getMazeDistance(my_pos, f)
            d_home  = self.getMazeDistance(f, home_tgt)
            cluster = sum(1 for ff in foods if self.getMazeDistance(f, ff) <= 2)
            depth   = self.dead_depth.get(f, 0)
            haz     = 20.0 if (f in danger and carry > 0 and not scared) else 0.0
            corr    = depth * 1.2 if carry > 0 and depth >= 2 else 0.0
            ally_p  = max(0, 4 - self.getMazeDistance(f, ally_pos)) * 1.5 \
                      if ally_pos and self.getMazeDistance(f, ally_pos) <= 4 else 0.0
            score   = d_me + 0.8 * d_home - 1.5 * cluster + haz + corr + ally_p + carry * d_home * 0.2
            if score < bs: bs, best = score, f
        return best

    def _nearest_capsule(self, gameState, my_pos):
        caps = [self._ipos(c) for c in self.getCapsules(gameState)]
        return min(caps, key=lambda c: self.getMazeDistance(my_pos, c)) if caps else None

    def _nearest_scared(self, gameState, my_pos):
        scared = [
            self._ipos(gameState.getAgentState(i).getPosition())
            for i in self.getOpponents(gameState)
            if gameState.getAgentState(i)
               and gameState.getAgentState(i).getPosition() is not None
               and not gameState.getAgentState(i).isPacman
               and getattr(gameState.getAgentState(i), 'scaredTimer', 0) >= SCARED_MIN
        ]
        return min(scared, key=lambda p: self.getMazeDistance(my_pos, p)) if scared else None

    # ------------------------------------------------------------------ #
    #  Defence                                                             #
    # ------------------------------------------------------------------ #

    def _defend_target(self, gameState, my_pos):
        invaders = [
            self._ipos(gameState.getAgentState(i).getPosition())
            for i in self.getOpponents(gameState)
            if gameState.getAgentState(i) and gameState.getAgentState(i).isPacman
               and gameState.getAgentState(i).getPosition() is not None
        ]
        if invaders:
            iv = min(invaders, key=lambda p: self.getMazeDistance(my_pos, p))
            return iv if self.getMazeDistance(my_pos, iv) <= 6 else self._cut_off(my_pos, iv)
        if self._last_eaten_pos and self.tick - self._last_eaten_tick <= 20:
            return self._cut_off(my_pos, self._last_eaten_pos)
        dcaps = [self._ipos(c) for c in self.getCapsulesYouAreDefending(gameState)]
        if dcaps:
            return min(dcaps, key=lambda c: self.getMazeDistance(my_pos, c))
        target = self.patrol_pts[self.patrol_idx % len(self.patrol_pts)]
        if self.getMazeDistance(my_pos, target) <= 1:
            self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_pts)
            target = self.patrol_pts[self.patrol_idx]
        return target

    def _cut_off(self, my_pos, invader_pos):
        cands = self.entrances or self.home_boundary
        best, bv = None, float('inf')
        for b in cands:
            my_d = self.getMazeDistance(my_pos, b)
            iv_d = self.getMazeDistance(invader_pos, b)
            if my_d <= iv_d + 2:
                v = my_d + 0.5 * iv_d
                if v < bv: bv, best = v, b
        return best or min(cands,
            key=lambda b: self.getMazeDistance(invader_pos, b) + 0.7 * self.getMazeDistance(my_pos, b))

    # ------------------------------------------------------------------ #
    #  A*                                                                  #
    # ------------------------------------------------------------------ #

    def _astar_step(self, gameState, start, target, avoid, legal) -> str:
        if start == target: return random.choice(legal)
        ghost_tiles = self._ghost_tiles(gameState) if avoid else set()
        pq   = PriorityQueue()
        pq.push((start, None), 0)
        seen: Dict[Tuple, int] = {}
        while not pq.isEmpty():
            cur, first_a = pq.pop()
            if cur == target:
                return first_a if (first_a and first_a in legal) else self._greedy(start, target, legal)
            g_cur = seen.get(cur, 0)
            for action in (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST):
                dx, dy = Actions.directionToVector(action)
                nxt    = (int(cur[0]+dx), int(cur[1]+dy))
                if not self._valid(nxt): continue
                if avoid and nxt in ghost_tiles and nxt != target: continue
                g_nxt = g_cur + 1
                if g_nxt >= seen.get(nxt, float('inf')): continue
                seen[nxt] = g_nxt
                fa      = first_a if first_a is not None else action
                ghost_b = max(0, 5 - self._dist_to_set(nxt, ghost_tiles)) * 2.0 \
                          if avoid and ghost_tiles else 0.0
                corr_b  = self.dead_depth.get(nxt, 0) * 0.5
                pq.push((nxt, fa), g_nxt + self.getMazeDistance(nxt, target) + ghost_b + corr_b)
        return self._greedy(start, target, legal)

    def _ghost_tiles(self, gameState) -> Set[Tuple]:
        tiles = set()
        for idx in self.getOpponents(gameState):
            st = gameState.getAgentState(idx)
            if st and not st.isPacman and st.getPosition() is not None \
               and getattr(st, 'scaredTimer', 0) <= 2:
                gx, gy = self._ipos(st.getPosition())
                for dx, dy in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]:
                    if self._valid((gx+dx, gy+dy)): tiles.add((gx+dx, gy+dy))
                prev = self._ghost_prev.get(idx)
                if prev:
                    px2 = gx + (gx - prev[0]); py2 = gy + (gy - prev[1])
                    if self._valid((px2, py2)):
                        for dx, dy in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]:
                            if self._valid((px2+dx, py2+dy)): tiles.add((px2+dx, py2+dy))
        return tiles

    def _escape(self, gameState, legal) -> str:
        my_pos   = self._pos(gameState)
        ghosts   = [self._ipos(gameState.getAgentState(i).getPosition())
                    for i in self.getOpponents(gameState)
                    if gameState.getAgentState(i) and not gameState.getAgentState(i).isPacman
                       and gameState.getAgentState(i).getPosition() is not None
                       and getattr(gameState.getAgentState(i), 'scaredTimer', 0) <= 2]
        home_tgt = self._nearest(my_pos, self.home_boundary)
        best, bs = None, float('-inf')
        for a in legal:
            dx, dy = Actions.directionToVector(a)
            nxt    = (int(my_pos[0]+dx), int(my_pos[1]+dy))
            if not self._valid(nxt): continue
            gd = min((self.getMazeDistance(nxt, g) for g in ghosts), default=10)
            hd = self.getMazeDistance(nxt, home_tgt)
            if gd - hd * 0.5 > bs: bs, best = gd - hd * 0.5, a
        return best or random.choice(legal)

    # ------------------------------------------------------------------ #
    #  Memory / shared state                                               #
    # ------------------------------------------------------------------ #

    def _update_memory(self, gameState):
        cur   = self._food_set(self.getFoodYouAreDefending(gameState))
        eaten = self._def_food_prev - cur
        if eaten:
            self._last_eaten_pos  = min(eaten,
                key=lambda p: self.getMazeDistance(self._ipos(self.start), p))
            self._last_eaten_tick = self.tick
        self._def_food_prev = cur
        for idx in self.getOpponents(gameState):
            st = gameState.getAgentState(idx)
            if st and st.getPosition() is not None and not st.isPacman:
                self._ghost_prev[idx] = self._ipos(st.getPosition())

    def _update_shared(self, gameState):
        st = gameState.getAgentState(self.index)
        _Shared.info[self.index] = {
            'pos':       self._pos(gameState),
            'carrying':  getattr(st, 'numCarrying', 0),
            'is_pacman': st.isPacman,
            'mode':      _Shared.info.get(self.index, {}).get('mode', 'ATTACK'),
        }

    # ------------------------------------------------------------------ #
    #  Threat helpers                                                      #
    # ------------------------------------------------------------------ #

    def _ghost_dist(self, gameState, pos) -> float:
        m = float('inf')
        for idx in self.getOpponents(gameState):
            st = gameState.getAgentState(idx)
            if st and not st.isPacman and st.getPosition() is not None \
               and getattr(st, 'scaredTimer', 0) <= 2:
                m = min(m, self.getMazeDistance(pos, self._ipos(st.getPosition())))
        return m

    def _all_ghosts_scared(self, gameState) -> bool:
        seen = 0
        for idx in self.getOpponents(gameState):
            st = gameState.getAgentState(idx)
            if st and st.getPosition() is not None and not st.isPacman:
                seen += 1
                if getattr(st, 'scaredTimer', 0) < SCARED_MIN: return False
        return seen > 0

    def _danger_tiles(self, gameState) -> Set[Tuple]:
        tiles = set()
        for idx in self.getOpponents(gameState):
            st = gameState.getAgentState(idx)
            if st and not st.isPacman and st.getPosition() is not None \
               and getattr(st, 'scaredTimer', 0) <= 2:
                gx, gy = self._ipos(st.getPosition())
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if abs(dx)+abs(dy) <= 3 and self._valid((gx+dx, gy+dy)):
                            tiles.add((gx+dx, gy+dy))
        return tiles

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def _oscillating(self) -> bool:
        return len(self._pos_hist) >= 6 \
               and Counter(self._pos_hist).most_common(1)[0][1] >= 4

    def _greedy(self, start, target, legal) -> str:
        best, bd = None, float('inf')
        for a in legal:
            dx, dy = Actions.directionToVector(a)
            nxt    = (int(start[0]+dx), int(start[1]+dy))
            d      = self.getMazeDistance(nxt, target)
            if d < bd: bd, best = d, a
        return best or random.choice(legal)

    def _nearest(self, pos, pts):
        return min(pts, key=lambda p: self.getMazeDistance(pos, p))

    def _dist_to_set(self, pos, s) -> float:
        return min((self.getMazeDistance(pos, t) for t in s), default=float('inf'))

    def _pos(self, gameState) -> Tuple:
        return self._ipos(gameState.getAgentPosition(self.index))

    def _ipos(self, pos) -> Tuple:
        return (int(round(pos[0])), int(round(pos[1])))

    def _valid(self, pos) -> bool:
        x, y = int(pos[0]), int(pos[1])
        return 0 <= x < self.walls.width and 0 <= y < self.walls.height \
               and not self.walls[x][y]

    def _food_set(self, grid) -> set:
        return {(x, y) for x in range(grid.width)
                for y in range(grid.height) if grid[x][y]}

    def getSuccessor(self, gameState, action):
        s   = gameState.generateSuccessor(self.index, action)
        pos = s.getAgentState(self.index).getPosition()
        return s.generateSuccessor(self.index, action) \
               if pos != nearestPoint(pos) else s
