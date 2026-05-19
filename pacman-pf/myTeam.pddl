; Header and description

(define (domain pacman_bool)
  ; remove requirements that are not needed
  (:requirements :strips :typing :negative-preconditions)

  ; ---------------- TYPES ----------------
  (:types 
    ; Type hierarchy
    enemy team - object
    enemy1 enemy2 - enemy
    ally current_agent - team
  )

  ; ---------------- PREDICATES ----------------
  (:predicates 
    ; Basic state predicates (these usually come from the game state)
    (enemy_around ?e - enemy ?a - team)          ; enemy e is close to agent a (about 4 tiles)
    (is_pacman ?x)                                ; x is on enemy side / acting as pacman
    (food_in_backpack ?a - team)                  ; agent a is carrying food
    (food_available)                              ; still food to eat on enemy side
    (capsule_available)                           ; at least one capsule on map
    (near_food ?a - current_agent)                ; food is near this agent
    (near_capsule ?a - current_agent)             ; capsule is near this agent

    ; Advanced state predicates (used to pick safer or smarter plans)
    (3_food_in_backpack ?a - team)                ; carrying 3+
    (5_food_in_backpack ?a - team)                ; carrying 5+
    (10_food_in_backpack ?a - team)               ; carrying 10+
    (winning)                                     ; we are leading
    (winning_gt3)                                 ; lead > 3
    (winning_gt5)                                 ; lead > 5
    (winning_gt10)                                ; lead > 10
    (winning_gt20)                                ; lead > 20
    (near_ally ?a - current_agent)                ; our ally is close to us
    (is_scared ?e)                                ; enemy is scared (after capsule)
    (endgame)                                     ; time is almost finished

    ; Enemy tracking predicates (from visible or noisy distance)
    (enemy_short_distance ?e - enemy ?a - current_agent)  ; enemy very close
    (enemy_medium_distance ?e - enemy ?a - current_agent) ; enemy mid distance
    (enemy_long_distance ?e - enemy ?a - current_agent)   ; enemy far
    (enemy_is_pacman ?e - enemy)                 ; enemy is invading our side
    (enemy_visible ?e - enemy)                   ; we can see this enemy

    ; Defensive predicates
    (invader_visible ?e - enemy)                  ; an enemy pacman is seen

    ; Cooperative predicates (info shared from our ally)
    (ally_attacking ?a - ally)                    ; ally is attacking
    (ally_defending ?a - ally)                    ; ally is defending
    (ally_returning ?a - ally)                    ; ally is going home
    (ally_in_danger ?a - ally)                    ; ally is being chased

    ; Helper / virtual predicates
    (ally_cover ?a - ally)                        ; ally is covering us, so attack is safer
    (safe_to_collect ?a - current_agent)          ; map is safe now to collect
    (carry_some ?a - team)                        ; carrying at least 1
    (threatened ?a - current_agent)               ; ghost is close, need to return

    ; Virtual goal states (planner effects, not real world)
    (defend_foods)                                ; we are in defend mode
    (maximize_score)                              ; we returned food / banked
    (safe_collection)                             ; we completed a safe collect action
  )
 
  ; === PRIMARY COLLECTION ACTIONS ===

  ; most permissive: if food exists and it is near, just grab it
  (:action risk_reward_collection
    :parameters (?a - current_agent)
    :precondition (and 
      (food_available)
      (near_food ?a)
    )
    :effect (and 
      (food_in_backpack ?a)   ; now carrying
      (safe_collection)       ; mark success
    )
  )

  ; safer version: only collect when planner thinks it is safe
  (:action safe_collection
    :parameters (?a - current_agent)
    :precondition (and 
      (food_available)
      (safe_to_collect ?a)
    )
    :effect (and 
      (food_in_backpack ?a)
      (safe_collection)
    )
  )

  ; attack when enemy defenders are not pacmen and ally is covering
  (:action aggressive_attack
    :parameters (?a - current_agent ?e1 - enemy1 ?e2 - enemy2 ?ally - ally)
    :precondition (and 
      (food_available)
      (not (enemy_is_pacman ?e1))
      (not (enemy_is_pacman ?e2))
      (ally_cover ?ally)
      (not (winning_gt10)) ; if already winning a lot, no need to risk
    )
    :effect (and 
      (food_in_backpack ?a)
      (safe_collection)
    )
  )

  ; attack together with ally when ally is also attacking
  (:action coordinated_attack
    :parameters (?a - current_agent ?ally - ally ?e1 - enemy1 ?e2 - enemy2)
    :precondition (and 
      (food_available)
      (not (enemy_is_pacman ?e1))
      (not (enemy_is_pacman ?e2))
      (ally_attacking ?ally)
    )
    :effect (and 
      (food_in_backpack ?a)
      (safe_collection)
    )
  )

  ; collect only when ally is defending, so map is safer
  (:action opportunistic_collect
    :parameters (?a - current_agent ?ally - ally)
    :precondition (and 
      (food_available)
      (near_food ?a)
      (ally_defending ?ally)
      (not (winning_gt20))  ; if lead is very high, do not risk
    )
    :effect (and 
      (food_in_backpack ?a)
      (safe_collection)
    )
  )

  ; === RETURN HOME ACTIONS ===

  ; return if we are chased / threatened
  (:action return_with_food_threat
    :parameters (?a - current_agent)
    :precondition (and 
      (is_pacman ?a)        ; we are on enemy side
      (carry_some ?a)       ; we have food
      (threatened ?a)       ; ghost close
    )
    :effect (and 
      (not (is_pacman ?a))          ; back to our side
      (not (food_in_backpack ?a))   ; food is banked
      (maximize_score)
    )
  )

  ; return quickly if carrying 10+ food
  (:action urgent_return_carry10
    :parameters (?a - current_agent)
    :precondition (and 
      (is_pacman ?a)
      (10_food_in_backpack ?a)
    )
    :effect (and 
      (not (is_pacman ?a))
      (not (food_in_backpack ?a))
      (maximize_score)
    )
  )

  ; return in endgame to secure points
  (:action urgent_return_endgame
    :parameters (?a - current_agent)
    :precondition (and 
      (is_pacman ?a)
      (endgame)
      (carry_some ?a)
    )
    :effect (and 
      (not (is_pacman ?a))
      (not (food_in_backpack ?a))
      (maximize_score)
    )
  )

  ; === DEFENSIVE ACTIONS ===

  ; chase invader when we see one and ally is not already defending
  (:action intercept_invader
    :parameters (?a - current_agent ?e - enemy ?ally - ally)
    :precondition (and 
      (invader_visible ?e)        ; we see invader
      (enemy_is_pacman ?e)        ; enemy is on our side
      (not (is_pacman ?a))        ; we are defender
      (not (ally_defending ?ally)) ; to avoid both chasing same target
    )
    :effect (and 
      (defend_foods)
      (not (enemy_is_pacman ?e))  ; we “removed” invader
    )
  )

  ; defend together if ally is already defending
  (:action coordinated_defence
    :parameters (?a - current_agent ?e1 - enemy1 ?e2 - enemy2 ?ally - ally)
    :precondition (and 
      (invader_visible ?e1)
      (not (is_pacman ?a))
      (ally_defending ?ally)
      (not (near_ally ?a))  ; spread out, not standing on same tile
    )
    :effect (and 
      (defend_foods)
    )
  )

  ; patrol when we are winning well and no invader is visible
  (:action patrol_win
    :parameters (?a - current_agent ?e1 - enemy1 ?e2 - enemy2)
    :precondition (and 
      (not (is_pacman ?a))
      (not (invader_visible ?e1))
      (not (invader_visible ?e2))
      (winning_gt5)
    )
    :effect (and 
      (defend_foods)
    )
  )

  ; patrol in endgame to protect last foods
  (:action patrol_endgame
    :parameters (?a - current_agent ?e1 - enemy1 ?e2 - enemy2)
    :precondition (and 
      (not (is_pacman ?a))
      (not (invader_visible ?e1))
      (not (invader_visible ?e2))
      (endgame)
    )
    :effect (and 
      (defend_foods)
    )
  )

  ; === SPECIAL ACTIONS ===

  ; eat capsule when near and enemy is around and not scared yet
  (:action eat_capsule
    :parameters (?a - current_agent ?e - enemy)
    :precondition (and 
      (capsule_available)
      (near_capsule ?a)
      (enemy_around ?e ?a)
      (not (is_scared ?e))
    )
    :effect (and 
      (is_scared ?e)             ; enemy becomes scared
      (not (capsule_available))  ; capsule removed
    )
  )

  ; help ally if ally is in danger and we are close
  (:action support_ally
    :parameters (?a - current_agent ?ally - ally)
    :precondition (and 
      (ally_in_danger ?ally)
      (near_ally ?a)
      (not (is_pacman ?a))  ; support only when we are on our side
    )
    :effect (and 
      (not (ally_in_danger ?ally)) ; we “saved” ally
    )
  )

  ; === IMPROVEMENT: Scared ghost hunting ===
  ; When an enemy ghost is scared (after capsule), our Pacman can chase and eat them.
  ; Each eaten scared ghost = big score swing.
  (:action hunt_scared_ghost
    :parameters (?a - current_agent ?e - enemy)
    :precondition (and 
      (is_pacman ?a)         ; we are on enemy side
      (is_scared ?e)         ; enemy ghost is scared (capsule active)
      (enemy_visible ?e)     ; we can see them
      (not (enemy_is_pacman ?e)) ; they are a ghost, not a pacman
    )
    :effect (and 
      (maximize_score)       ; eating them scores points
    )
  )

)
