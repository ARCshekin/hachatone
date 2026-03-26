#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import socket
import sys

AVOID_DISTANCE = 64.0
STUCK_TICKS_THRESHOLD = 5
STUCK_MOVE_EPS = 2.0
ESCAPE_OVERRIDE_TICKS = 8

_LAST_POS: list[float] | None = None
_STUCK_TICKS = 0
_FORCED_SIDE = 0  # -1 => prefer left, +1 => prefer right
_FORCED_SIDE_TICKS_LEFT = 0


def _norm(dx: float, dy: float) -> list[float]:
    length = math.hypot(dx, dy)
    if length <= 1e-8:
        return [1.0, 0.0]
    return [dx / length, dy / length]


def _distance(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _line_intersects_aabb(start: list[float], end: list[float], obstacle: dict) -> float | None:
    center = obstacle.get("center") or [0.0, 0.0]
    half = obstacle.get("half_size") or [0.0, 0.0]
    min_x = float(center[0]) - float(half[0])
    max_x = float(center[0]) + float(half[0])
    min_y = float(center[1]) - float(half[1])
    max_y = float(center[1]) + float(half[1])

    sx = float(start[0])
    sy = float(start[1])
    ex = float(end[0])
    ey = float(end[1])
    dx = ex - sx
    dy = ey - sy

    t_min = 0.0
    t_max = 1.0
    for axis in ("x", "y"):
        origin = sx if axis == "x" else sy
        d = dx if axis == "x" else dy
        slab_min = min_x if axis == "x" else min_y
        slab_max = max_x if axis == "x" else max_y
        if abs(d) < 1e-9:
            if origin < slab_min or origin > slab_max:
                return None
            continue
        inv = 1.0 / d
        t1 = (slab_min - origin) * inv
        t2 = (slab_max - origin) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None

    if 0.0 <= t_min <= 1.0:
        return t_min
    if 0.0 <= t_max <= 1.0:
        return t_max
    return None


def _first_blocker(start: list[float], end: list[float], obstacles: list[dict]) -> dict | None:
    best_t = None
    best = None
    for obstacle in obstacles:
        if not bool(obstacle.get("solid", False)):
            continue
        t = _line_intersects_aabb(start, end, obstacle)
        if t is None:
            continue
        if best_t is None or t < best_t:
            best_t = t
            best = obstacle
    return best


def _navigate_towards(
    target: list[float],
    you_pos: list[float],
    enemy_pos: list[float],
    obstacles: list[dict],
    forced_side: int = 0,
) -> list[float]:
    direct = _norm(float(target[0]) - float(you_pos[0]), float(target[1]) - float(you_pos[1]))
    direct_end = [
        float(you_pos[0]) + direct[0] * AVOID_DISTANCE,
        float(you_pos[1]) + direct[1] * AVOID_DISTANCE,
    ]
    if _first_blocker(you_pos, direct_end, obstacles) is None:
        return direct

    # If direct lane is blocked, try two tangent-like sidesteps around the blocker.
    left = [-direct[1], direct[0]]
    right = [direct[1], -direct[0]]
    enemy_vec = [float(enemy_pos[0]) - float(you_pos[0]), float(enemy_pos[1]) - float(you_pos[1])]
    cross = enemy_vec[0] * direct[1] - enemy_vec[1] * direct[0]
    if forced_side < 0:
        options = [left, right]
    elif forced_side > 0:
        options = [right, left]
    else:
        options = [left, right] if cross >= 0.0 else [right, left]

    for side in options:
        candidate = _norm(direct[0] * 0.55 + side[0] * 0.9, direct[1] * 0.55 + side[1] * 0.9)
        candidate_end = [
            float(you_pos[0]) + candidate[0] * AVOID_DISTANCE,
            float(you_pos[1]) + candidate[1] * AVOID_DISTANCE,
        ]
        if _first_blocker(you_pos, candidate_end, obstacles) is None:
            return candidate

    # Last fallback: pure sidestep (better than standing still in front of a wall).
    return options[0]


def _nearest_pickup(you_pos: list[float], pickups: list[dict]) -> dict | None:
    best = None
    best_dist = float("inf")
    for pickup in pickups:
        if float(pickup.get("cooldown", 0.0) or 0.0) > 0.0:
            continue
        position = pickup.get("position") or [0.0, 0.0]
        dist = _distance(you_pos, position)
        if dist < best_dist:
            best_dist = dist
            best = pickup
    return best


def _nearest_letterbox(you_pos: list[float], letterboxes: list[dict]) -> dict | None:
    best = None
    best_dist = float("inf")
    for letterbox in letterboxes:
        if not bool(letterbox.get("ready", False)):
            continue
        position = letterbox.get("position") or [0.0, 0.0]
        dist = _distance(you_pos, position)
        if dist < best_dist:
            best_dist = dist
            best = letterbox
    return best


def decide(message: dict, seq: int) -> dict | None:
    global _LAST_POS, _STUCK_TICKS, _FORCED_SIDE, _FORCED_SIDE_TICKS_LEFT

    if message.get("type") != "tick":
        return None

    you = message.get("you") or {}
    enemy = message.get("enemy") or {}
    snapshot = message.get("snapshot") or {}
    you_pos = you.get("position") or [0.0, 0.0]
    enemy_pos = enemy.get("position") or [0.0, 0.0]
    obstacles = list(snapshot.get("obstacles") or [])
    dx = float(enemy_pos[0]) - float(you_pos[0])
    dy = float(enemy_pos[1]) - float(you_pos[1])
    aim = _norm(dx, dy)
    distance = math.hypot(dx, dy)

    weapon = you.get("weapon")
    has_weapon = isinstance(weapon, dict) and int(weapon.get("ammo", 0) or 0) > 0
    weapon_type = str((weapon or {}).get("type") or "").lower()

    move = [0.0, 0.0]
    shoot = False
    kick = False
    pickup = False

    forced_side = _FORCED_SIDE if _FORCED_SIDE_TICKS_LEFT > 0 else 0

    if not has_weapon:
        nearest_pickup = _nearest_pickup(you_pos, list(snapshot.get("pickups") or []))
        if nearest_pickup is not None:
            target = nearest_pickup.get("position") or [0.0, 0.0]
            move = _navigate_towards(target, you_pos, enemy_pos, obstacles, forced_side=forced_side)
            aim = move
            pickup = _distance(you_pos, target) <= 24.0
        else:
            nearest_letterbox = _nearest_letterbox(you_pos, list(snapshot.get("letterboxes") or []))
            if nearest_letterbox is not None:
                target = nearest_letterbox.get("position") or [0.0, 0.0]
                move = _navigate_towards(target, you_pos, enemy_pos, obstacles, forced_side=forced_side)
                aim = move
                kick = _distance(you_pos, target) <= 24.0
            else:
                move = _navigate_towards(enemy_pos, you_pos, enemy_pos, obstacles, forced_side=forced_side)
                kick = distance <= 28.0
    else:
        desired_range = 170.0 if weapon_type == "revolver" else 105.0
        if distance > desired_range + 18.0:
            move = _navigate_towards(enemy_pos, you_pos, enemy_pos, obstacles, forced_side=forced_side)
        elif distance < desired_range - 35.0:
            retreat_target = [float(you_pos[0]) - aim[0] * 80.0, float(you_pos[1]) - aim[1] * 80.0]
            move = _navigate_towards(retreat_target, you_pos, enemy_pos, obstacles, forced_side=forced_side)
        else:
            strafe_target = [float(you_pos[0]) + aim[1] * 80.0, float(you_pos[1]) - aim[0] * 80.0]
            move = _navigate_towards(strafe_target, you_pos, enemy_pos, obstacles, forced_side=forced_side)

        shoot = bool(enemy.get("alive", False))
        kick = distance <= 28.0 and float(you.get("kick_cooldown", 0.0) or 0.0) <= 0.0

    # Anti-stuck: if movement command exists but position hardly changes for several ticks,
    # temporarily force an opposite bypass side around blockers.
    move_mag = math.hypot(float(move[0]), float(move[1]))
    if _LAST_POS is not None and move_mag > 0.2:
        step_moved = _distance(you_pos, _LAST_POS)
        if step_moved < STUCK_MOVE_EPS:
            _STUCK_TICKS += 1
        else:
            _STUCK_TICKS = 0
    else:
        _STUCK_TICKS = 0

    if _STUCK_TICKS >= STUCK_TICKS_THRESHOLD:
        if _FORCED_SIDE == 0:
            _FORCED_SIDE = -1 if (seq % 2 == 0) else 1
        else:
            _FORCED_SIDE = -_FORCED_SIDE
        _FORCED_SIDE_TICKS_LEFT = ESCAPE_OVERRIDE_TICKS
        _STUCK_TICKS = 0
    elif _FORCED_SIDE_TICKS_LEFT > 0:
        _FORCED_SIDE_TICKS_LEFT -= 1
        if _FORCED_SIDE_TICKS_LEFT <= 0:
            _FORCED_SIDE = 0

    _LAST_POS = [float(you_pos[0]), float(you_pos[1])]

    return {
        "type": "command",
        "seq": seq,
        "move": move,
        "aim": aim,
        "shoot": shoot,
        "kick": kick,
        "pickup": pickup,
        "drop": False,
        "throw": False,
        "interact": False,
    }


def main() -> int:
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 9001

    with socket.create_connection((host, port), timeout=15) as sock:
        sock.settimeout(None)
        reader = sock.makefile("r", encoding="utf-8", newline="\n")
        writer = sock.makefile("w", encoding="utf-8", newline="\n")
        writer.write(json.dumps({"type": "register", "name": "sim_opponent"}) + "\n")
        writer.flush()

        seq = 0
        for line in reader:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            command = decide(payload, seq)
            if command is None:
                continue
            writer.write(json.dumps(command) + "\n")
            writer.flush()
            seq += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
