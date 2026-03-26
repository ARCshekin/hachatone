#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import socket
from pathlib import Path


INPUT_SIZE = 20
OUTPUT_SIZE = 7


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _len2(x: float, y: float) -> float:
    return math.hypot(x, y)


def _norm(dx: float, dy: float) -> list[float]:
    mag = _len2(dx, dy)
    if mag <= 1e-8:
        return [1.0, 0.0]
    return [dx / mag, dy / mag]


def _nearest_pickup(you_pos: list[float], pickups: list[dict]) -> tuple[float, float, float]:
    best_dx, best_dy, best_dist = 0.0, 0.0, 10_000.0
    for pickup in pickups:
        if float(pickup.get("cooldown", 0.0) or 0.0) > 0.0:
            continue
        pos = pickup.get("position") or [0.0, 0.0]
        dx = float(pos[0]) - float(you_pos[0])
        dy = float(pos[1]) - float(you_pos[1])
        d = _len2(dx, dy)
        if d < best_dist:
            best_dx, best_dy, best_dist = dx, dy, d
    if best_dist >= 9_999.0:
        return 0.0, 0.0, 1.0
    return best_dx, best_dy, best_dist


def _nearest_enemy_projectile(you_pos: list[float], projectiles: list[dict], my_player_id: int) -> tuple[float, float, float]:
    best_dx, best_dy, best_dist = 0.0, 0.0, 10_000.0
    for projectile in projectiles:
        owner = int(projectile.get("owner_id", projectile.get("owner", 0)) or 0)
        if owner == my_player_id:
            continue
        pos = projectile.get("position") or [0.0, 0.0]
        dx = float(pos[0]) - float(you_pos[0])
        dy = float(pos[1]) - float(you_pos[1])
        d = _len2(dx, dy)
        if d < best_dist:
            best_dx, best_dy, best_dist = dx, dy, d
    if best_dist >= 9_999.0:
        return 0.0, 0.0, 1.0
    return best_dx, best_dy, best_dist


class RuntimePerceptron:
    def __init__(self, genome: list[float], hidden_size_1: int, hidden_size_2: int) -> None:
        expected = (
            hidden_size_1 * INPUT_SIZE
            + hidden_size_1
            + hidden_size_2 * hidden_size_1
            + hidden_size_2
            + OUTPUT_SIZE * hidden_size_2
            + OUTPUT_SIZE
        )
        if len(genome) != expected:
            raise ValueError(f"Genome length mismatch: expected {expected}, got {len(genome)}")
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.seq = 0

        cursor = 0
        self.w1 = genome[cursor : cursor + hidden_size_1 * INPUT_SIZE]
        cursor += hidden_size_1 * INPUT_SIZE
        self.b1 = genome[cursor : cursor + hidden_size_1]
        cursor += hidden_size_1
        self.w2 = genome[cursor : cursor + hidden_size_2 * hidden_size_1]
        cursor += hidden_size_2 * hidden_size_1
        self.b2 = genome[cursor : cursor + hidden_size_2]
        cursor += hidden_size_2
        self.w3 = genome[cursor : cursor + OUTPUT_SIZE * hidden_size_2]
        cursor += OUTPUT_SIZE * hidden_size_2
        self.b3 = genome[cursor : cursor + OUTPUT_SIZE]

    def _forward(self, features: list[float]) -> list[float]:
        hidden1 = [0.0] * self.hidden_size_1
        for h in range(self.hidden_size_1):
            base = h * INPUT_SIZE
            s = self.b1[h]
            for i in range(INPUT_SIZE):
                s += self.w1[base + i] * features[i]
            hidden1[h] = math.tanh(s)
        hidden2 = [0.0] * self.hidden_size_2
        for h2 in range(self.hidden_size_2):
            base = h2 * self.hidden_size_1
            s = self.b2[h2]
            for h1 in range(self.hidden_size_1):
                s += self.w2[base + h1] * hidden1[h1]
            hidden2[h2] = math.tanh(s)
        out = [0.0] * OUTPUT_SIZE
        for o in range(OUTPUT_SIZE):
            base = o * self.hidden_size_2
            s = self.b3[o]
            for h2 in range(self.hidden_size_2):
                s += self.w3[base + h2] * hidden2[h2]
            out[o] = s
        return out

    def act(self, message: dict, player_id: int) -> dict:
        self.seq += 1
        features, info = build_features(message, player_id)
        logits = self._forward(features)

        move_x = math.tanh(logits[0])
        move_y = math.tanh(logits[1])
        move_len = _len2(move_x, move_y)
        if move_len > 1.0:
            move_x /= move_len
            move_y /= move_len

        to_enemy = info["to_enemy"]
        default_aim = _norm(to_enemy[0], to_enemy[1])
        aim_x = 0.6 * math.tanh(logits[2]) + 0.4 * default_aim[0]
        aim_y = 0.6 * math.tanh(logits[3]) + 0.4 * default_aim[1]
        aim = _norm(aim_x, aim_y)

        enemy_dist = float(info["enemy_dist"])
        shoot = _sigmoid(logits[4]) > 0.5 and enemy_dist < 320.0
        kick = _sigmoid(logits[5]) > 0.55 and enemy_dist < 30.0
        pickup = _sigmoid(logits[6]) > 0.55

        return {
            "type": "command",
            "seq": self.seq,
            "move": [move_x, move_y],
            "aim": aim,
            "shoot": shoot,
            "kick": kick,
            "pickup": pickup,
            "drop": False,
            "throw": False,
            "interact": False,
        }


def build_features(message: dict, player_id: int) -> tuple[list[float], dict]:
    you = message.get("you") or {}
    enemy = message.get("enemy") or {}
    snapshot = message.get("snapshot") or {}

    you_pos = you.get("position") or [0.0, 0.0]
    enemy_pos = enemy.get("position") or [0.0, 0.0]
    dx = float(enemy_pos[0]) - float(you_pos[0])
    dy = float(enemy_pos[1]) - float(you_pos[1])
    enemy_dist = _len2(dx, dy)

    you_vel = you.get("velocity") or [0.0, 0.0]
    enemy_vel = enemy.get("velocity") or [0.0, 0.0]
    me_speed = _len2(float(you_vel[0]), float(you_vel[1]))
    enemy_speed = _len2(float(enemy_vel[0]), float(enemy_vel[1]))

    pick_dx, pick_dy, pick_dist = _nearest_pickup(you_pos, list(snapshot.get("pickups") or []))
    proj_dx, proj_dy, proj_dist = _nearest_enemy_projectile(
        you_pos,
        list(snapshot.get("projectiles") or []),
        player_id,
    )

    weapon = you.get("weapon") if isinstance(you.get("weapon"), dict) else None
    enemy_weapon = enemy.get("weapon") if isinstance(enemy.get("weapon"), dict) else None
    me_has_weapon = 1.0 if weapon is not None and int(weapon.get("ammo", 0) or 0) > 0 else 0.0
    enemy_has_weapon = 1.0 if enemy_weapon is not None and int(enemy_weapon.get("ammo", 0) or 0) > 0 else 0.0
    me_ammo = float((weapon or {}).get("ammo", 0) or 0) / 35.0
    enemy_ammo = float((enemy_weapon or {}).get("ammo", 0) or 0) / 35.0

    features = [
        _clamp(dx / 500.0, -1.0, 1.0),
        _clamp(dy / 500.0, -1.0, 1.0),
        _clamp(enemy_dist / 500.0, 0.0, 1.0),
        _clamp(float(you_vel[0]) / 300.0, -1.0, 1.0),
        _clamp(float(you_vel[1]) / 300.0, -1.0, 1.0),
        _clamp(float(enemy_vel[0]) / 300.0, -1.0, 1.0),
        _clamp(float(enemy_vel[1]) / 300.0, -1.0, 1.0),
        _clamp(me_speed / 300.0, 0.0, 1.0),
        _clamp(enemy_speed / 300.0, 0.0, 1.0),
        me_has_weapon,
        enemy_has_weapon,
        _clamp(me_ammo, 0.0, 1.0),
        _clamp(enemy_ammo, 0.0, 1.0),
        _clamp(float(you.get("shoot_cooldown", 0.0) or 0.0) / 0.5, 0.0, 1.0),
        _clamp(float(you.get("kick_cooldown", 0.0) or 0.0) / 1.0, 0.0, 1.0),
        _clamp(float(enemy.get("shoot_cooldown", 0.0) or 0.0) / 0.5, 0.0, 1.0),
        _clamp(pick_dx / 500.0, -1.0, 1.0),
        _clamp(pick_dy / 500.0, -1.0, 1.0),
        _clamp(pick_dist / 500.0, 0.0, 1.0),
        _clamp(proj_dist / 500.0, 0.0, 1.0),
    ]
    if proj_dist < 0.999:
        features[18] = _clamp(proj_dx / 500.0, -1.0, 1.0)
        features[19] = _clamp(proj_dy / 500.0, -1.0, 1.0)

    if len(features) != INPUT_SIZE:
        raise RuntimeError(f"Feature size mismatch: expected {INPUT_SIZE}, got {len(features)}")
    return features, {"to_enemy": [dx, dy], "enemy_dist": enemy_dist}


def load_policy(path: Path) -> RuntimePerceptron:
    data = json.loads(path.read_text(encoding="utf-8"))
    hidden_size_1 = int(data.get("hidden_size_1", data.get("hidden_size", 16)))
    hidden_size_2 = int(data.get("hidden_size_2", 40))
    genome = data.get("genome")
    if genome is None:
        genome = data.get("best_genome")
    if not isinstance(genome, list):
        raise ValueError("Genome JSON must contain list field: genome or best_genome")
    genome_f = [float(x) for x in genome]
    return RuntimePerceptron(
        genome=genome_f,
        hidden_size_1=hidden_size_1,
        hidden_size_2=hidden_size_2,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained GA perceptron as a socket bot.")
    parser.add_argument("--genome-json", default="ga_results/best_genome.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--name", default="ga_runtime_bot")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = load_policy(Path(args.genome_json).expanduser().resolve())
    player_id = 0

    with socket.create_connection((args.host, args.port), timeout=15) as sock:
        sock.settimeout(None)
        reader = sock.makefile("r", encoding="utf-8", newline="\n")
        writer = sock.makefile("w", encoding="utf-8", newline="\n")
        writer.write(json.dumps({"type": "register", "name": args.name}) + "\n")
        writer.flush()

        for line in reader:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = payload.get("type")
            if msg_type == "hello":
                player_id = int(payload.get("player_id", 0) or 0)
                continue
            if msg_type != "tick" or player_id not in (1, 2):
                continue

            cmd = policy.act(payload, player_id=player_id)
            writer.write(json.dumps(cmd) + "\n")
            writer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
