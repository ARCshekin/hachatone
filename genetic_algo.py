#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# Make web_port internals importable when running from repo root.
REPO_ROOT = Path(__file__).resolve().parent
WEB_PORT_ROOT = REPO_ROOT / "game" / "web_port"
if str(WEB_PORT_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_PORT_ROOT))

from game.level_loader import get_levels_count, load_level
from game.models import PlayerCommand, Vec2
from game.simulation import GameSimulation

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


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


def _length(x: float, y: float) -> float:
    return math.hypot(x, y)


def _distance(a: Vec2, b: Vec2) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _v_sub(a: Vec2, b: Vec2) -> Vec2:
    return Vec2(a.x - b.x, a.y - b.y)


def _v_mul(v: Vec2, s: float) -> Vec2:
    return Vec2(v.x * s, v.y * s)


def _normalize(v: Vec2) -> Vec2:
    mag = math.hypot(v.x, v.y)
    if mag <= 1e-8:
        return Vec2()
    return Vec2(v.x / mag, v.y / mag)


@dataclass
class MatchStats:
    score_a: float
    score_b: float
    winner_id: int | None
    reason: str
    ticks: int


@dataclass
class EloMatchTask:
    i: int
    j: int
    level_index: int
    seed: int


@dataclass
class EloVsSimTask:
    i: int
    level_index: int
    seed: int
    bot_is_player1: bool


class PerceptronPolicy:
    def __init__(self, genome: list[float], hidden_size_1: int, hidden_size_2: int, device: str = "cpu") -> None:
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.genome = genome
        self.seq = 0
        self.device = device
        self._torch_mode = bool(torch is not None and device.startswith("cuda"))

        cursor = 0
        h1_in = hidden_size_1 * INPUT_SIZE
        self.w1 = genome[cursor : cursor + h1_in]
        cursor += h1_in

        self.b1 = genome[cursor : cursor + hidden_size_1]
        cursor += hidden_size_1

        h2_in = hidden_size_2 * hidden_size_1
        self.w2 = genome[cursor : cursor + h2_in]
        cursor += h2_in

        self.b2 = genome[cursor : cursor + hidden_size_2]
        cursor += hidden_size_2

        out_h2 = OUTPUT_SIZE * hidden_size_2
        self.w3 = genome[cursor : cursor + out_h2]
        cursor += out_h2

        self.b3 = genome[cursor : cursor + OUTPUT_SIZE]
        if self._torch_mode:
            self._init_torch_weights()

    def _init_torch_weights(self) -> None:
        assert torch is not None
        dev = torch.device(self.device)
        self.tw1 = torch.tensor(self.w1, dtype=torch.float32, device=dev).reshape(self.hidden_size_1, INPUT_SIZE)
        self.tb1 = torch.tensor(self.b1, dtype=torch.float32, device=dev)
        self.tw2 = torch.tensor(self.w2, dtype=torch.float32, device=dev).reshape(self.hidden_size_2, self.hidden_size_1)
        self.tb2 = torch.tensor(self.b2, dtype=torch.float32, device=dev)
        self.tw3 = torch.tensor(self.w3, dtype=torch.float32, device=dev).reshape(OUTPUT_SIZE, self.hidden_size_2)
        self.tb3 = torch.tensor(self.b3, dtype=torch.float32, device=dev)

    @staticmethod
    def genome_size(hidden_size_1: int, hidden_size_2: int) -> int:
        return (
            hidden_size_1 * INPUT_SIZE
            + hidden_size_1
            + hidden_size_2 * hidden_size_1
            + hidden_size_2
            + OUTPUT_SIZE * hidden_size_2
            + OUTPUT_SIZE
        )

    @staticmethod
    def random_genome(hidden_size_1: int, hidden_size_2: int, rng: random.Random) -> list[float]:
        size = PerceptronPolicy.genome_size(hidden_size_1, hidden_size_2)
        return [rng.uniform(-1.0, 1.0) for _ in range(size)]

    def _forward(self, features: list[float]) -> list[float]:
        if self._torch_mode:
            assert torch is not None
            x = torch.tensor(features, dtype=torch.float32, device=torch.device(self.device))
            h1 = torch.tanh(self.tw1 @ x + self.tb1)
            h2 = torch.tanh(self.tw2 @ h1 + self.tb2)
            out = self.tw3 @ h2 + self.tb3
            return [float(v) for v in out.tolist()]

        hidden1: list[float] = [0.0] * self.hidden_size_1
        for h in range(self.hidden_size_1):
            base = h * INPUT_SIZE
            s = self.b1[h]
            for i in range(INPUT_SIZE):
                s += self.w1[base + i] * features[i]
            hidden1[h] = math.tanh(s)

        hidden2: list[float] = [0.0] * self.hidden_size_2
        for h2 in range(self.hidden_size_2):
            base = h2 * self.hidden_size_1
            s = self.b2[h2]
            for h1 in range(self.hidden_size_1):
                s += self.w2[base + h1] * hidden1[h1]
            hidden2[h2] = math.tanh(s)

        out: list[float] = [0.0] * OUTPUT_SIZE
        for o in range(OUTPUT_SIZE):
            base = o * self.hidden_size_2
            s = self.b3[o]
            for h2 in range(self.hidden_size_2):
                s += self.w3[base + h2] * hidden2[h2]
            out[o] = s
        return out

    def act(self, sim: GameSimulation, player_id: int) -> PlayerCommand:
        self.seq += 1
        me = sim.players[player_id]
        enemy = sim.players[1 if player_id == 2 else 2]

        features = build_features(sim, player_id)
        logits = self._forward(features)

        move_x = math.tanh(logits[0])
        move_y = math.tanh(logits[1])
        move_len = _length(move_x, move_y)
        if move_len > 1.0:
            move_x /= move_len
            move_y /= move_len

        to_enemy_x = enemy.position.x - me.position.x
        to_enemy_y = enemy.position.y - me.position.y
        default_aim_len = _length(to_enemy_x, to_enemy_y)
        if default_aim_len <= 1e-6:
            default_aim_x, default_aim_y = 1.0, 0.0
        else:
            default_aim_x = to_enemy_x / default_aim_len
            default_aim_y = to_enemy_y / default_aim_len

        aim_x = 0.6 * math.tanh(logits[2]) + 0.4 * default_aim_x
        aim_y = 0.6 * math.tanh(logits[3]) + 0.4 * default_aim_y
        aim_len = _length(aim_x, aim_y)
        if aim_len <= 1e-6:
            aim_x, aim_y = 1.0, 0.0
        else:
            aim_x /= aim_len
            aim_y /= aim_len

        enemy_dist = default_aim_len
        shoot = _sigmoid(logits[4]) > 0.5 and enemy_dist < 320.0
        kick = _sigmoid(logits[5]) > 0.55 and enemy_dist < 30.0
        pickup = _sigmoid(logits[6]) > 0.55

        return PlayerCommand(
            seq=self.seq,
            move=Vec2(move_x, move_y),
            aim=Vec2(aim_x, aim_y),
            shoot=shoot,
            kick=kick,
            pickup=pickup,
            drop=False,
            throw=False,
            interact=False,
        )


class SimOpponentPolicy:
    """In-process policy mirroring sim_opponent baseline behavior."""

    def __init__(self) -> None:
        self.seq = 0

    @staticmethod
    def _nearest_pickup(sim: GameSimulation, you_pos: Vec2):
        best = None
        best_dist = 1e18
        for pickup in sim.pickups.values():
            if pickup.cooldown > 0.0:
                continue
            d = _distance(you_pos, pickup.position)
            if d < best_dist:
                best = pickup
                best_dist = d
        return best

    @staticmethod
    def _nearest_letterbox(sim: GameSimulation, you_pos: Vec2):
        best = None
        best_dist = 1e18
        for obstacle_id, cooldown in sim.letterbox_cooldowns.items():
            if cooldown > 0.0:
                continue
            obstacle = sim.obstacles.get(obstacle_id)
            if obstacle is None:
                continue
            d = _distance(you_pos, obstacle.center)
            if d < best_dist:
                best = obstacle
                best_dist = d
        return best

    def act(self, sim: GameSimulation, player_id: int) -> PlayerCommand:
        self.seq += 1
        me = sim.players[player_id]
        enemy = sim.players[1 if player_id == 2 else 2]
        to_enemy = _v_sub(enemy.position, me.position)
        aim = _normalize(to_enemy)
        distance = _distance(me.position, enemy.position)

        has_weapon = me.current_weapon is not None and me.current_weapon.ammo > 0
        weapon_type = str(me.current_weapon.weapon_type.value).lower() if me.current_weapon is not None else ""

        move = Vec2()
        shoot = False
        kick = False
        pickup = False

        if not has_weapon:
            nearest_pickup = self._nearest_pickup(sim, me.position)
            if nearest_pickup is not None:
                move = _normalize(_v_sub(nearest_pickup.position, me.position))
                aim = move if _distance(move, Vec2()) > 1e-6 else aim
                pickup = _distance(me.position, nearest_pickup.position) <= 24.0
            else:
                nearest_letterbox = self._nearest_letterbox(sim, me.position)
                if nearest_letterbox is not None:
                    move = _normalize(_v_sub(nearest_letterbox.center, me.position))
                    aim = move if _distance(move, Vec2()) > 1e-6 else aim
                    kick = _distance(me.position, nearest_letterbox.center) <= 24.0
                else:
                    move = aim
                    kick = distance <= 28.0
        else:
            desired_range = 170.0 if weapon_type == "revolver" else 105.0
            if distance > desired_range + 18.0:
                move = aim
            elif distance < desired_range - 35.0:
                move = _v_mul(aim, -1.0)
            else:
                move = Vec2(aim.y, -aim.x)
            shoot = bool(enemy.alive)
            kick = distance <= 28.0 and me.kick_cooldown <= 0.0

        if _distance(aim, Vec2()) <= 1e-6:
            aim = Vec2(1.0, 0.0)
        return PlayerCommand(
            seq=self.seq,
            move=move,
            aim=aim,
            shoot=shoot,
            kick=kick,
            pickup=pickup,
            drop=False,
            throw=False,
            interact=False,
        )


def _nearest_pickup(sim: GameSimulation, me: Vec2) -> tuple[float, float, float]:
    best_dx = 0.0
    best_dy = 0.0
    best_dist = 10_000.0
    for pickup in sim.pickups.values():
        if pickup.cooldown > 0.0:
            continue
        dx = pickup.position.x - me.x
        dy = pickup.position.y - me.y
        d = _length(dx, dy)
        if d < best_dist:
            best_dist = d
            best_dx = dx
            best_dy = dy
    if best_dist >= 9_999.0:
        return 0.0, 0.0, 1.0
    return best_dx, best_dy, best_dist


def _enemy_projectile_danger(sim: GameSimulation, player_id: int, me: Vec2) -> tuple[float, float, float]:
    best_dx = 0.0
    best_dy = 0.0
    best_dist = 10_000.0
    for projectile in sim.projectiles.values():
        if projectile.owner_id == player_id:
            continue
        dx = projectile.position.x - me.x
        dy = projectile.position.y - me.y
        d = _length(dx, dy)
        if d < best_dist:
            best_dist = d
            best_dx = dx
            best_dy = dy
    if best_dist >= 9_999.0:
        return 0.0, 0.0, 1.0
    return best_dx, best_dy, best_dist


def build_features(sim: GameSimulation, player_id: int) -> list[float]:
    me = sim.players[player_id]
    enemy = sim.players[1 if player_id == 2 else 2]

    dx = enemy.position.x - me.position.x
    dy = enemy.position.y - me.position.y
    dist = _length(dx, dy)
    dist_n = _clamp(dist / 500.0, 0.0, 1.0)
    rel_x = _clamp(dx / 500.0, -1.0, 1.0)
    rel_y = _clamp(dy / 500.0, -1.0, 1.0)

    me_speed = me.velocity.length()
    enemy_speed = enemy.velocity.length()

    pick_dx, pick_dy, pick_dist = _nearest_pickup(sim, me.position)
    proj_dx, proj_dy, proj_dist = _enemy_projectile_danger(sim, player_id, me.position)

    me_has_weapon = 1.0 if me.current_weapon is not None and me.current_weapon.ammo > 0 else 0.0
    enemy_has_weapon = 1.0 if enemy.current_weapon is not None and enemy.current_weapon.ammo > 0 else 0.0
    me_ammo = float(me.current_weapon.ammo) / 35.0 if me.current_weapon is not None else 0.0
    enemy_ammo = float(enemy.current_weapon.ammo) / 35.0 if enemy.current_weapon is not None else 0.0

    features = [
        rel_x,
        rel_y,
        dist_n,
        _clamp(me.velocity.x / 300.0, -1.0, 1.0),
        _clamp(me.velocity.y / 300.0, -1.0, 1.0),
        _clamp(enemy.velocity.x / 300.0, -1.0, 1.0),
        _clamp(enemy.velocity.y / 300.0, -1.0, 1.0),
        _clamp(me_speed / 300.0, 0.0, 1.0),
        _clamp(enemy_speed / 300.0, 0.0, 1.0),
        me_has_weapon,
        enemy_has_weapon,
        _clamp(me_ammo, 0.0, 1.0),
        _clamp(enemy_ammo, 0.0, 1.0),
        _clamp(me.shoot_cooldown / 0.5, 0.0, 1.0),
        _clamp(me.kick_cooldown / 1.0, 0.0, 1.0),
        _clamp(enemy.shoot_cooldown / 0.5, 0.0, 1.0),
        _clamp(pick_dx / 500.0, -1.0, 1.0),
        _clamp(pick_dy / 500.0, -1.0, 1.0),
        _clamp(pick_dist / 500.0, 0.0, 1.0),
        _clamp(proj_dist / 500.0, 0.0, 1.0),
    ]

    # Keep fixed input size.
    if len(features) != INPUT_SIZE:
        raise RuntimeError(f"Feature size mismatch: expected {INPUT_SIZE}, got {len(features)}")

    # Lightly blend projectile direction into existing distance features if projectile exists.
    if proj_dist < 0.999:
        features[18] = _clamp(proj_dx / 500.0, -1.0, 1.0)
        features[19] = _clamp(proj_dy / 500.0, -1.0, 1.0)

    return features


def run_match(
    genome_a: list[float],
    genome_b: list[float],
    *,
    hidden_size_1: int,
    hidden_size_2: int,
    device: str,
    level_index: int,
    seed: int,
    round_time_limit_seconds: float,
    ldtk_path: Path,
) -> MatchStats:
    level = load_level(ldtk_path, level_index=level_index, seed=seed)
    sim = GameSimulation(level=level, seed=seed, round_time_limit_seconds=round_time_limit_seconds)

    policy_a = PerceptronPolicy(genome_a, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2, device=device)
    policy_b = PerceptronPolicy(genome_b, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2, device=device)

    while not sim.is_finished():
        commands = {
            1: policy_a.act(sim, player_id=1),
            2: policy_b.act(sim, player_id=2),
        }
        sim.step(commands)

    result = sim.result
    if result is None:
        return MatchStats(score_a=1.0, score_b=1.0, winner_id=None, reason="unknown", ticks=sim.tick)

    if result.winner_id == 1:
        return MatchStats(score_a=3.0, score_b=0.0, winner_id=1, reason=result.reason, ticks=sim.tick)
    if result.winner_id == 2:
        return MatchStats(score_a=0.0, score_b=3.0, winner_id=2, reason=result.reason, ticks=sim.tick)
    return MatchStats(score_a=1.0, score_b=1.0, winner_id=None, reason=result.reason, ticks=sim.tick)


def _run_match_task(
    task: EloMatchTask,
    population: list[list[float]],
    hidden_size_1: int,
    hidden_size_2: int,
    device: str,
    round_time_limit_seconds: float,
    ldtk_path: Path,
) -> tuple[int, int, MatchStats]:
    stats = run_match(
        population[task.i],
        population[task.j],
        hidden_size_1=hidden_size_1,
        hidden_size_2=hidden_size_2,
        device=device,
        level_index=task.level_index,
        seed=task.seed,
        round_time_limit_seconds=round_time_limit_seconds,
        ldtk_path=ldtk_path,
    )
    return task.i, task.j, stats


def run_match_vs_sim(
    genome: list[float],
    *,
    hidden_size_1: int,
    hidden_size_2: int,
    device: str,
    level_index: int,
    seed: int,
    round_time_limit_seconds: float,
    ldtk_path: Path,
    bot_is_player1: bool,
) -> MatchStats:
    level = load_level(ldtk_path, level_index=level_index, seed=seed)
    sim = GameSimulation(level=level, seed=seed, round_time_limit_seconds=round_time_limit_seconds)
    ga_policy = PerceptronPolicy(genome, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2, device=device)
    sim_policy = SimOpponentPolicy()

    ga_player_id = 1 if bot_is_player1 else 2
    sim_player_id = 2 if bot_is_player1 else 1

    while not sim.is_finished():
        commands = {
            ga_player_id: ga_policy.act(sim, player_id=ga_player_id),
            sim_player_id: sim_policy.act(sim, player_id=sim_player_id),
        }
        sim.step(commands)

    result = sim.result
    if result is None:
        return MatchStats(score_a=1.0, score_b=1.0, winner_id=None, reason="unknown", ticks=sim.tick)
    if result.winner_id == ga_player_id:
        return MatchStats(score_a=3.0, score_b=0.0, winner_id=1, reason=result.reason, ticks=sim.tick)
    if result.winner_id == sim_player_id:
        return MatchStats(score_a=0.0, score_b=3.0, winner_id=2, reason=result.reason, ticks=sim.tick)
    return MatchStats(score_a=1.0, score_b=1.0, winner_id=None, reason=result.reason, ticks=sim.tick)


def _run_match_vs_sim_task(
    task: EloVsSimTask,
    population: list[list[float]],
    hidden_size_1: int,
    hidden_size_2: int,
    device: str,
    round_time_limit_seconds: float,
    ldtk_path: Path,
) -> tuple[int, MatchStats]:
    stats = run_match_vs_sim(
        population[task.i],
        hidden_size_1=hidden_size_1,
        hidden_size_2=hidden_size_2,
        device=device,
        level_index=task.level_index,
        seed=task.seed,
        round_time_limit_seconds=round_time_limit_seconds,
        ldtk_path=ldtk_path,
        bot_is_player1=task.bot_is_player1,
    )
    return task.i, stats


def crossover(parent_a: list[float], parent_b: list[float], rng: random.Random) -> list[float]:
    child: list[float] = []
    for wa, wb in zip(parent_a, parent_b):
        if rng.random() < 0.5:
            child.append(wa)
        else:
            child.append(wb)
    return child


def mutate(genome: list[float], rng: random.Random, mutation_rate: float, mutation_std: float) -> None:
    for i in range(len(genome)):
        if rng.random() < mutation_rate:
            genome[i] += rng.gauss(0.0, mutation_std)
            genome[i] = _clamp(genome[i], -3.0, 3.0)


def _progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    ratio = _clamp(done / total, 0.0, 1.0)
    filled = int(ratio * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _sparkline(values: list[float], width: int = 42) -> str:
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    if len(values) <= width:
        sampled = values
    else:
        sampled = []
        step = len(values) / width
        for i in range(width):
            idx = min(len(values) - 1, int(i * step))
            sampled.append(values[idx])
    vmin = min(sampled)
    vmax = max(sampled)
    if abs(vmax - vmin) <= 1e-9:
        return blocks[0] * len(sampled)
    chars: list[str] = []
    for v in sampled:
        t = (v - vmin) / (vmax - vmin)
        chars.append(blocks[min(len(blocks) - 1, int(t * (len(blocks) - 1)))])
    return "".join(chars)


def _actual_score_for_elo(match_stats: MatchStats, as_player_a: bool) -> float:
    if match_stats.winner_id is None:
        return 0.5
    if as_player_a:
        return 1.0 if match_stats.winner_id == 1 else 0.0
    return 1.0 if match_stats.winner_id == 2 else 0.0


def _make_rating_pairings(
    active_indices: list[int],
    ratings: list[float],
    rng: random.Random,
    pairing_window: int,
) -> list[tuple[int, int]]:
    unpaired = list(active_indices)
    rng.shuffle(unpaired)
    pairs: list[tuple[int, int]] = []
    while len(unpaired) >= 2:
        i = unpaired.pop()
        unpaired.sort(key=lambda idx: abs(ratings[idx] - ratings[i]))
        window = max(1, min(pairing_window, len(unpaired)))
        pick = rng.randrange(window)
        j = unpaired.pop(pick)
        pairs.append((i, j))
    return pairs


def elo_tournament_scores(
    population: list[list[float]],
    *,
    hidden_size_1: int,
    hidden_size_2: int,
    elo_rounds: int,
    sim_round_fraction: float,
    cull_after_fraction: float,
    cull_fraction: float,
    elo_k_factor: float,
    pairing_window: int,
    round_time_limit_seconds: float,
    ldtk_path: Path,
    levels_count: int,
    workers: int,
    device: str,
    rng: random.Random,
    generation_idx: int,
    generations_total: int,
) -> list[float]:
    n = len(population)
    ratings = [1400.0 for _ in range(n)]
    sim_rounds = int(round(elo_rounds * _clamp(sim_round_fraction, 0.0, 1.0)))
    population_rounds = max(0, elo_rounds - sim_rounds)
    round_plan = [True] * sim_rounds + [False] * population_rounds
    rng.shuffle(round_plan)
    cull_after_round = max(1, int(math.ceil(elo_rounds * _clamp(cull_after_fraction, 0.0, 1.0))))
    cull_after_round = min(cull_after_round, elo_rounds)
    survivors_count = max(2, n - int(n * _clamp(cull_fraction, 0.0, 0.95)))
    survivors_count = min(survivors_count, n)

    total_matches = 0
    for r_idx, is_sim_round in enumerate(round_plan):
        participants = n if r_idx < cull_after_round else survivors_count
        total_matches += participants if is_sim_round else participants // 2

    done_matches = 0
    start_ts = time.monotonic()
    sim_rating = 1400.0
    active_indices = list(range(n))
    culled = False

    with ProcessPoolExecutor(max_workers=workers) if workers > 1 else None as executor:
        for round_idx, is_sim_round in enumerate(round_plan):
            if not is_sim_round:
                pairs = _make_rating_pairings(active_indices, ratings, rng=rng, pairing_window=pairing_window)
                tasks = [
                    EloMatchTask(
                        i=i,
                        j=j,
                        level_index=rng.randrange(levels_count),
                        seed=rng.randrange(1_000_000_000),
                    )
                    for i, j in pairs
                ]

                if workers <= 1:
                    results = [
                        _run_match_task(
                            task=t,
                            population=population,
                            hidden_size_1=hidden_size_1,
                            hidden_size_2=hidden_size_2,
                            device=device,
                            round_time_limit_seconds=round_time_limit_seconds,
                            ldtk_path=ldtk_path,
                        )
                        for t in tasks
                    ]
                else:
                    assert executor is not None
                    futures = [
                        executor.submit(
                            _run_match_task,
                            t,
                            population,
                            hidden_size_1,
                            hidden_size_2,
                            device,
                            round_time_limit_seconds,
                            ldtk_path,
                        )
                        for t in tasks
                    ]
                    results = [f.result() for f in futures]

                for i, j, match_stats in results:
                    ra = ratings[i]
                    rb = ratings[j]
                    expected_a = 1.0 / (1.0 + (10.0 ** ((rb - ra) / 400.0)))
                    expected_b = 1.0 - expected_a
                    actual_a = _actual_score_for_elo(match_stats, as_player_a=True)
                    actual_b = _actual_score_for_elo(match_stats, as_player_a=False)
                    ratings[i] = ra + elo_k_factor * (actual_a - expected_a)
                    ratings[j] = rb + elo_k_factor * (actual_b - expected_b)

                    done_matches += 1
                    if done_matches == 1 or done_matches == total_matches or done_matches % max(1, total_matches // 30) == 0:
                        elapsed = max(1e-6, time.monotonic() - start_ts)
                        speed = done_matches / elapsed
                        eta = (total_matches - done_matches) / max(1e-6, speed)
                        bar = _progress_bar(done_matches, total_matches)
                        print(
                            f"\rgen {generation_idx}/{generations_total} "
                            f"{bar} {done_matches}/{total_matches} "
                            f"{speed:.2f} m/s ETA {eta:.1f}s",
                            end="",
                            flush=True,
                        )
            else:
                tasks_vs_sim = [
                    EloVsSimTask(
                        i=i,
                        level_index=rng.randrange(levels_count),
                        seed=rng.randrange(1_000_000_000),
                        bot_is_player1=((round_idx + i + generation_idx) % 2 == 0),
                    )
                    for i in active_indices
                ]

                if workers <= 1:
                    results_vs_sim = [
                        _run_match_vs_sim_task(
                            task=t,
                            population=population,
                            hidden_size_1=hidden_size_1,
                            hidden_size_2=hidden_size_2,
                            device=device,
                            round_time_limit_seconds=round_time_limit_seconds,
                            ldtk_path=ldtk_path,
                        )
                        for t in tasks_vs_sim
                    ]
                else:
                    assert executor is not None
                    futures = [
                        executor.submit(
                            _run_match_vs_sim_task,
                            t,
                            population,
                            hidden_size_1,
                            hidden_size_2,
                            device,
                            round_time_limit_seconds,
                            ldtk_path,
                        )
                        for t in tasks_vs_sim
                    ]
                    results_vs_sim = [f.result() for f in futures]

                for i, match_stats in results_vs_sim:
                    ra = ratings[i]
                    expected_a = 1.0 / (1.0 + (10.0 ** ((sim_rating - ra) / 400.0)))
                    actual_a = _actual_score_for_elo(match_stats, as_player_a=True)
                    ratings[i] = ra + elo_k_factor * (actual_a - expected_a)

                    done_matches += 1
                    if done_matches == 1 or done_matches == total_matches or done_matches % max(1, total_matches // 30) == 0:
                        elapsed = max(1e-6, time.monotonic() - start_ts)
                        speed = done_matches / elapsed
                        eta = (total_matches - done_matches) / max(1e-6, speed)
                        bar = _progress_bar(done_matches, total_matches)
                        print(
                            f"\rgen {generation_idx}/{generations_total} "
                            f"{bar} {done_matches}/{total_matches} "
                            f"{speed:.2f} m/s ETA {eta:.1f}s",
                            end="",
                            flush=True,
                        )

            if not culled and (round_idx + 1) >= cull_after_round and len(active_indices) > survivors_count:
                ranked = sorted(active_indices, key=lambda idx: ratings[idx], reverse=True)
                kept = ranked[:survivors_count]
                dropped = ranked[survivors_count:]
                active_indices = kept
                culled = True
                dropped_avg = sum(ratings[i] for i in dropped) / max(1, len(dropped))
                print(
                    f"\n[gen {generation_idx:03d}] cull applied: kept={len(kept)} dropped={len(dropped)} "
                    f"(dropped avg elo={dropped_avg:.1f})"
                )
    print()
    return ratings


def evolve(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)

    ldtk_path = Path(args.ldtk_path).expanduser().resolve()
    if not ldtk_path.exists():
        raise FileNotFoundError(f"LDTK file not found: {ldtk_path}")
    levels_count = get_levels_count(ldtk_path)
    if levels_count <= 0:
        raise RuntimeError("No levels in LDTK file.")

    population = [
        PerceptronPolicy.random_genome(args.hidden_size_1, args.hidden_size_2, rng)
        for _ in range(args.population_size)
    ]

    history: list[dict] = []
    best_genome = population[0]
    best_score = -1e18
    run_started = time.monotonic()
    generation_best_dir = Path(args.generation_best_dir).expanduser().resolve()
    generation_best_dir.mkdir(parents=True, exist_ok=True)

    for generation in range(1, args.generations + 1):
        scores = elo_tournament_scores(
            population,
            hidden_size_1=args.hidden_size_1,
            hidden_size_2=args.hidden_size_2,
            elo_rounds=args.elo_rounds,
            sim_round_fraction=args.sim_round_fraction,
            cull_after_fraction=args.cull_after_fraction,
            cull_fraction=args.cull_fraction,
            elo_k_factor=args.elo_k_factor,
            pairing_window=args.pairing_window,
            round_time_limit_seconds=args.round_time_limit_seconds,
            ldtk_path=ldtk_path,
            levels_count=levels_count,
            workers=args.workers,
            device=args.device,
            rng=rng,
            generation_idx=generation,
            generations_total=args.generations,
        )
        ranking = sorted(range(len(population)), key=lambda idx: scores[idx], reverse=True)
        best_idx, second_idx = ranking[0], ranking[1]
        gen_best_score = scores[best_idx]
        gen_avg_score = sum(scores) / max(1, len(scores))

        if gen_best_score > best_score:
            best_score = gen_best_score
            best_genome = list(population[best_idx])

        history.append(
            {
                "generation": generation,
                "best_score": gen_best_score,
                "avg_score": gen_avg_score,
                "best_index": best_idx,
                "second_index": second_idx,
            }
        )
        generation_best_payload = {
            "generation": generation,
            "generation_best_score": gen_best_score,
            "generation_avg_score": gen_avg_score,
            "hidden_size_1": args.hidden_size_1,
            "hidden_size_2": args.hidden_size_2,
            "metric": "elo_rating",
            "genome": list(population[best_idx]),
        }
        generation_best_path = generation_best_dir / f"gen_{generation:03d}_best.json"
        generation_best_path.write_text(
            json.dumps(generation_best_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        top_k = min(args.show_top_k, len(ranking))
        top_str = ", ".join(f"#{idx}:{scores[idx]:.1f}" for idx in ranking[:top_k])
        print(f"[gen {generation:03d}] best={gen_best_score:.2f} avg={gen_avg_score:.2f} | top {top_k}: {top_str}")
        best_curve = _sparkline([float(h["best_score"]) for h in history])
        avg_curve = _sparkline([float(h["avg_score"]) for h in history])
        print(f"  best trend: {best_curve}")
        print(f"  avg  trend: {avg_curve}")

        parent_a = population[best_idx]
        parent_b = population[second_idx]
        next_population: list[list[float]] = [list(parent_a), list(parent_b)]
        while len(next_population) < args.population_size:
            child = crossover(parent_a, parent_b, rng)
            mutate(
                child,
                rng,
                mutation_rate=args.mutation_rate,
                mutation_std=args.mutation_std,
            )
            next_population.append(child)
        population = next_population

    elapsed_total = time.monotonic() - run_started

    return {
        "seed": args.seed,
        "population_size": args.population_size,
        "generations": args.generations,
        "hidden_size_1": args.hidden_size_1,
        "hidden_size_2": args.hidden_size_2,
        "elo_rounds": args.elo_rounds,
        "sim_round_fraction": args.sim_round_fraction,
        "cull_after_fraction": args.cull_after_fraction,
        "cull_fraction": args.cull_fraction,
        "elo_k_factor": args.elo_k_factor,
        "pairing_window": args.pairing_window,
        "workers": args.workers,
        "device": args.device,
        "round_time_limit_seconds": args.round_time_limit_seconds,
        "best_score": best_score,
        "best_genome": best_genome,
        "history": history,
        "ldtk_path": str(ldtk_path),
        "elapsed_seconds": elapsed_total,
        "generation_best_dir": str(generation_best_dir),
    }


def save_history_csv(history: list[dict], output_path: Path) -> None:
    lines = ["generation,best_score,avg_score,best_index,second_index"]
    for row in history:
        lines.append(
            f"{int(row['generation'])},{float(row['best_score'])},"
            f"{float(row['avg_score'])},{int(row['best_index'])},{int(row['second_index'])}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_ldtk = REPO_ROOT / "game" / "web_port" / "assets" / "levels" / "test_ldtk_project.ldtk"
    parser = argparse.ArgumentParser(
        description="Genetic algorithm for GAICA bots with two-hidden-layer perceptron policy."
    )
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--hidden-size-1", type=int, default=60)
    parser.add_argument("--hidden-size-2", type=int, default=40)
    parser.add_argument("--elo-rounds", type=int, default=30, help="how many ELO pairing rounds per generation")
    parser.add_argument(
        "--sim-round-fraction",
        type=float,
        default=0.5,
        help="fraction of ELO rounds evaluated vs sim_opponent (0.0..1.0)",
    )
    parser.add_argument(
        "--cull-after-fraction",
        type=float,
        default=0.3,
        help="after this fraction of rounds, remove weakest bots once (0.0..1.0)",
    )
    parser.add_argument(
        "--cull-fraction",
        type=float,
        default=0.5,
        help="fraction of weakest active bots removed at cull step (0.0..0.95)",
    )
    parser.add_argument("--elo-k-factor", type=float, default=24.0, help="K-factor for ELO updates")
    parser.add_argument("--pairing-window", type=int, default=10, help="random pick among nearest opponents by rating")
    parser.add_argument("--mutation-rate", type=float, default=0.08)
    parser.add_argument("--mutation-std", type=float, default=0.22)
    parser.add_argument("--round-time-limit-seconds", type=float, default=40.0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--device", default="cpu", help="cpu or cuda (requires torch + CUDA)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ldtk-path", default=str(default_ldtk))
    parser.add_argument("--output", default=str(REPO_ROOT / "ga_results" / "best_genome.json"))
    parser.add_argument("--history-csv", default=str(REPO_ROOT / "ga_results" / "history.csv"))
    parser.add_argument(
        "--generation-best-dir",
        default=str(REPO_ROOT / "ga_results" / "generation_bests"),
        help="directory where the best individual of every generation is stored",
    )
    parser.add_argument("--show-top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.population_size < 4:
        raise ValueError("population-size must be >= 4")
    if args.generations < 1:
        raise ValueError("generations must be >= 1")
    if args.hidden_size_1 < 2:
        raise ValueError("hidden-size-1 must be >= 2")
    if args.hidden_size_2 < 2:
        raise ValueError("hidden-size-2 must be >= 2")
    if args.elo_rounds < 1:
        raise ValueError("elo-rounds must be >= 1")
    if args.sim_round_fraction < 0.0 or args.sim_round_fraction > 1.0:
        raise ValueError("sim-round-fraction must be in [0.0, 1.0]")
    if args.cull_after_fraction < 0.0 or args.cull_after_fraction > 1.0:
        raise ValueError("cull-after-fraction must be in [0.0, 1.0]")
    if args.cull_fraction < 0.0 or args.cull_fraction > 0.95:
        raise ValueError("cull-fraction must be in [0.0, 0.95]")
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    if args.device.startswith("cuda"):
        if torch is None:
            raise ValueError("device=cuda requested but torch is not installed")
        if not torch.cuda.is_available():
            raise ValueError("device=cuda requested but CUDA is not available")
        if args.workers > 1:
            print("Warning: device=cuda with workers>1 can reduce performance; forcing workers=1")
            args.workers = 1

    results = evolve(args)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    history_csv_path = Path(args.history_csv).expanduser().resolve()
    history_csv_path.parent.mkdir(parents=True, exist_ok=True)
    save_history_csv(results["history"], history_csv_path)
    print(f"Saved best genome to: {output_path}")
    print(f"Saved training history to: {history_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
