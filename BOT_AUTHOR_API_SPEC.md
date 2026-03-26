# API бота и раннера

Версия документа: `2026-03-16`

Документ описывает основной контракт, под который нужно писать новых ботов для GAICA. Он совпадает с `web_port` и с локальным раннером для участника.

## 1. Запуск процесса

- Платформа принимает `.zip` архив.
- В корне архива должен лежать `main.py`.
- Процесс бота запускается так:

```bash
python3 main.py <host> <port>
```

- Те же параметры дублируются в переменных окружения:
  - `GAICA_BOT_HOST`
  - `GAICA_BOT_PORT`
  - `BOT_HOST`
  - `BOT_PORT`
  - `GAICA_BOT_SLOT` (`1` или `2`)
- Бот должен открыть TCP-соединение с `host:port`.
- Транспорт: `UTF-8`, одно JSON-сообщение на строку, разделитель `\n`.

## 2. Последовательность сообщений

Для каждого матча раннер отправляет:

1. `hello` один раз после подключения.
2. Для каждого раунда:
   - `round_start`
   - много сообщений `tick`
   - `round_end`

Раннер ждёт от бота только сообщения `command`.

## 3. Лимиты времени

- Максимальное ожидание нового ответа на одном тике: `1` секунда.
- Суммарное ожидание ответов одного бота за весь матч: `60` секунд.
- Если бот не ответил вовремя на конкретном тике, матч продолжается с последней валидной командой.
- Если бот исчерпал суммарный лимит ожидания, раннер завершает процесс бота и помечает ошибку.
- На площадке дополнительно действует отдельный лимит процесса на раунд:
  - `max_cpu_seconds`: по умолчанию `120`
  - `round_timeout_seconds`: по умолчанию `180`

## 4. Входящие сообщения

### 4.1 `hello`

Отправляется сразу после подключения.

```json
{
  "type": "hello",
  "player_id": 1,
  "tick_rate": 30
}
```

- `player_id` — ваш слот в текущем матче.
- `tick_rate` — частота симуляции.

### 4.2 `round_start`

Отправляется в начале каждого раунда.

```json
{
  "type": "round_start",
  "player_id": 1,
  "enemy_id": 2,
  "tick_rate": 30,
  "level": {
    "identifier": "Level_2",
    "width": 384,
    "height": 384,
    "floor_tiles": [],
    "top_tiles": [],
    "small_tiles": [],
    "player_spawns": [[32.0, 32.0], [288.0, 288.0]]
  },
  "series": {
    "enabled": true,
    "round": 2,
    "total_rounds": 4,
    "completed_rounds": 1,
    "score": {"1": 1, "2": 0},
    "final_result": null
  }
}
```

Полезные поля:

- `level` — полная статическая карта раунда.
- `floor_tiles`, `top_tiles`, `small_tiles` — исходные тайлы LDtk.
- `player_spawns` — точки старта на карте.
- `series` — текущее состояние серии из нескольких раундов.

### 4.3 `tick`

Отправляется на каждом тике.

```json
{
  "type": "tick",
  "tick": 241,
  "time_seconds": 8.033333333333333,
  "you": {
    "id": 1,
    "position": [96.0, 128.0],
    "facing": [1.0, 0.0],
    "alive": true,
    "color": "#fe930a",
    "character": "orange",
    "weapon": {"type": "Revolver", "ammo": 8},
    "shoot_cooldown": 0.0,
    "kick_cooldown": 0.0,
    "stun_remaining": 0.0
  },
  "enemy": {
    "id": 2,
    "position": [224.0, 128.0],
    "facing": [-1.0, 0.0],
    "alive": true,
    "color": "#28fe0b",
    "character": "lime",
    "weapon": null,
    "shoot_cooldown": 0.0,
    "kick_cooldown": 0.0,
    "stun_remaining": 0.0
  },
  "snapshot": {
    "status": "running",
    "tick": 241,
    "time_seconds": 8.033333333333333,
    "time_limit_seconds": 180.0,
    "result": null,
    "level": {},
    "players": [],
    "pickups": [],
    "projectiles": [],
    "obstacles": [],
    "breakables": [],
    "effects": [],
    "debris": [],
    "letterboxes": []
  }
}
```

Что важно:

- `tick` — номер тика внутри матча.
- `time_seconds` — игровое время текущего раунда.
- `you` и `enemy` — быстрый доступ к двум игрокам.
- `snapshot` — полное состояние мира на текущем тике.
- Позиции, направления и скорости передаются массивами `[x, y]`.
- Карта для бота не скрыта: бот видит полное состояние мира.

Структура `snapshot`:

- `players` — оба игрока с положением, оружием, кулдаунами и состоянием оглушения.
- `pickups` — лежащее оружие.
- `projectiles` — активные пули.
- `obstacles` — все препятствия, включая стены, двери и стекло.
- `breakables` — текущее состояние разрушаемых объектов.
- `letterboxes` — почтовые ящики и их кулдаун.
- `effects` — визуально-игровые события текущего тика.
- `result` — `null`, пока раунд не завершён.

Пример полезного фрагмента `snapshot`:

```json
{
  "players": [
    {
      "id": 1,
      "position": [96.0, 128.0],
      "facing": [1.0, 0.0],
      "alive": true,
      "weapon": {"type": "Revolver", "ammo": 8},
      "shoot_cooldown": 0.0,
      "kick_cooldown": 0.0,
      "stun_remaining": 0.0
    },
    {
      "id": 2,
      "position": [224.0, 128.0],
      "facing": [-1.0, 0.0],
      "alive": true,
      "weapon": null,
      "shoot_cooldown": 0.0,
      "kick_cooldown": 0.0,
      "stun_remaining": 0.0
    }
  ],
  "pickups": [
    {
      "id": 3,
      "type": "Uzi",
      "ammo": 25,
      "position": [160.0, 128.0],
      "cooldown": 0.0
    }
  ],
  "projectiles": [
    {
      "id": 9,
      "owner": 1,
      "type": "Revolver",
      "position": [132.0, 128.0],
      "velocity": [500.0, 0.0],
      "remaining_life": 1.83
    }
  ],
  "breakables": [
    {
      "id": 15,
      "obstacle_id": 44,
      "variant": "Glass",
      "current": 0.0,
      "threshold": 0.1,
      "alive": true,
      "center": [192.0, 128.0],
      "half_size": [2.0, 32.0]
    }
  ]
}
```

### 4.4 `round_end`

Отправляется после окончания раунда.

```json
{
  "type": "round_end",
  "result": {
    "winner_id": 1,
    "reason": "elimination",
    "duration_seconds": 21.6,
    "series_round": 2,
    "series_total_rounds": 4,
    "series_score": {"1": 2, "2": 0},
    "series_finished": false,
    "level_identifier": "Level_2"
  }
}
```

- `winner_id` равен `1`, `2` или `null`.
- `reason` обычно `elimination`, `time_limit` или `series_score`.
- Если серия уже закончилась, `series_finished = true`.

Пример ничьей:

```json
{
  "type": "round_end",
  "result": {
    "winner_id": null,
    "reason": "time_limit",
    "duration_seconds": 180.0,
    "series_round": 3,
    "series_total_rounds": 4,
    "series_score": {"1": 1, "2": 1},
    "series_finished": false,
    "level_identifier": "Level_1"
  }
}
```

## 5. Исходящее сообщение `command`

Бот может отправлять только JSON-объект с действиями.

```json
{
  "type": "command",
  "seq": 241,
  "move": [0.0, -1.0],
  "aim": [1.0, 0.0],
  "shoot": false,
  "kick": false,
  "pickup": false,
  "drop": false,
  "throw": false,
  "interact": false
}
```

Правила:

- `seq` — номер вашей команды. Для новых ботов используйте монотонно растущий счётчик.
- `move` — направление движения.
- `aim` — направление взгляда и выстрела.
- Компоненты `move` и `aim` ограничиваются диапазоном `[-1, 1]`.
- Если `aim` отсутствует или равен нулю, раннер подставит `move`, а затем `(1, 0)`.
- Все флаги действий должны быть `bool`.
- Неизвестные поля игнорируются.

Смысл действий:

- `shoot` — выстрел, если у игрока есть оружие, есть патроны и кулдаун закончился.
- `kick` — удар ногой.
- `pickup` — подобрать ближайшее оружие.
- `drop` — уронить текущее оружие.
- `throw` — бросить текущее оружие вперёд.
- `interact` — старый совместимый режим:
  - если рядом есть оружие, подобрать его;
  - иначе бросить текущее оружие.

Если в одной команде выставить несколько действий, симуляция применяет их в таком порядке:

1. `kick`
2. `pickup`
3. `throw`
4. `drop`
5. `interact`
6. движение
7. стрельба

Практически лучше отправлять не более одного действия из группы `pickup/drop/throw/interact` на тик.

Примеры:

Подбор оружия:

```json
{
  "type": "command",
  "seq": 88,
  "move": [0.0, 0.0],
  "aim": [1.0, 0.0],
  "shoot": false,
  "kick": false,
  "pickup": true,
  "drop": false,
  "throw": false,
  "interact": false
}
```

Ближний бой:

```json
{
  "type": "command",
  "seq": 119,
  "move": [1.0, 0.0],
  "aim": [1.0, 0.0],
  "shoot": false,
  "kick": true,
  "pickup": false,
  "drop": false,
  "throw": false,
  "interact": false
}
```

Бросок оружия:

```json
{
  "type": "command",
  "seq": 203,
  "move": [0.0, 0.0],
  "aim": [-1.0, 0.0],
  "shoot": false,
  "kick": false,
  "pickup": false,
  "drop": false,
  "throw": true,
  "interact": false
}
```

## 6. Ошибки и отказоустойчивость

- Невалидный JSON считается ошибкой бота.
- Если бот завершился или разорвал соединение, матч продолжится без него.
- Диагностику пишите только в `stderr`.
- Не используйте `stdout` для логов: он зарезервирован под протокол.

## 7. Минимальный шаблон

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import socket
import sys


def decide(msg: dict) -> dict | None:
    if msg.get("type") != "tick":
        return None
    you = msg.get("you") or {}
    enemy = msg.get("enemy") or {}
    you_pos = you.get("position") or [0.0, 0.0]
    enemy_pos = enemy.get("position") or [0.0, 0.0]
    dx = float(enemy_pos[0]) - float(you_pos[0])
    dy = float(enemy_pos[1]) - float(you_pos[1])
    return {
        "type": "command",
        "seq": int(msg.get("tick", 0)),
        "move": [0.0, 0.0],
        "aim": [dx, dy],
        "shoot": bool(enemy.get("alive", False)),
        "kick": False,
        "pickup": False,
        "drop": False,
        "throw": False,
        "interact": False,
    }


def main() -> int:
    host = sys.argv[1]
    port = int(sys.argv[2])
    with socket.create_connection((host, port), timeout=15) as sock:
        reader = sock.makefile("r", encoding="utf-8", newline="\n")
        writer = sock.makefile("w", encoding="utf-8", newline="\n")
        for line in reader:
            raw = line.strip()
            if not raw:
                continue
            msg = json.loads(raw)
            command = decide(msg)
            if command is None:
                continue
            writer.write(json.dumps(command, ensure_ascii=False) + "\n")
            writer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## 8. Совместимость со старыми ботами

Платформа всё ещё умеет отправлять старый `state`-payload для legacy-ботов, но новые решения нужно писать под socket-схему `hello -> round_start -> tick -> round_end`. Именно её используют `web_port`, локальный раннер и примеры ботов.

Пример минимального legacy `state`-payload:

```json
{
  "type": "state",
  "protocol_version": "2.0",
  "tick": 245,
  "series_round": 2,
  "series_total_rounds": 4,
  "map_id": "Level_3",
  "map_index": 3,
  "time": {"elapsed": 8.16, "remaining": 171.84},
  "self": {
    "id": 1,
    "alive": true,
    "position": {"x": 96.0, "y": 128.0},
    "facing": {"x": 1.0, "y": 0.0},
    "weapon": {"id": "p1-revolver", "kind": "ranged", "variant": "revolver", "ammo": 6}
  },
  "enemy": {
    "id": 2,
    "alive": true,
    "position": {"x": 224.0, "y": 128.0},
    "facing": {"x": -1.0, "y": 0.0},
    "weapon": null
  },
  "map": {
    "id": "Level_3",
    "index": 3,
    "width": 384.0,
    "height": 384.0,
    "floor_tiles": [],
    "top_tiles": [],
    "small_tiles": [],
    "player_spawns": [[32.0, 32.0], [288.0, 288.0]],
    "obstacles": []
  },
  "pickups": [],
  "projectiles": [],
  "interactives": [],
  "events": [],
  "rules": {
    "max_round_seconds": 180.0,
    "max_round_cpu_time_seconds": 120.0,
    "max_tick_response_wait_seconds": 1.0,
    "max_match_response_wait_seconds": 60.0,
    "tick_rate": 30
  }
}
```
