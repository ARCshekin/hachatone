#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENOME_JSON="${1:-$ROOT_DIR/ga_results/best_genome.json}"
BOT_PORT="${BOT_PORT:-9001}"
WEB_PORT="${WEB_PORT:-8080}"

cleanup() {
  if [[ -n "${P_SERVER:-}" ]]; then kill "$P_SERVER" 2>/dev/null || true; fi
  if [[ -n "${P_GA:-}" ]]; then kill "$P_GA" 2>/dev/null || true; fi
  if [[ -n "${P_SIM:-}" ]]; then kill "$P_SIM" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

echo "Genome: $GENOME_JSON"
echo "Starting web server on ports bot=$BOT_PORT web=$WEB_PORT ..."
cd "$ROOT_DIR/game/web_port"
python3 main.py --bot-port "$BOT_PORT" --web-port "$WEB_PORT" > "$ROOT_DIR/ga_results/web_port.log" 2>&1 &
P_SERVER=$!

sleep 1
echo "Starting GA runtime bot..."
cd "$ROOT_DIR"
python3 ga_runtime_bot.py --genome-json "$GENOME_JSON" --host 127.0.0.1 --port "$BOT_PORT" --name "ga_champion" \
  > "$ROOT_DIR/ga_results/ga_bot.log" 2>&1 &
P_GA=$!

echo "Starting sim_opponent..."
python3 sim_opponent.py 127.0.0.1 "$BOT_PORT" > "$ROOT_DIR/ga_results/sim_opponent.log" 2>&1 &
P_SIM=$!

echo "Match running. Open: http://127.0.0.1:$WEB_PORT/"
echo "Logs:"
echo "  ga_results/web_port.log"
echo "  ga_results/ga_bot.log"
echo "  ga_results/sim_opponent.log"
echo "Press Ctrl+C to stop all processes."

wait "$P_SERVER"
