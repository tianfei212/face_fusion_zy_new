#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "/home/ubuntu/miniconda3/etc/profile.d/conda.sh" ]; then
  set +u
  source "/home/ubuntu/miniconda3/etc/profile.d/conda.sh"
  conda activate ai_face_change
  set -u
fi

release_port() {
  local port="$1"
  if command -v fuser >/dev/null 2>&1; then
    if fuser -n tcp "$port" >/dev/null 2>&1; then
      echo "释放端口 $port"
      fuser -k -n tcp "$port" >/dev/null 2>&1 || true
    fi
    return
  fi

  local pids
  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "释放端口 $port (PID: $pids)"
    kill $pids || true
  fi
}

if [ -z "${FRONTEND_PORTS:-}" ]; then
  if [ -n "${FRONTEND_PORT:-}" ]; then
    FRONTEND_PORTS="$FRONTEND_PORT"
  else
    FRONTEND_PORTS="5001 5100"
  fi
fi
for port in $FRONTEND_PORTS; do
  release_port "$port"
done

if [ ! -f "$ROOT_DIR/frontend/package.json" ]; then
  echo "未找到 $ROOT_DIR/frontend/package.json，无法启动前端"
  exit 1
fi

for port in $FRONTEND_PORTS; do
  nohup npm --prefix "$ROOT_DIR/frontend" run dev -- --host 0.0.0.0 --port "$port" > "/tmp/ai_face_change_frontend_${port}.log" 2>&1 &
  echo "前端已在后台启动 port=$port PID=$!"
done
