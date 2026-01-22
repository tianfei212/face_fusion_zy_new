#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "/home/ubuntu/miniconda3/etc/profile.d/conda.sh" ]; then
  set +u
  source "/home/ubuntu/miniconda3/etc/profile.d/conda.sh"
  conda activate ai_face_change
  set -u
fi

export BLANKEND_HW_TRANSCODE=1
if [ -n "${CONDA_PREFIX:-}" ]; then
  export CUDA_HOME="$CONDA_PREFIX"
  export CUDA_PATH="$CONDA_PREFIX"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
  TRT_LIB_DIR="$("$CONDA_PREFIX/bin/python" -c "import os,tensorrt_libs;print(os.path.dirname(tensorrt_libs.__file__))" 2>/dev/null || true)"
  if [ -n "${TRT_LIB_DIR:-}" ]; then
    export LD_LIBRARY_PATH="$TRT_LIB_DIR:${LD_LIBRARY_PATH:-}"
  fi
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

release_port 8001
release_port 8100

echo "清理所有残留进程..."
pkill -9 -f "ai_face_change" || true
pkill -9 -f "uvicorn" || true
pkill -9 -f "multiprocessing.spawn" || true

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export SIMPLE_FORWARD="${SIMPLE_FORWARD:-0}"
echo "SIMPLE_FORWARD=$SIMPLE_FORWARD"

mkdir -p "$ROOT_DIR/blankend/logs"
# Start API server (Relay disabled)
nohup env BLANKEND_RELAY_ENABLED=0 python3 -m uvicorn blankend.main:app --host 0.0.0.0 --port 8001 > "$ROOT_DIR/blankend/logs/backend_8001.log" 2>&1 &
echo "后端8001已在后台启动 PID=$!"

# Start Worker server (Relay enabled)
nohup env BLANKEND_RELAY_ENABLED=1 python3 -m uvicorn blankend.main:app --host 0.0.0.0 --port 8100 > "$ROOT_DIR/blankend/logs/backend_8100.log" 2>&1 &
echo "后端8100已在后台启动 PID=$!"
