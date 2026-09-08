 # 自启动设置方法
# 终端输入gnome-session-properties
# gnome-terminal --working-directory=/home/auto/burn-your-bridges -- bash -lc "bash /home/auto/burn-your-bridges/run_camera_remote_hero.sh; exec bash"

#!/bin/bash

APP_DIR="/home/auto/burn-your-bridges"
CONFIG_PATH="configs/remote_hero.yaml"
APP_CMD=("./build/remote_hero" "$CONFIG_PATH")
RESTART_DELAY=2
LOG_DIR="$APP_DIR/logs"
WATCHDOG_LOG="$LOG_DIR/remote_hero_watchdog.log"
OPENVINO_SETUP="/opt/intel/openvino_2025.4.0/setupvars.sh"

stop_requested=0
child_pid=""

stop_child()
{
  stop_requested=1
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill "$child_pid" 2>/dev/null
    wait "$child_pid" 2>/dev/null
  fi
}

trap stop_child INT TERM

source "$OPENVINO_SETUP"
source /opt/ros/jazzy/setup.bash
source /home/auto/burn-your-bridges/install/setup.bash
source /home/auto/burn-your-bridges/sp_ws/install/setup.bash
set -u
cd "$APP_DIR"
mkdir -p "$LOG_DIR" patterns imgs

export LD_LIBRARY_PATH="/opt/MVS/lib/64:$APP_DIR/io/hikrobot/lib/amd64:$APP_DIR/io/mindvision/lib/amd64:${LD_LIBRARY_PATH:-}"

export GST_DEBUG=3

echo "[$(date '+%F %T')] remote_hero watchdog started." | tee -a "$WATCHDOG_LOG"

while [[ "$stop_requested" -eq 0 ]]; do
  echo "[$(date '+%F %T')] starting: ${APP_CMD[*]}" | tee -a "$WATCHDOG_LOG"

  "${APP_CMD[@]}" & #只写入终端
  # "${APP_CMD[@]}" > >(tee -a "$WATCHDOG_LOG") 2>&1 & 写入终端和log
  # "${APP_CMD[@]}" >/dev/null 2>>"$WATCHDOG_LOG" &  都不写入 io性能最强
  child_pid=$!
  wait "$child_pid"
  status=$?
  child_pid=""

  if [[ "$stop_requested" -ne 0 ]]; then
    break
  fi

  echo "[$(date '+%F %T')] program exited with status $status; restarting in ${RESTART_DELAY}s." \
    | tee -a "$WATCHDOG_LOG"
  sleep "$RESTART_DELAY"
done

echo "[$(date '+%F %T')] remote_hero watchdog stopped." | tee -a "$WATCHDOG_LOG"
