#!/bin/bash
# newton's job: build every star file in queue/ (in name order), forever. A file is done when make_stars.py has
# built (or given up on: runs/*/FAILED) every star in it; new files dropped into queue/ are picked up on the next pass.
# One runner only (a lock); it starts after any make_stars.py already building the queue has finished.
#   setsid nohup tools/newton_queue.sh > outputs/queue.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
mkdir -p queue/done
exec 9> queue/.lock
flock -n 9 || { echo "a queue runner is already running"; exit 0; }
export MESA_DIR=$HOME/MESA/mesa MESASDK_ROOT=$HOME/MESA/mesasdk-26.6.1
source $MESASDK_ROOT/bin/mesasdk_init.sh
while pgrep -f "make_stars.py queue/" > /dev/null; do sleep 60; done
while true; do
  for f in $(ls queue/*.toml 2>/dev/null | sort); do
    echo "$(date '+%F %T') start $f"
    python3 make_stars.py "$f"  # non-zero if any star failed; failures are final (runs/*/FAILED), so go on
    rc=$?
    echo "$(date '+%F %T') end $f (exit $rc)"
    mv "$f" queue/done/
  done
  sleep 600
done
