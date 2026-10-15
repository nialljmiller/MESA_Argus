#!/bin/bash
# newton's job: build every star file in queue/ (in name order), forever. A file is done when make_stars.py has
# built (or given up on: runs/*/FAILED) every star in it; new files dropped into queue/ are picked up on the next pass.
#   setsid nohup tools/newton_queue.sh > outputs/queue.log 2>&1 &
cd "$(dirname "$0")/.."
export MESA_DIR=$HOME/MESA/mesa MESASDK_ROOT=$HOME/MESA/mesasdk-26.6.1
source $MESASDK_ROOT/bin/mesasdk_init.sh
mkdir -p queue/done
while true; do
  for f in $(ls queue/*.toml 2>/dev/null | sort); do
    echo "$(date '+%F %T') start $f"
    python3 make_stars.py "$f"  # exits non-zero if any star failed; failures are final (FAILED), so go on
    echo "$(date '+%F %T') end $f (exit $?)"
    mv "$f" queue/done/
  done
  sleep 600
done
