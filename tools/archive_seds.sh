#!/bin/bash
# Archive finished tracks' per-model SED CSVs (runs/<track>/SED/, written by the colors module with make_csv) into
# runs/<track>/SED.tar.zst, losslessly, and remove the loose CSVs only after the archive is checked against them:
# zstd's own test, tar --compare (every file's bytes, size, mode and time), the member count and the total bytes.
# A track counts as finished when it has seds.h5 (runner.py packs it when the run ends), no live .running lock, no
# `star` process working in it, and no SED file newer than 30 minutes. RSP runs (runs/*/fine/SED) are not touched:
# rsp.pack already removes their CSVs.
#   tools/archive_seds.sh [--dry-run] [--now] [track ...]      (default: every runs/*/SED)
# --now: the caller has just finished these tracks (runner.py, after packing seds.h5), so their SED files may be new.
# Restore a track's CSVs exactly:  tar -I zstd -xf runs/<track>/SED.tar.zst -C runs/<track>
set -uo pipefail
cd "$(dirname "$0")/.."
RUNS="$(cd runs && pwd -P)"
LOG=outputs/archive_seds.log
DRY=0
NOW=0
while [ $# -gt 0 ]; do
    case "$1" in --dry-run) DRY=1;; --now) NOW=1;; *) break;; esac
    shift
done
mkdir -p outputs
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

finished() {  # $1: run folder
    local run=$1 pid p
    [ -f "$run/seds.h5" ] || { echo "no seds.h5"; return 1; }
    if [ -f "$run/.running" ]; then
        pid=$(awk '{print $1; exit}' "$run/.running")
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && { echo "running (pid $pid)"; return 1; }
    fi
    for p in $(pgrep -x star); do
        case "$(readlink "/proc/$p/cwd" 2>/dev/null)" in "$run"|"$run"/*) echo "star process $p works in it"; return 1;; esac
    done
    [ $NOW = 1 ] || [ -z "$(find "$run/SED" -newermt '-30 minutes' -print -quit)" ] || {
        echo "SED written in the last 30 min"; return 1; }
    [ -z "$(find "$run/SED" -mindepth 1 ! -type f -print -quit)" ] || { echo "SED holds something other than files"; return 1; }
    [ ! -e "$run/SED.tar.zst" ] || { echo "SED.tar.zst already exists"; return 1; }
}

archive() {  # $1: run folder; returns 0 only when SED/ is archived, checked and removed
    local run=$1 sed=$1/SED out=$1/SED.tar.zst part=$1/SED.tar.zst.part n_disk b_disk n_arch b_arch t0=$SECONDS
    [[ "$sed" == "$RUNS"/*/SED && "${sed#"$RUNS"/}" != */*/* ]] || { log "REFUSED $sed: not runs/<track>/SED"; return 1; }
    n_disk=$(find "$sed" -type f | wc -l)
    b_disk=$(find "$sed" -type f -printf '%s\n' | awk '{s += $1} END {printf "%d", s}')
    tar -cf - -C "$run" SED | zstd -q -3 -T4 > "$part" || { log "FAIL $run: tar | zstd"; rm -f "$part"; return 1; }
    zstd -tq "$part" || { log "FAIL $run: zstd -t"; rm -f "$part"; return 1; }
    zstd -dcq "$part" | tar --compare -f - -C "$run" > /dev/null || { log "FAIL $run: tar --compare differs"; rm -f "$part"; return 1; }
    read -r n_arch b_arch < <(zstd -dcq "$part" | tar -tvf - | awk '$1 ~ /^-/ {n++; s += $3} END {printf "%d %d\n", n, s}')
    [ "$n_arch" = "$n_disk" ] && [ "$b_arch" = "$b_disk" ] || {
        log "FAIL $run: archive has $n_arch files / $b_arch B, disk $n_disk / $b_disk"; rm -f "$part"; return 1; }
    mv "$part" "$out" && sync -f "$out" || { log "FAIL $run: mv/sync"; return 1; }
    rm -rf -- "$sed" || { log "FAIL $run: rm"; return 1; }
    log "ok ${run#"$RUNS"/}: $n_disk files, $((b_disk / 1000000)) MB -> $(( $(stat -c %s "$out") / 1000000 )) MB ($((SECONDS - t0)) s)"
}

if [ $# -gt 0 ]; then runs=(); for t in "$@"; do runs+=("$RUNS/$t"); done
else runs=(); for s in "$RUNS"/*/SED; do [ -d "$s" ] && runs+=("${s%/SED}"); done; fi
log "start: ${#runs[@]} SED folders$([ $DRY = 1 ] && echo ' (dry run)'); disk free $(df -h --output=avail "$RUNS" | tail -1 | tr -d ' ')"
for run in "${runs[@]}"; do
    [ -d "$run/SED" ] || { log "skip ${run#"$RUNS"/}: no SED/"; continue; }
    why=$(finished "$run") || { log "skip ${run#"$RUNS"/}: $why"; continue; }
    if [ $DRY = 1 ]; then log "would archive ${run#"$RUNS"/}"; else archive "$run"; fi
done
log "end; disk free $(df -h --output=avail "$RUNS" | tail -1 | tr -d ' ')"
