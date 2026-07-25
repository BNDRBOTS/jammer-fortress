#!/usr/bin/env sh
# JAMMER-FORTRESS launcher. Usage: ./run.sh [serve|selftest|console] [args...]
set -e
cd "$(dirname "$0")"
MODE="${1:-serve}"
[ $# -gt 0 ] && shift
case "$MODE" in
  serve)    exec python3 -m jammerfortress serve "$@" ;;
  selftest) exec python3 -m jammerfortress selftest ;;
  console)  exec python3 -m jammerfortress ;;
  *) echo "usage: ./run.sh [serve|selftest|console]"; exit 2 ;;
esac
