#!/usr/bin/env bash
# run_regressed_flip_perf.sh --- run flip_perf_harness.py against every flip_id
# flagged `"regressed": true` in a karnage perf_report.json.
#
# Usage:
#   scripts/run_regressed_flip_perf.sh [--perf-report PATH] [-- extra flip_perf_harness.py args]
#
# Any arguments after the recognized flags are forwarded verbatim to
# flip_perf_harness.py. num_prompts is a swept factor (not a single fixed value)
# via --scale-spec, so a full run is staged rather than one big sweep:
#
#   # Phase 1: full grid at n=10 and n=100 (~12h for 5 models x 2 flip_ids)
#   scripts/run_regressed_flip_perf.sh -- --scale-spec "10:10:5,100:5:5"
#
#   # Phase 2: escalate only the (model, flip_site) pairs that showed a real,
#   # gate-passing effect in Phase 1's summary.csv to n=1000 -- reuse the SAME
#   # --output-dir the Phase 1 run used so results accumulate in one place.
#   scripts/run_regressed_flip_perf.sh -- \
#     --models qwen,gemma --target-flip-ids 39172 \
#     --scale-spec "1000:3:4" --bench-timeout 3600 \
#     --output-dir results/flip-perf-regressed-<phase1-timestamp>
#
# An existing run made under the old exact-match correctness gate can be
# retrofitted in place (recomputes correctness_pass/valid from bench_result.json
# files already on disk, no new vllm serve/bench serve runs) via:
#   python3 scripts/flip_perf_harness.py --retrofit-correctness \
#     --output-dir results/flip-perf-regressed-<old-timestamp> --golden-refs 5
set -euo pipefail

_THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_THIS_DIR/.." && pwd)"

PERF_REPORT="$REPO_ROOT/results/perf-vllm-trtion-attn/perf_report.json"
OUTPUT_DIR="$REPO_ROOT/results/flip-perf-regressed-$(date +%Y%m%d-%H%M%S)"

if [[ "${1:-}" == "--perf-report" ]]; then
  PERF_REPORT="$2"
  shift 2
fi

# Accept (and discard) a conventional "--" separator before forwarded args, as
# shown in this script's own usage examples above -- flip_perf_harness.py's
# argparse treats a literal "--" as "everything after this is positional" and
# rejects --scale-spec etc. following it, so it must not be forwarded verbatim.
if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ ! -f "$PERF_REPORT" ]]; then
  echo "perf_report.json not found: $PERF_REPORT" >&2
  exit 1
fi

FLIP_IDS_FILE="$(mktemp)"
trap 'rm -f "$FLIP_IDS_FILE"' EXIT

python3 -c "
import json, sys
report = json.load(open(sys.argv[1]))
ids = sorted({r['flip_id'] for r in report if r.get('regressed')})
if not ids:
    sys.exit(f'No regressed==true entries in {sys.argv[1]}')
json.dump(ids, open(sys.argv[2], 'w'))
print(f'{len(ids)} regressed flip_id(s) from {sys.argv[1]}: {ids}', file=sys.stderr)
" "$PERF_REPORT" "$FLIP_IDS_FILE"

# Not exec'd: the EXIT trap above must still fire (to clean up $FLIP_IDS_FILE)
# after this finishes, which exec would skip by replacing the shell process.
python3 "$_THIS_DIR/flip_perf_harness.py" \
  --target-flip-ids-file "$FLIP_IDS_FILE" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
