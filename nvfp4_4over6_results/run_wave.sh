#!/bin/bash
# Run ablation arms sequentially on this node; a failed arm does not block
# the rest of the wave. Per-arm output goes to logs/<arm>.log, wave-level
# progress to stdout (redirect at the call site).
set -u
ABL=/home/scratch.hanlinb_ent_1/agent_scratch/nvfp4_4over6/ablation
for ARM in "$@"; do
  echo "=== $(date -u +%FT%TZ) $(hostname): launching $ARM ==="
  bash $ABL/run_ablation_arm.sh $ARM 7200 > $ABL/logs/$ARM.log 2>&1
  echo "=== $(date -u +%FT%TZ) $(hostname): $ARM exited $? ==="
done
echo "=== $(date -u +%FT%TZ) $(hostname): wave complete ==="
