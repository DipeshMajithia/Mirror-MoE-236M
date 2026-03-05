#!/bin/bash
# MirrorAI V2 — SOTA Reconstruction Marathon (48-Hour Script)
# This script executes Phase 1.1, 2.4, and 3.5 sequentially.

ROOT_DIR="/Users/paramindia/Desktop/D/MirrorAI_Clean"
cd "$ROOT_DIR"

echo "============================================================"
echo "🚀 MirrorAI V2 — SOTA RECONSTRUCTION MARATHON STARTING"
echo "============================================================"
date
df -h .

# --- Phase 1.1: Backbone Expansion ---
echo "\n[PHASE 1.1] Backbone Expansion (70,000 steps)..."
/opt/homebrew/bin/python3.11 v2/run_v2_train.py --phase 1
if [ $? -ne 0 ]; then echo "❌ Phase 1.1 FAILED"; exit 1; fi
df -h .

# --- Phase 2.4: Router Refresh ---
echo "\n[PHASE 2.4] Router Refresh (5,000 steps)..."
/opt/homebrew/bin/python3.11 v2/run_v2_train.py --phase 2
if [ $? -ne 0 ]; then echo "❌ Phase 2.4 FAILED"; exit 1; fi
df -h .

# --- Phase 3.5: Master Echo ---
echo "\n[PHASE 3.5] Master Echo (10,000 steps)..."
/opt/homebrew/bin/python3.11 v2/run_v2_train.py --phase 3
if [ $? -ne 0 ]; then echo "❌ Phase 3.5 FAILED"; exit 1; fi
df -h .

echo "\n============================================================"
echo "🏆 MIRRORAI V2 — SOTA RECONSTRUCTION COMPLETE!"
echo "============================================================"
date
df -h .
