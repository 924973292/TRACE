#!/usr/bin/env bash
# Shared arm list: baseline + dialogue SESSION_KV + uniform keep-ratio sweep.
# Source from run/harvest scripts; do not execute directly.
KV_PRUNE_ARMS=(
  00_baseline
  01_dialogue_kv_noprune
  02_dialogue_kv_r99
  03_dialogue_kv_r95
  04_dialogue_kv_r40
  05_dialogue_kv_r20
  06_dialogue_kv_r10
  07_dialogue_kv_r05
)

# Arms that get ledger vs baseline (all except 00_baseline).
KV_PRUNE_LEDGER_ARMS=(
  01_dialogue_kv_noprune
  02_dialogue_kv_r99
  03_dialogue_kv_r95
  04_dialogue_kv_r40
  05_dialogue_kv_r20
  06_dialogue_kv_r10
  07_dialogue_kv_r05
)

_kv_prune_arm_meta() {
  case "$1" in
    00_baseline) echo "OFF|none" ;;
    01_dialogue_kv_noprune) echo "ON|none" ;;
    02_dialogue_kv_r99) echo "ON|99%" ;;
    03_dialogue_kv_r95) echo "ON|95%" ;;
    04_dialogue_kv_r40) echo "ON|40%" ;;
    05_dialogue_kv_r20) echo "ON|20%" ;;
    06_dialogue_kv_r10) echo "ON|10%" ;;
    07_dialogue_kv_r05) echo "ON|5%" ;;
    *) echo "?|?" ;;
  esac
}
