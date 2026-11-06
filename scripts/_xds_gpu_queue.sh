#!/usr/bin/env bash
# Serial per-GPU job queue for multi-GPU pods (e.g. 14075: 4 GPUs).
#
#   XDS_GPU=2 bash scripts/_xds_gpu_queue.sh <jobs.tsv>
#
# Same TSV format as _xds_port_queue.sh:
#   BENCHMARK|TASK|TRACK|EXP_ARM|RATIO|HIST_N|OUT[|LIMIT[|EXTRA_ENV]]
# Differences: pins CUDA_VISIBLE_DEVICES to XDS_GPU and serializes with a
# per-GPU flock only (no global pgrep wait), so queues on different GPUs of
# the same pod run in parallel while each GPU stays strictly serial.
set -uo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=eval_env_defaults.sh
source "${_SCRIPT_DIR}/eval_env_defaults.sh"
cd "$REPO"

JOBS="${1:?jobs file required}"
GPU="${XDS_GPU:?XDS_GPU required (GPU index for this queue)}"
QLOG="logs/xds/$(basename "${JOBS%.tsv}").gpu${GPU}.queue.log"
mkdir -p logs/xds
exec >> "$QLOG" 2>&1

echo "=== gpu-queue start $(date) gpu=${GPU} jobs=${JOBS} ==="
while IFS='|' read -r benchmark task track arm ratio histn out limit extraenv; do
  [[ -z "$benchmark" || "$benchmark" == \#* ]] && continue
  if grep -q "aggregate" "${out}/run.log" 2>/dev/null || \
     ls "${out}"/results/*.json > /dev/null 2>&1; then
    echo "SKIP done: ${out}"
    continue
  fi
  echo "--- RUN $(date) ${benchmark}/${track} arm=${arm} ratio=${ratio} hist=${histn} gpu=${GPU} -> ${out}"
  env_extra=()
  [[ "$ratio" != "-" && -n "$ratio" ]] && env_extra+=("PRUNE_RATIO=${ratio}")
  [[ "$histn" != "-" && -n "$histn" ]] && env_extra+=("GEOPACT_EVAL_HISTORY_N=${histn}")
  [[ "${limit:-}" != "-" && -n "${limit:-}" ]] && env_extra+=("LIMIT=${limit}")
  if [[ "${extraenv:-}" != "-" && -n "${extraenv:-}" ]]; then
    IFS=',' read -ra _kvs <<< "$extraenv"
    for _kv in "${_kvs[@]}"; do
      [[ -n "$_kv" ]] && env_extra+=("$_kv")
    done
  fi
  mkdir -p "$out"
  # Per-OUT claim, non-blocking. The done-check above only skips cells that have
  # already FINISHED; when the same TSV is deliberately launched on two pods (to
  # use whichever frees a GPU first) two queues can otherwise start the same cell
  # concurrently and write over each other's run dir. A per-OUT lock makes the
  # second one skip instead. Held for the duration of the job via fd 9.
  # The lock lives on the SHARED repo volume, not /tmp: /tmp is pod-local, so a
  # /tmp lock would only serialise queues inside one pod and miss exactly the
  # cross-pod race it is meant to prevent.
  # NOTE: flock does NOT work here. This volume is NFS, where flock(2) needs
  # server-side NLM to be honoured across hosts; on this mount it succeeds
  # locally and grants the same "exclusive" lock on two pods at once. Verified
  # 2026-07-31: with the flock version live on both pods,
  # logs/w46_uitars_multi/omnigui/dense ran simultaneously on 18737 and 11283.
  # mkdir(2) is atomic over NFS and is the portable primitive, so the claim is a
  # DIRECTORY whose creation succeeds for exactly one caller.
  mkdir -p .xds_claims
  _claim=".xds_claims/$(echo "$out" | tr '/' '_').d"
  if ! mkdir "$_claim" 2>/dev/null; then
    # Stale claim recovery: if the owner is gone and the cell still has no
    # results, reclaim it. Otherwise defer to the live owner.
    _owner_host=$(cat "${_claim}/host" 2>/dev/null || true)
    _owner_pid=$(cat "${_claim}/pid" 2>/dev/null || true)
    if [[ "$_owner_host" == "$(hostname)" ]] && \
       [[ -n "$_owner_pid" ]] && ! kill -0 "$_owner_pid" 2>/dev/null; then
      echo "RECLAIM stale claim from dead pid ${_owner_pid}: ${out}"
      rm -rf "$_claim"
      mkdir "$_claim" 2>/dev/null || { echo "SKIP raced: ${out}"; continue; }
    else
      echo "SKIP claimed by ${_owner_host:-?}/${_owner_pid:-?}: ${out}"
      continue
    fi
  fi
  hostname > "${_claim}/host" 2>/dev/null || true
  echo $$ > "${_claim}/pid" 2>/dev/null || true
  date >  "${_claim}/since" 2>/dev/null || true

  flock "/tmp/xds_gpu${GPU}.lock" env CUDA_VISIBLE_DEVICES="$GPU" \
    BENCHMARK="$benchmark" TASK="$task" TRACK="$track" EXP_ARM="$arm" \
    OUT="$out" "${env_extra[@]}" \
    bash scripts/run_cross_dataset_track_one.sh
  rc=$?
  rm -rf "$_claim"
  echo "--- DONE rc=${rc} ${benchmark}/${track}"
done < "$JOBS"
echo "=== gpu-queue finished $(date) gpu=${GPU} ==="
