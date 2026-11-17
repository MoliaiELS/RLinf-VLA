#!/usr/bin/env bash
set -euo pipefail
tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$tool_dir/runtime.env"
mode="${1:?mode}" partition="${2:?partition}" gpu="${3:?gpu}" lane="${4:?lane}" hours="${5:?hours}" task="${6:-}"
evaluation_root="${7:-}"
data="$FYP_ROBOTWIN_DATA"
input="${FYP_ROBOTWIN_INPUT_ROOT:-$data}"
mkdir -p "$data/logs" "$data/evidence"
exec 9> "$data/submit.lock"
flock 9
test -s "$data/evidence/build.complete"
if [[ "$mode" != probe ]]; then
    test -s "$input/models/X-VLA-RoboTwin2--${MODEL_REV:0:12}/verified.json"
    robot="$FYP_ROBOTWIN_CODE/RoboTwin-$ROBOTWIN_REV"
    if [[ "$mode" == easy-preflight ]]; then
        test "$lane" = easy
        test -s "$data/easy-preflight-inputs/$ASSETS_REV/qualification.json"
        robot="$FYP_ROBOTWIN_CODE/RoboTwin-easy-preflight-$ROBOTWIN_REV"
    fi
    for arm in left right; do
        test -s "$robot/assets/embodiments/aloha-agilex/curobo_${arm}.yml"
    done
fi
# Count assigned and requested GPUs across the user's experiments.
own="$(squeue -h -u "$(id -un)" -o '%b')"
own_gpus="$(awk 'BEGIN {n=0} {for (i=1;i<=NF;i++) if ($i ~ /^gres[/:]gpu(:[[:alnum:]_-]+)?:[0-9]+$/) {split($i,a,":"); n+=a[length(a)]}} END{print n}' <<< "${own//,/ }")"
if (( own_gpus >= 2 )); then echo 'Two own GPUs are already allocated or requested' >&2; exit 1; fi
job_args=("$mode")
cpus=8
if [[ "$mode" == concurrency-pilot ]]; then
    test "$lane" = easy
    test -z "$task"
    test "$gpu" = 4090
    test "$hours" = 1
    test -r "${evaluation_root:?pinned evaluation root required}/tools/xvla/robotwin/runner.py"
    job_args+=("$evaluation_root")
    cpus=16
else
    test -z "$evaluation_root"
    job_args+=(--lane "$lane")
    if [[ -n "$task" ]]; then job_args+=(--task "$task"); fi
fi
printf -v launch '%q ' bash "$tool_dir/job.sh" "${job_args[@]}"
jobid="$(sbatch --parsable --partition="$partition" --exclude=4090node1 --nodes=1 --ntasks=1 \
    --cpus-per-task="$cpus" --time="$((10#$hours * 60))" --gres="gpu:$gpu:1" \
    --job-name="rw-b0-$mode-$lane" --output="$data/logs/%j.out" --error="$data/logs/%j.err" \
    --export=ALL --wrap="$launch")"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$jobid" "$mode" "$lane" "$gpu" "$(git -C "$tool_dir" rev-parse HEAD)" "$task" >> "$data/evidence/jobs.tsv"
echo "ROBOTWIN_JOB_ID=$jobid"
