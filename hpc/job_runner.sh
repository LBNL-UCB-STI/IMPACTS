#!/bin/bash

# Slurm submission wrapper for impacts HPC jobs.

set -euo pipefail

error_exit() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
    echo "ERROR: job_runner.sh failed!" >&2
    echo "  Line: $1" >&2
    echo "  Command: $2" >&2
    echo "  Exit code: $3" >&2
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
    exit "$3"
}

trap 'error_exit ${LINENO} "$BASH_COMMAND" $?' ERR

RANDOM_PART="$(head -c 32 /dev/urandom | base64 | tr -dc A-Z0-9 | head -c 8)"
DATETIME="$(date "+%Y.%m.%d-%H.%M.%S")"
JOB_NAME="$RANDOM_PART.$DATETIME"

IMPACTS_DIR="${IMPACTS_DIR:-/global/scratch/users/$USER/sources/impacts}"

config_file=""
partition_arg="lr7"
account_arg=""
high_mem=false
hours_arg=""

while [ $# -gt 0 ]; do
    case "$1" in
    -c)
        config_file="${2:-}"
        shift 2
        ;;
    -p)
        partition_arg="${2:-}"
        shift 2
        ;;
    -a|--account)
        account_arg="${2:-}"
        shift 2
        ;;
    --high-mem|-H)
        high_mem=true
        shift
        ;;
    -t|--hours)
        hours_arg="${2:-}"
        shift 2
        ;;
    -h|--help)
        echo "Usage: $0 -c <config> -a <account> [-p partition] [--high-mem|-H] [-t hours]"
        echo "  -c             Path to impacts settings.yaml (required)"
        echo "  -a, --account  Slurm account name (required)"
        echo "  -p             Partition: lr7 (default) or lr8"
        echo "  --high-mem     For lr7: request 480G instead of 240G"
        echo "  -t, --hours    Job time limit in hours (default: 24)"
        exit 0
        ;;
    *)
        echo "Usage: $0 -c <config> -a <account> [-p partition] [--high-mem|-H] [-t hours]"
        exit 2
        ;;
    esac
done

if [ -z "$config_file" ]; then
    echo "ERROR: config file is required. Use -c <settings.yaml>" >&2
    exit 2
fi

if [ -z "$account_arg" ]; then
    echo "ERROR: Slurm account is required. Use -a <account>" >&2
    echo "Example:" >&2
    echo "  ./hpc/job_runner.sh -c examples/pipeline/pilates/settings.yaml -a my_account" >&2
    exit 2
fi

case "$partition_arg" in
    lr8)
        PARTITION="lr8"
        QOS="lr8_normal"
        NUM_CPUS=128
        MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-700}"
        ;;
    lr7)
        PARTITION="lr7"
        QOS="lr_normal"
        NUM_CPUS=56
        if [ "$high_mem" = true ]; then
            MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-480}"
        else
            MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-240}"
        fi
        ;;
    *)
        echo "ERROR: unsupported partition '$partition_arg' (expected lr7 or lr8)"
        exit 2
        ;;
esac

if [ -n "$hours_arg" ]; then
    days=$(( hours_arg / 24 ))
    remaining_hours=$(( hours_arg % 24 ))
    TIME_LIMIT="$(printf '%d-%02d:00:00' "$days" "$remaining_hours")"
else
    TIME_LIMIT="${EXPECTED_EXECUTION_DURATION:-1-00:00:00}"
fi

# Resolve config path relative to IMPACTS_DIR if not absolute
if [ "${config_file#/}" != "$config_file" ]; then
    CONFIG_PATH="$config_file"
else
    CONFIG_PATH="$IMPACTS_DIR/$config_file"
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: config file not found: $CONFIG_PATH"
    exit 1
fi

JOB_LOG_DIR="/global/scratch/users/$USER/impacts_logs"
mkdir -p "$JOB_LOG_DIR"
JOB_LOG_FILE_PATH="$JOB_LOG_DIR/log_${DATETIME}_${RANDOM_PART}.log"

sbatch \
    --partition="$PARTITION" \
    --exclusive \
    --cpus-per-task="$NUM_CPUS" \
    --mem="${MEMORY_LIMIT_GB}G" \
    --qos="$QOS" \
    --account="$account_arg" \
    --job-name="impacts.$JOB_NAME" \
    --output="$JOB_LOG_FILE_PATH" \
    --time="$TIME_LIMIT" \
    --export="ALL,JOB_LOG_FILE_PATH=$JOB_LOG_FILE_PATH" \
    "$IMPACTS_DIR/hpc/job.sh" \
    "$CONFIG_PATH"

echo "Job submitted. Log: $JOB_LOG_FILE_PATH"