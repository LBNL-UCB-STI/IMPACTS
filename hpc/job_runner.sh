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
stage_arg="pipeline"
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
    -s|--stage)
        stage_arg="${2:-}"
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
        echo "Usage: $0 -c <config> -a <account> [-s stage] [-p partition] [--high-mem|-H] [-t hours]"
        echo "  -c             Path to impacts config file (required)"
        echo "  -a, --account  Slurm account name (required)"
        echo "  -s, --stage    Stage to run: pipeline (default), preprocess, presim, activities,"
        echo "                 fleet, postsim, emissions, inmap, aermod, exposure, postprocess, analysis"
        echo "  -p             Partition: lr4, lr5, lr6, lr7 (default), lr8, lr_bigmem, cm1, cm2"
        echo "  --high-mem     Request high-memory config (lr6: 180G, lr7: 480G)"
        echo "  -t, --hours    Job time limit in hours (default: 24)"
        exit 0
        ;;
    *)
        echo "Usage: $0 -c <config> -a <account> [-s stage] [-p partition] [--high-mem|-H] [-t hours]"
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
    echo "  ./hpc/job_runner.sh -c examples/pilates/settings.yaml -a my_account" >&2
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
    lr6)
        PARTITION="lr6"
        QOS="lr_normal"
        NUM_CPUS=32
        if [ "$high_mem" = true ]; then
            MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-180}"
        else
            MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-90}"
        fi
        ;;
    lr5)
        PARTITION="lr5"
        QOS="lr_normal"
        NUM_CPUS=28
        MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-60}"
        ;;
    lr4)
        PARTITION="lr4"
        QOS="lr_normal"
        NUM_CPUS=24
        MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-60}"
        ;;
    lr_bigmem)
        PARTITION="lr_bigmem"
        QOS="lr_bigmem"
        NUM_CPUS=32
        MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-1400}"
        ;;
    cm1)
        PARTITION="cm1"
        QOS="cm_normal"
        NUM_CPUS=48
        MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-240}"
        ;;
    cm2)
        PARTITION="cm2"
        QOS="cm_normal"
        NUM_CPUS=64
        MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-240}"
        ;;
    *)
        echo "ERROR: unsupported partition '$partition_arg'"
        echo "Available: lr4, lr5, lr6, lr7 (default), lr8, lr_bigmem, cm1, cm2"
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

IMPACTS_OUTPUT_FOLDER="$(awk '
    /^[^[:space:]#][^:]*:/ {
        section=$1
        sub(":", "", section)
    }
    section == "impacts" && /^[[:space:]]+local_output_folder:[[:space:]]*/ {
        sub(/^[[:space:]]*local_output_folder:[[:space:]]*/, "", $0)
        gsub(/^[[:space:]'\''"]+|[[:space:]'\''"]+$/, "", $0)
        print
        exit
    }
' "$CONFIG_PATH")"

if [ -z "$IMPACTS_OUTPUT_FOLDER" ]; then
    echo "ERROR: impacts.local_output_folder not found in config: $CONFIG_PATH" >&2
    exit 1
fi

CONFIG_DIR="$(cd "$(dirname "$CONFIG_PATH")" && pwd)"
case "$IMPACTS_OUTPUT_FOLDER" in
    /*)
        JOB_LOG_DIR="$IMPACTS_OUTPUT_FOLDER"
        ;;
    ~/*)
        JOB_LOG_DIR="$HOME/${IMPACTS_OUTPUT_FOLDER#~/}"
        ;;
    *)
        CONFIG_RELATIVE_LOG_DIR="$CONFIG_DIR/$IMPACTS_OUTPUT_FOLDER"
        CWD_RELATIVE_LOG_DIR="$IMPACTS_DIR/$IMPACTS_OUTPUT_FOLDER"
        if [ -e "$CONFIG_RELATIVE_LOG_DIR" ] || [ ! -e "$CWD_RELATIVE_LOG_DIR" ]; then
            JOB_LOG_DIR="$CONFIG_RELATIVE_LOG_DIR"
        else
            JOB_LOG_DIR="$CWD_RELATIVE_LOG_DIR"
        fi
        ;;
esac

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
    "$CONFIG_PATH" \
    "$stage_arg"

echo "Job submitted. Log: $JOB_LOG_FILE_PATH"
