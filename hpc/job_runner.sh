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
PILATES_TIMESTAMP="$(date "+%Y%m%d-%H%M%S")"
JOB_NAME="$RANDOM_PART.$DATETIME"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMPACTS_DIR="${IMPACTS_DIR:-$REPO_ROOT}"
IMPACTS_DIR="$(cd "$IMPACTS_DIR" && pwd)"
export IMPACTS_DIR

config_file=""
stage_arg="pipeline"
partition_arg="lr7"
account_arg=""
high_mem=false
hours_arg=""
watch=false
profile_arg="${IMPACTS_PROFILE:-none}"

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
    -w|--watch)
        watch=true
        shift
        ;;
    --profile)
        profile_arg="${2:-}"
        shift 2
        ;;
    -h|--help)
        echo "Usage: $0 -c <settings-or-manifest> -a <account> [-s stage] [-p partition] [--high-mem|-H] [-t hours] [--profile mode] [-w]"
        echo "  -c             Settings YAML for pipeline/preprocess/presim/activities/postsim,"
        echo "                 activities_manifest.yaml for fleet, pipeline_manifest.yaml for"
        echo "                 emissions/inmap/aermod/exposure/postprocess"
        echo "  -a, --account  Slurm account name (required)"
        echo "  -s, --stage    Stage to run: pipeline (default), preprocess, presim, activities,"
        echo "                 fleet, postsim, emissions, inmap, aermod, exposure, postprocess"
        echo "  -p             Partition: lr4, lr5, lr6, lr7 (default), lr8, lr_bigmem, cm1, cm2"
        echo "  --high-mem     Request high-memory config (lr6: 180G, lr7: 480G)"
        echo "  -t, --hours    Job time limit in hours (default: 24)"
        echo "  --profile      Profiling mode: none (default), time, cpu, memray, all"
        echo "  -w, --watch    Stream log output after submission (Ctrl+C detaches, job keeps running)"
        exit 0
        ;;
    *)
        echo "Usage: $0 -c <settings-or-manifest> -a <account> [-s stage] [-p partition] [--high-mem|-H] [-t hours] [--profile mode] [-w]"
        exit 2
        ;;
    esac
done

if [ -z "$config_file" ]; then
    echo "ERROR: settings or manifest file is required. Use -c <settings.yaml|pipeline_manifest.yaml|activities_manifest.yaml>" >&2
    exit 2
fi

if [ -z "$account_arg" ]; then
    echo "ERROR: Slurm account is required. Use -a <account>" >&2
    echo "Example:" >&2
    echo "  ./hpc/job_runner.sh -c examples/pilates/settings.yaml -a my_account" >&2
    exit 2
fi

case "$profile_arg" in
    none|time|cpu|memray|all)
        ;;
    *)
        echo "ERROR: unsupported profile mode '$profile_arg'"
        echo "Available: none, time, cpu, memray, all"
        exit 2
        ;;
esac

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
    echo "ERROR: settings or manifest file not found: $CONFIG_PATH"
    exit 1
fi

_stage_uses_run_manifest() {
    case "$stage_arg" in
        fleet|emissions|inmap|aermod|exposure|postprocess)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

_top_level_yaml_value() {
    local key="$1"
    awk -v want="$key" '
        /^[^[:space:]#][^:]*:/ {
            key=$1
            sub(":", "", key)
            if (key == want) {
                sub(/^[^:]+:[[:space:]]*/, "", $0)
                gsub(/^[[:space:]'\''"]+|[[:space:]'\''"]+$/, "", $0)
                print
                exit
            }
        }
    ' "$CONFIG_PATH"
}

IMPACTS_OUTPUT_FOLDER=""
MANIFEST_OUTPUT_DIR=""
if _stage_uses_run_manifest; then
    MANIFEST_OUTPUT_DIR="$(_top_level_yaml_value output_dir)"
    if [ -z "$MANIFEST_OUTPUT_DIR" ]; then
        echo "ERROR: stage '$stage_arg' requires a manifest with top-level output_dir: $CONFIG_PATH" >&2
        exit 1
    fi
else
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
        echo "ERROR: impacts.local_output_folder not found in settings file: $CONFIG_PATH" >&2
        exit 1
    fi
fi

_settings_run_value() {
    local key="$1"
    awk -v want="$key" '
        function val(s,    v) {
            v = s; sub(/^[^:]+:[[:space:]]*/, "", v)
            gsub(/^[[:space:]'\''"]+|[[:space:]'\''"]+$/, "", v)
            return v
        }
        /^run:/ { in_run=1; next }
        /^[^[:space:]#]/ { in_run=0 }
        in_run && $0 ~ "^[[:space:]]+" want ":[[:space:]]*" { print val($0); exit }
    ' "$CONFIG_PATH"
}

_required_settings_run_value() {
    local key="$1"
    local value
    value="$(_settings_run_value "$key")"
    if [ -z "$value" ]; then
        echo "ERROR: run.$key not found in settings file: $CONFIG_PATH" >&2
        exit 1
    fi
    printf '%s\n' "$value"
}

_settings_run_label() {
    local label
    label="$(_settings_run_value output_run_name)"
    if [ -z "$label" ]; then
        label="$(_required_settings_run_value scenario)"
    fi
    printf '%s\n' "$label"
}

CONFIG_DIR="$(cd "$(dirname "$CONFIG_PATH")" && pwd)"
if _stage_uses_run_manifest; then
    OUTPUT_ROOT_RAW="$MANIFEST_OUTPUT_DIR"
else
    OUTPUT_ROOT_RAW="$IMPACTS_OUTPUT_FOLDER"
fi

case "$OUTPUT_ROOT_RAW" in
    /*)
        JOB_LOG_DIR="$OUTPUT_ROOT_RAW"
        ;;
    ~/*)
        JOB_LOG_DIR="$HOME/${OUTPUT_ROOT_RAW#~/}"
        ;;
    *)
        CONFIG_RELATIVE_LOG_DIR="$CONFIG_DIR/$OUTPUT_ROOT_RAW"
        CWD_RELATIVE_LOG_DIR="$IMPACTS_DIR/$OUTPUT_ROOT_RAW"
        if _stage_uses_run_manifest; then
            JOB_LOG_DIR="$CONFIG_RELATIVE_LOG_DIR"
        elif [ -e "$CONFIG_RELATIVE_LOG_DIR" ] || [ ! -e "$CWD_RELATIVE_LOG_DIR" ]; then
            JOB_LOG_DIR="$CONFIG_RELATIVE_LOG_DIR"
        else
            JOB_LOG_DIR="$CWD_RELATIVE_LOG_DIR"
        fi
        ;;
esac

ACTIVITIES_OUTPUT_DIR=""
POSTSIM_OUTPUT_DIR=""
case "$stage_arg" in
    activities)
        ACTIVITIES_REGION="$(_required_settings_run_value region)"
        ACTIVITIES_LABEL="$(_settings_run_label)"
        ACTIVITIES_OUTPUT_DIR="$JOB_LOG_DIR/impacts_presim--${ACTIVITIES_REGION}--${ACTIVITIES_LABEL}/activities"
        JOB_LOG_DIR="$ACTIVITIES_OUTPUT_DIR"
        ;;
    postsim)
        POSTSIM_REGION="$(_required_settings_run_value region)"
        POSTSIM_LABEL="$(_settings_run_label)"
        POSTSIM_OUTPUT_BASENAME="impacts-postsim--${POSTSIM_REGION}--${POSTSIM_LABEL}--${PILATES_TIMESTAMP}"
        POSTSIM_OUTPUT_DIR="$JOB_LOG_DIR/$POSTSIM_OUTPUT_BASENAME"
        POSTSIM_SUFFIX=2
        while [ -e "$POSTSIM_OUTPUT_DIR" ]; do
            POSTSIM_OUTPUT_DIR="$JOB_LOG_DIR/${POSTSIM_OUTPUT_BASENAME}_$(printf '%02d' "$POSTSIM_SUFFIX")"
            POSTSIM_SUFFIX=$((POSTSIM_SUFFIX + 1))
        done
        JOB_LOG_DIR="$POSTSIM_OUTPUT_DIR"
        ;;
esac

mkdir -p "$JOB_LOG_DIR"
JOB_LOG_FILE_PATH="$JOB_LOG_DIR/log_${DATETIME}_${RANDOM_PART}.log"
SBATCH_EXPORT="ALL,JOB_LOG_FILE_PATH=$JOB_LOG_FILE_PATH,IMPACTS_PROFILE=$profile_arg"
if [ -n "$POSTSIM_OUTPUT_DIR" ]; then
    SBATCH_EXPORT="$SBATCH_EXPORT,IMPACTS_POSTSIM_OUTPUT_DIR=$POSTSIM_OUTPUT_DIR"
fi

SBATCH_OUTPUT="$(sbatch \
    --partition="$PARTITION" \
    --exclusive \
    --cpus-per-task="$NUM_CPUS" \
    --mem="${MEMORY_LIMIT_GB}G" \
    --qos="$QOS" \
    --account="$account_arg" \
    --job-name="impacts.$JOB_NAME" \
    --output="$JOB_LOG_FILE_PATH" \
    --time="$TIME_LIMIT" \
    --export="$SBATCH_EXPORT" \
    "$IMPACTS_DIR/hpc/job.sh" \
    "$CONFIG_PATH" \
    "$stage_arg")"

echo "$SBATCH_OUTPUT"
JOB_ID="$(printf '%s' "$SBATCH_OUTPUT" | sed -n 's/.*Submitted batch job //p' | tr -d '[:space:]')"
echo "Job submitted. Log: $JOB_LOG_FILE_PATH"
if [ -n "$ACTIVITIES_OUTPUT_DIR" ]; then
    echo "Activities output: $ACTIVITIES_OUTPUT_DIR"
fi
if [ -n "$POSTSIM_OUTPUT_DIR" ]; then
    echo "Postsim output: $POSTSIM_OUTPUT_DIR"
fi

# -- parse settings fields for run summary when -c is a settings file --------
_IMP_SCENARIO="manifest"
_region="from manifest"
_fleet="from manifest"
if ! _stage_uses_run_manifest; then
    _IMP_SCENARIO="" _RUN_REGION="" _RUN_SCENARIO="" _RUN_YEAR=""
    _INC_PASSENGER="" _INC_FREIGHT=""
    while IFS='=' read -r _k _v; do
        case "$_k" in
            IMP_SCENARIO)  _IMP_SCENARIO="$_v" ;;
            RUN_REGION)    _RUN_REGION="$_v" ;;
            RUN_SCENARIO)  _RUN_SCENARIO="$_v" ;;
            RUN_YEAR)      _RUN_YEAR="$_v" ;;
            INC_PASSENGER) _INC_PASSENGER="$_v" ;;
            INC_FREIGHT)   _INC_FREIGHT="$_v" ;;
        esac
    done < <(awk '
        function val(s,    v) {
            v = s; sub(/^[^:]+:[[:space:]]*/, "", v)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", v); return v
        }
        /^run:/     { top="run";     sub1=""; sub2="" }
        /^impacts:/ { top="impacts"; sub1=""; sub2="" }
        /^[^[:space:]#]/ && !/^run:/ && !/^impacts:/ { top=""; sub1=""; sub2="" }
        top != "" && /^  [^[:space:]]/ {
            sub1=$0; sub(/^[[:space:]]+/,"",sub1); sub(/:.*$/,"",sub1); sub2=""
            if (top=="run") {
                if (sub1=="region")     print "RUN_REGION="   val($0)
                if (sub1=="scenario")   print "RUN_SCENARIO=" val($0)
                if (sub1=="start_year") print "RUN_YEAR="     val($0)
            }
            if (top=="impacts" && sub1=="scenario") print "IMP_SCENARIO=" val($0)
        }
        top != "" && sub1 != "" && /^    [^[:space:]]/ {
            sub2=$0; sub(/^[[:space:]]+/,"",sub2); sub(/:.*$/,"",sub2)
            if (top=="impacts" && sub1=="emissions") {
                if (sub2=="include_passenger") print "INC_PASSENGER=" val($0)
                if (sub2=="include_freight")   print "INC_FREIGHT="   val($0)
            }
        }
    ' "$CONFIG_PATH")

    _fleet=""
    if [ "${_INC_PASSENGER:-true}" != "false" ]; then _fleet="passenger"; fi
    if [ "${_INC_FREIGHT:-true}" != "false" ]; then
        _fleet="${_fleet:+$_fleet  }freight"
    else
        _fleet="${_fleet:+$_fleet  }(freight: off)"
    fi

    _region="${_RUN_REGION:-?}"
    if [ -n "${_RUN_SCENARIO:-}" ]; then _region="$_region / ${_RUN_SCENARIO}"; fi
    if [ -n "${_RUN_YEAR:-}"     ]; then _region="$_region  (${_RUN_YEAR})";    fi
fi

_tdays="${TIME_LIMIT%%-*}"
_thours="${TIME_LIMIT#*-}"; _thours="${_thours%%:*}"; _thours=$(( 10#$_thours ))
if   [ "$_thours" -eq 0 ]; then _tlabel="${_tdays}d"
elif [ "$_tdays"  -eq 0 ]; then _tlabel="${_thours}h"
else                             _tlabel="${_tdays}d ${_thours}h"
fi

printf '\n'
printf '  Scenario : %s\n' "${_IMP_SCENARIO:-?}"
printf '  Region   : %s\n' "$_region"
printf '  Fleet    : %s\n' "${_fleet:-(unknown)}"
printf '  Output   : %s\n' "$JOB_LOG_DIR"
printf '  Stage    : %s\n' "$stage_arg"
printf '  Profile  : %s\n' "$profile_arg"
printf '  Resources: %s  %sG  %s\n' "$PARTITION" "$MEMORY_LIMIT_GB" "$_tlabel"
printf '\n'

if [ "$watch" = true ]; then
    printf '\n'
    [ -n "$JOB_ID" ] && printf '  Job %s queued on %s.\n' "$JOB_ID" "$PARTITION"
    printf '  Ctrl+C to detach at any time — job will keep running.\n\n'

    _detach() {
        printf '\n\n  ── Detached ──────────────────────────────────────────\n'
        printf '  Re-attach : tail -f %s\n' "$JOB_LOG_FILE_PATH"
        [ -n "$JOB_ID" ] && printf '  Status    : squeue -j %s\n' "$JOB_ID"
        [ -n "$JOB_ID" ] && printf '  Cancel    : scancel %s\n' "$JOB_ID"
        printf '  ──────────────────────────────────────────────────────\n\n'
        exit 0
    }
    trap - ERR
    trap _detach INT

    printf '  Waiting for log'
    while [ ! -f "$JOB_LOG_FILE_PATH" ]; do
        printf '.'
        sleep 2
    done
    printf '\n\n'

    tail -f "$JOB_LOG_FILE_PATH" || true
fi
