#!/bin/bash

# HPC execution entrypoint for a scheduled Slurm job.

set -euo pipefail

error_exit() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
    echo "ERROR: job.sh failed!" >&2
    echo "  Line: $1" >&2
    echo "  Command: $2" >&2
    echo "  Exit code: $3" >&2
    echo "  Job log: ${JOB_LOG_FILE_PATH:-<not set>}" >&2
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
    exit "$3"
}

trap 'error_exit ${LINENO} "$BASH_COMMAND" $?' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMPACTS_DIR="${IMPACTS_DIR:-$REPO_ROOT}"
IMPACTS_DIR="$(cd "$IMPACTS_DIR" && pwd)"
export IMPACTS_DIR
VENV_PATH="${IMPACTS_VENV_PATH:-$IMPACTS_DIR/venv_hpc}"
REQUIREMENTS_FILE="${IMPACTS_REQUIREMENTS_FILE:-$IMPACTS_DIR/hpc/requirements-hpc.txt}"

show_system_info() {
    echo "=== MEMORY INFORMATION ==="
    free -h
    echo "=== CPU INFORMATION ==="
    nproc
    echo "=========================="
}

install_python_deps() {
    local req_file="$1"
    local setup_file="$IMPACTS_DIR/setup.cfg"
    local marker="$VENV_PATH/.last_requirements_hash"
    local current_hash
    # The editable project install uses --no-deps below, so dependency metadata
    # changes in setup.cfg must also invalidate the HPC requirements marker.
    if [ -f "$setup_file" ]; then
        current_hash="$({ sha256sum "$req_file"; sha256sum "$setup_file"; } | sha256sum | awk '{print $1}')"
    else
        current_hash="$(sha256sum "$req_file" | awk '{print $1}')"
    fi

    if [ ! -f "$marker" ] || [ "$current_hash" != "$(cat "$marker")" ]; then
        echo "Installing/updating Python dependencies from $req_file and setup.cfg ..."
        "$VENV_PATH/bin/python3" -m pip install --upgrade pip setuptools wheel
        "$VENV_PATH/bin/python3" -m pip install -r "$req_file"
        printf "%s\n" "$current_hash" > "$marker"
    else
        echo "Python dependencies up to date; skipping pip install."
    fi
}

configure_python_geospatial_data_paths() {
    local pyproj_data_dir
    local gdal_data_dir

    # Use sys.prefix (venv root) to build paths — avoids .resolve() following
    # symlinks into a different venv and producing PROJ database version mismatches.
    pyproj_data_dir="$("$VENV_PATH/bin/python3" -c '
import sys
from pathlib import Path
py_ver = "python{}.{}".format(*sys.version_info[:2])
p = Path(sys.prefix) / "lib" / py_ver / "site-packages" / "pyproj" / "proj_dir" / "share" / "proj"
print(str(p))
')"
    if [ ! -f "$pyproj_data_dir/proj.db" ]; then
        echo "ERROR: pyproj PROJ database not found: $pyproj_data_dir/proj.db" >&2
        exit 1
    fi

    export PROJ_DATA="$pyproj_data_dir"
    export PROJ_LIB="$pyproj_data_dir"

    gdal_data_dir="$("$VENV_PATH/bin/python3" -c '
import sys
from pathlib import Path
py_ver = "python{}.{}".format(*sys.version_info[:2])
p = Path(sys.prefix) / "lib" / py_ver / "site-packages" / "rasterio" / "gdal_data"
print(str(p) if p.is_dir() else "")
')"
    if [ -d "$gdal_data_dir" ]; then
        export GDAL_DATA="$gdal_data_dir"
        echo "GDAL data: $GDAL_DATA"
    fi

    "$VENV_PATH/bin/python3" -c 'import pyproj; pyproj.CRS.from_epsg(3857); print(f"PROJ data: {pyproj.datadir.get_data_dir()}")'
}

if [ "${1:-}" = "" ]; then
    echo "Usage: $0 <settings-or-manifest> [stage]"
    exit 2
fi

CONFIG_FILE="$1"
STAGE="${2:-pipeline}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: settings or manifest file not found: $CONFIG_FILE"
    exit 1
fi

echo "Setting up HPC runtime environment..."
set +u
module load gcc/11.4.0
module load proj/9.2.1
module load python/3.11.6
set -u

export LD_LIBRARY_PATH=/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/gcc-11.4.0-nfcdl6bpyabpnhhasfzu6y4ge4kfskvl/lib64:${LD_LIBRARY_PATH:-}

cd "$IMPACTS_DIR"

if [ ! -x "$VENV_PATH/bin/python3" ]; then
    echo "Creating virtual environment at $VENV_PATH ..."
    python3 -m venv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "ERROR: requirements file not found: $REQUIREMENTS_FILE"
    exit 1
fi

install_python_deps "$REQUIREMENTS_FILE"
configure_python_geospatial_data_paths

# Install impacts itself in editable mode
"$VENV_PATH/bin/python3" -m pip install -e "$IMPACTS_DIR" --no-deps --quiet

echo "Python: $("$VENV_PATH/bin/python3" --version) @ $VENV_PATH/bin/python3"
echo "Config: $CONFIG_FILE"

show_system_info

# Cap thread pools to avoid oversubscription
THREADS="${IMPACTS_THREADS:-8}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export BLIS_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
echo "Thread caps: $THREADS (OMP/MKL/OPENBLAS/NUMEXPR/BLIS/VECLIB)"

PROFILE_MODE="${IMPACTS_PROFILE:-none}"
PROFILE_ARGS=()
if [ "$PROFILE_MODE" != "none" ]; then
    PROFILE_ARGS=(--profile "$PROFILE_MODE")
fi
echo "Profile: $PROFILE_MODE"

echo "Stage: $STAGE"
case "$STAGE" in
    pipeline)
        "$VENV_PATH/bin/python3" -u -m impacts pipeline --config "$CONFIG_FILE" "${PROFILE_ARGS[@]}"
        ;;
    preprocess|presim|activities|postsim)
        "$VENV_PATH/bin/python3" -u -m impacts "$STAGE" --config "$CONFIG_FILE" "${PROFILE_ARGS[@]}"
        ;;
    fleet)
        "$VENV_PATH/bin/python3" -u -m impacts fleet --activities-manifest "$CONFIG_FILE" "${PROFILE_ARGS[@]}"
        ;;
    emissions|inmap|aermod|exposure|postprocess)
        "$VENV_PATH/bin/python3" -u -m impacts "$STAGE" --run-manifest "$CONFIG_FILE" "${PROFILE_ARGS[@]}"
        ;;
    *)
        echo "ERROR: unsupported stage '$STAGE'"
        echo "Supported: pipeline, preprocess, presim, activities, postsim, fleet, emissions, inmap, aermod, exposure, postprocess"
        exit 2
        ;;
esac

if [ "$STAGE" = "pipeline" ]; then
    echo "Pipeline job complete: stage=$STAGE log=${JOB_LOG_FILE_PATH:-<not set>}"
else
    echo "Stage job complete: stage=$STAGE log=${JOB_LOG_FILE_PATH:-<not set>}"
fi
