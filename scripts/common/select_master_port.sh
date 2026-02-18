#!/bin/bash

set -euo pipefail

min_port="${1:-9900}"
max_port="${2:-9999}"
job_id="${SLURM_JOB_ID:-0}"

if ! [[ "${min_port}" =~ ^[0-9]+$ && "${max_port}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Port bounds must be integers. Got: min='${min_port}', max='${max_port}'." >&2
    exit 1
fi

if (( min_port > max_port )); then
    echo "ERROR: Invalid port range ${min_port}-${max_port}." >&2
    exit 1
fi

if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    job_id=0
fi

if ! command -v ss >/dev/null 2>&1; then
    echo "ERROR: 'ss' command not found; cannot check for open ports." >&2
    exit 1
fi

is_port_in_use() {
    local port="$1"
    ss -lntu 2>/dev/null | grep -q ":${port} "
}

range_size=$((max_port - min_port + 1))
ideal_port=$((min_port + (job_id % range_size)))
selected_port=""

if ! is_port_in_use "${ideal_port}"; then
    selected_port="${ideal_port}"
else
    for port in $(seq "${min_port}" "${max_port}"); do
        if ! is_port_in_use "${port}"; then
            selected_port="${port}"
            break
        fi
    done
fi

if [[ -z "${selected_port}" ]]; then
    echo "ERROR: Could not find any free port in the range ${min_port}-${max_port}." >&2
    exit 1
fi

echo "${selected_port}"
