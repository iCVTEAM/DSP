#!/usr/bin/env bash
set -e

CONFIGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            shift
            CONFIGS=($1)
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

echo "CONFIGS:   ${CONFIGS[@]}"

run_with_retry(){
    local cmd=("$@")
    local max=3
    local attempt=1

    while (( attempt <= max )); do
        echo "Attempt $attempt/$max: ${cmd[*]}"
        
        set +e
        "${cmd[@]}"
        status=$?
        set -e

        if [[ $status -eq 0 ]]; then
            return 0
        fi

        ((attempt++))
        sleep 2
    done

    return 1
}

for config in "${CONFIGS[@]}"; do
        echo "config: $config"

        run_with_retry \
        accelerate launch --multi_gpu --gpu_ids 0,1,2,3 --num_processes 4 main.py --config ./configs/$config.yaml -m train -p base
done