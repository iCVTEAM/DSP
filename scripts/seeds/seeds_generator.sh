#!/usr/bin/env bash
set -e

SEEDS=()
METASEED=""
NUM_SEED=-1
META_START=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --metaseed)
            shift
            METASEED=$1
            ;;
        --num_seed)
            shift
            NUM_SEED=$1
            ;;
        --meta_start)
            shift
            META_START=$1
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

if [[ -z "$METASEED" ]]; then
    echo "Error: --metaseed must be provided."
    exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_PROCESSES=${#GPU_ARRAY[@]}

if [[ -n "$METASEED" ]]; then
    if [[ $NUM_SEED -lt 0 ]]; then
        echo "Error: you must set --num_seed when using --metaseed."
        exit 1
    fi

    echo "[INFO] Using metaseed=$METASEED"
    echo "[INFO] Will use seeds index range [$META_START, $NUM_SEED)"

    mapfile -t ALL_SEEDS < <(
        shuf -i 0-9999 \
            --random-source=<(awk -v s="$METASEED" 'BEGIN { while (1) printf "%s", s }') \
            | head -n $NUM_SEED
    )

    # SEEDS=("${ALL_SEEDS[@]:META_START:NUM_SEED-META_START}")
    SEEDS=("${ALL_SEEDS[@]:$META_START:$((NUM_SEED - META_START))}")

    echo "[INFO] Total Seeds Generated: ${#ALL_SEEDS[@]}"
    echo "[INFO] Using Seeds: ${SEEDS[@]}"
fi

> "seeds-$METASEED.txt"

for seed in "${SEEDS[@]}"; do
    echo $seed >> "seeds-$METASEED.txt"
done