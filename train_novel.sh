#!/usr/bin/env bash
set -e

CONFIGS=()
SEEDS=()
RUN_IDS=()
GPU_IDS=""
METASEED=""
NUM_SEED=-1
META_START=0
K_SHOTS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            shift
            CONFIGS=($1)
            ;;
        --seed)
            shift
            SEEDS=($1)
            ;;
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
        --run_id)
            shift
            RUN_IDS=($1)
            ;;
        --k_shot)
            shift
            K_SHOTS=($1)
            ;;
        --gpu_ids)
            shift
            GPU_IDS=$1
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

if [[ ${#CONFIGS[@]} -eq 0 ]] \
 || [[ -z "$GPU_IDS" ]] \
 || [[ ${#K_SHOTS[@]} -eq 0 ]] \
 || [[ ${#RUN_IDS[@]} -eq 0 ]]; then
    echo "Error: missing required arguments."
    echo "You must provide: --config, --gpu_ids, --run_id, --k_shot"
    echo "And one of: --seed OR --metaseed"
    exit 1
fi

if [[ -z "$METASEED" && ${#SEEDS[@]} -eq 0 ]]; then
    echo "Error: either --seed or --metaseed must be provided."
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

echo "CONFIGS:   ${CONFIGS[@]}"
echo "SEEDS:     ${SEEDS[@]}"
echo "RUN_IDS:    ${RUN_IDS[@]}"
echo "K_SHOTS:    ${K_SHOTS[@]}"
echo "GPU_IDS:   ${GPU_IDS}"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "META_START: $META_START"
echo "NUM_SEED: $NUM_SEED"
echo "Actual training episodes: ${#SEEDS[@]}"

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

for k_shot in "${K_SHOTS[@]}"; do
    for run_id in "${RUN_IDS[@]}"; do
        for config in "${CONFIGS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                echo "config: $config, seed: $seed, run: $run_id, k_shot: $k_shot"

                run_with_retry \
                accelerate launch --multi_gpu --gpu_ids $GPU_IDS --num_processes $NUM_PROCESSES main.py \
                --config ./configs/$config.yaml -m train -p novel -s $seed -k $k_shot -r $run_id
            done
        done
    done
done