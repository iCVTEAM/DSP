#!/usr/bin/env bash
set -e

CONFIGS=()
SEEDS=()
CKPTS=()
RUN_IDS=()
GPU_IDS=""
METASEED=""
NUM_SEED=-1
META_START=0
K_SHOTS=()
VIZ=0

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
        --ckpt)
            shift
            CKPTS=($1)
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
        --viz)
            VIZ=1
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

if [[ ${#CONFIGS[@]} -eq 0 ]] \
 || [[ ${#CKPTS[@]} -eq 0 ]] \
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
echo "CKPTS:     ${CKPTS[@]}"
echo "GPU_IDS:   ${GPU_IDS}"

for k_shot in "${K_SHOTS[@]}"; do
    for run_id in "${RUN_IDS[@]}"; do
        for config in "${CONFIGS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for ckpt in "${CKPTS[@]}"; do
                    echo "config: $config, seed: $seed, ckpt: $ckpt, run: $run_id, k_shot: $k_shot"

                    EXP_DIR="$DSP_PROJECT_DIR/outputs/$config/novel/run-$run_id/$k_shot-shot/shuffle_seed-$seed/checkpoint-$ckpt"

                    if [[ "$VIZ" -eq 1 ]]; then
                        VIZ_DIR="$EXP_DIR/FasterRCNN_Viz"
                    else
                        VIZ_DIR=""
                    fi

                    CUDA_VISIBLE_DEVICES=$GPU_IDS \
                    bash test-exdark.sh \
                        "$EXP_DIR" "$VIZ_DIR"
                done
            done
        done
    done
done