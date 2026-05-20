#!/usr/bin/env bash
set -e

CONFIGS=()
SEEDS=()
CKPTS=()
RUN_IDS=()
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
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

if [[ ${#CONFIGS[@]} -eq 0 ]] \
 || [[ ${#CKPTS[@]} -eq 0 ]] \
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

for k_shot in "${K_SHOTS[@]}"; do
    for run_id in "${RUN_IDS[@]}"; do
        for config in "${CONFIGS[@]}"; do
                for ckpt in "${CKPTS[@]}"; do
                    python summarize-partial.py --cfg $config -r $run_id -c $ckpt -k $k_shot \
                    -m yolo_mAP yolo_mAP50 yolo_mAP75 FasterRCNN_mAP FasterRCNN_mAP50 FasterRCNN_mAP75 \
                    FasterRCNN_mAP_airport FasterRCNN_mAP_chimney FasterRCNN_mAP_dam FasterRCNN_mAP_trainstation FasterRCNN_mAP_windmill \
                    FasterRCNN_mAP_corals FasterRCNN_mAP_cuttlefish FasterRCNN_mAP_jellyfish FasterRCNN_mAP_turtle \
                    FasterRCNN_mAP_diver FasterRCNN_mAP_holothurian FasterRCNN_mAP_scallop FasterRCNN_mAP_starfish \
                    FasterRCNN_Subset_mAP FasterRCNN_Subset_mAP50 FasterRCNN_Subset_mAP75 \
                    FasterRCNN_Subset_mAP_airport FasterRCNN_Subset_mAP_chimney FasterRCNN_Subset_mAP_dam FasterRCNN_Subset_mAP_trainstation FasterRCNN_Subset_mAP_windmill \
                    FasterRCNN_Subset_mAP_corals FasterRCNN_Subset_mAP_cuttlefish FasterRCNN_Subset_mAP_jellyfish FasterRCNN_Subset_mAP_turtle \
                    FasterRCNN_mAP_bus FasterRCNN_mAP_dog FasterRCNN_mAP_motorbike FasterRCNN_mAP_table \
                    FID KID_mean\
                    -s "${SEEDS[@]}"
            done
        done
    done
done