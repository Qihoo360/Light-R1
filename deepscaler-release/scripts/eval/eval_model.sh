set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Default values
MODEL_PATH="$HOME/DeepScaleR-1.5B-Preview"
# Possible values: aime, amc, math, minerva, olympiad_bench
DATATYPES=("aime")
OUTPUT_DIR="$HOME"  # Add default output directory

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --datasets)
            # Convert space-separated arguments into array
            shift
            DATATYPES=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                DATATYPES+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --model <model_path> --datasets dataset1 dataset2 ... --output-dir <output_directory>"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"
PROCESSED_DATA_PATH="${SCRIPT_DIR}/processed_data"

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Processed Data Path: ${PROCESSED_DATA_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"

# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        data.path=${PROCESSED_DATA_PATH}/${DATA_TYPE}.parquet \
        data.output_path=${OUTPUT_DIR}/${DATA_TYPE}.parquet \
        data.n_samples=1 \
        data.batch_size=1024 \
        model.path=${MODEL_PATH} \
        rollout.temperature=0.6 \
        rollout.response_length=16384 \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.max_model_len=8192 \
        rollout.gpu_memory_utilization=0.8 \
        rollout.tensor_model_parallel_size=2 \
        actor.strategy=fsdp \
        actor.ulysses_sequence_parallel_size=1 \
        actor.fsdp_config.fsdp_size=-1 \
        rollout.disable_log_stats=True \
        rollout.enable_chunked_prefill=True \
        rollout.n=1 \
        +actor_rollout_ref.rollout.enable_chunked_prefill=False \
        +actor.optim.lr=1e-3 \
        +data.skip_format_reward=True \
        +model.trust_remote_code=True 

    python light_r1_postprocess.py \
        --model_path=${MODEL_PATH} \
        --save_path=${OUTPUT_DIR} \
        --benchmark=${DATA_TYPE}
done
# +data.skip_format_reward=True是默认行为，跳过校验答案正确性时的格式检查，没<think>也没事
# nnodes增大，则可增大gpu_memory_utilization至0.9-0.95
# 如遇OOM，一般减小gpu_memory_utilization即可
