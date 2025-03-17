# Light-R1 SFT used a slightly different internal version codebase. This script is the closest counterpart in 360-LLaMA-Factory
# Light-R1 DPO used 360-LLaMA-Factory directly
hostfile="hostfile.8nodes"
deepspeed --hostfile $hostfile src/train.py \
    --stage sft \
    --do_train \
    --max_steps -1 \
    --model_name_or_path [your sft stage1 model path] \
    --template qwen \
    --dataset [your sft stage2 data name] \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --sequence_parallel_size 1 \
    --gradient_checkpointing True \
    --flash_attn fa2  \
    --cache_dir .cache \
    --overwrite_cache \
    --cutoff_len 20000 \
    --output_dir [your output dir] \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --save_strategy epoch \
    --logging_steps 1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 0.1 \
    --warmup_ratio 0.01 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --save_only_model True \
    --num_train_epochs 100 \
    --bf16 true \
    --plot_loss \
    --seed 42 \
    --do_eval false \
    --deepspeed ./examples/deepspeed/ds_z3_offload_config.json \
    --report_to tensorboard \
    --overwrite_output_dir \
    --ddp_timeout 180000000 \
    --enable_liger_kernel
# num_train_epochs 100, with simultaneous model selection
# stage2 Light-R1 chose epoch 8