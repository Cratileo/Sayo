export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd PATH/YOUR/DIR
accelerate launch \
    --config_file config/deepspeed_zero1.yaml \
    main.py \
    --model_name_or_path PATH/YOUR/MODEL \
    --dataset_name PATH/YOUR/DATASETS \
    --output_dir PATH/YOUR/OUTPUT/DIR \
    --learning_rate 5e-6 \
    --dtype bfloat16 \
    --max_prompt_length 6500 \
    --max_completion_length 110 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules "all-linear" \
    --log_completions \
    --report_to tensorboard \
    --attn_implementation flash_attention_2 \
    --per_device_train_batch_size 64 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 10 \
    --num_train_epochs 3 \
    --beta 0.001 \
    --num_generations 16 \
    --temperature 1.0 \
    --top_p 0.9 \
    --weight_decay 1e-2 \
    --max_grad_norm 0.8 \
    --dataloader_num_workers 16
