export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TORCHDYNAMO_DISABLE="1"
export TORCH_EXTENSIONS_DIR="/data/workspace/howard/workspace/.cache"

export OMP_NUM_THREADS=2

LOG_DIR="/data/workspace/howard/workspace/Llava/log"

    # --per_device_eval_batch_size=2 \

deepspeed --include=localhost:0,1 --master_port=10234 \
    /data/workspace/howard/workspace/Llava/llava_stage_2.0_train.py \
    --output_dir=/data/workspace/howard/workspace/Llava/output_dir/stage_2_table_vqa_ko \
    --run_name=llava-2.0 \
    --cache_dir=/data/workspace/howard/workspace/.cache/.llava_2_preprocess \
    --model_name_or_path=jp1924/koGemma2-it-KoLLaVA-9b-stage1.0 \
    --cache_file_name=preprocessor.arrow \
    --preprocessing_batched=true \
    --preprocessing_num_workers=16 \
    --preprocessing_batch_size=100 \
    --dataset_repo_ls \
        kaki-paper/table-vqa-ko-converted \
    --train_dataset_prefix=train \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=1 \
    --seed=42 \
    --do_train=true \
    --do_eval=false \
    --do_predict=false \
    --report_to=none \
    --learning_rate=2e-5 \
    --lr_scheduler_type=cosine \
    --warmup_ratio=0.03 \
    --weight_decay=0 \
    --save_strategy=steps \
    --eval_steps=1000 \
    --eval_strategy=no \
    --save_steps=1000 \
    --logging_strategy=steps \
    --logging_steps=1 \
    --bf16=true \
    --tf32=true \
    --use_liger_kernel=true \
    --gradient_checkpointing=true \
    --gradient_checkpointing_kwargs='{"use_reentrant": false}' \
    --group_by_length=false \
    --attn_implementation='eager' \
    --vision_feature_select_strategy='full' \
    --use_liger_kernel=true \
    --torch_empty_cache_steps=50 \
    --torch_compile=true \
    --deepspeed=/data/workspace/howard/workspace/Llava/ds_config/ZeRO_3_act_check.json \
    --response_template='[6176, 18145, 235292, 108]' \
    --data_max_length=1500 \
    --instruction_template='[106, 6176, 4926, 235292, 108]' \
    > $LOG_DIR/kogemma2_llava_stage_2.0_train.log 2>&1