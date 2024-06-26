#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --time=24:00:00            
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=32           
#SBATCH --mem=80G                
#SBATCH --job-name=test
#SBATCH --gres=gpu:1 

source /etc/profile.d/modules.sh
module use /opt/nvidia/hpc_sdk/modulefiles/
module load nvhpc
source /u/area/ddoimo/anaconda3/bin/activate ./env_amd

#conda activate ./env_amd
export OMP_NUM_THREADS=32

#export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
#echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    #--use_deepspeed \
    #--deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path  /u/area/ddoimo/ddoimo/llama/llama_v2/models_hf/llama-2-7b \
    --use_flash_attn \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --tokenizer_name /u/area/ddoimo/ddoimo/llama/llama_v2/models_hf/llama-2-7b \
    --use_slow_tokenizer \
    --train_file /orfeo/cephfs/scratch/area/ddoimo/open-instruct/data/processed/dolly/dolly_data.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --eval_every 50 \
    --output_dir ./results/${MODEL_SIZE}_lora/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 &&

python open_instruct/merge_lora.py \
    --base_model_name_or_path ../hf_llama2_models/${MODEL_SIZE} \
    --lora_model_name_or_path $RESULTS_FOLDER \
    --output_dir ./results/${MODEL_SIZE}_lora_merged/ \
    --save_tokenizer
