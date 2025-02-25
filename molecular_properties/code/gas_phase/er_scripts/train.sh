#!/bin/bash

data_path="../../../data/gas_phase/"  # replace to your data path
save_dir="./save_finetune"  # replace to your save path
MASTER_PORT=10086
dict_name="pretrain_dict.txt"
weight_path="../../../weight/pretrain/checkpoint_best.pt"    # replace to your ckpt path
task_name="er" 
task_num=1 
loss_func="finetune_smooth_mae"
consistent_loss=0
consistent_loss_formatted=$(echo "${consistent_loss}" | sed 's/\./_/g')
noise_level=0
only_polar=0
conf_size=1
seed=0
epoch=300    
update_freq=1

lr_values=(0.0003)
local_batch_size_values=(128)
dropout_values=(0.2)
warmup_values=(0.1)
encoder_layers_values=(13)
n_gpu=1

echo "Number of GPUs to be used: $n_gpu"

if [ "$task_name" == "qm7dft" ] || [ "$task_name" == "qm8dft" ] || [ "$task_name" == "qm9dft" ] || [ "$task_name" == "er" ]; then
    metric="valid_agg_mae"
elif [ "$task_name" == "esol" ] || [ "$task_name" == "freesolv" ] || [ "$task_name" == "lipo" ]; then
    metric="valid_agg_rmse"
else 
    metric="valid_agg_auc"
fi

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

for lr in "${lr_values[@]}"; do
  for local_batch_size in "${local_batch_size_values[@]}"; do
    for dropout in "${dropout_values[@]}"; do
      for warmup in "${warmup_values[@]}"; do
        for encoder_layers in "${encoder_layers_values[@]}"; do
          
          global_batch_size=$(($local_batch_size * $n_gpu * $update_freq))
          exp=${task_name}_lr${lr}_bs${global_batch_size}_dropout${dropout}_warmup${warmup}_layers${encoder_layers}
          log_path=./$task_name
          echo "$exp"
          mkdir -p ${log_path}/$exp

          echo "Running experiment with lr=$lr, local_batch_size=$local_batch_size, dropout=$dropout, warmup=$warmup, encoder_layers=$encoder_layers"

          torchrun --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --task-name $task_name --user-dir ../unimol --train-subset train --valid-subset valid \
                 --conf-size $conf_size \
                 --num-workers 8 --ddp-backend=c10d \
                 --dict-name $dict_name \
                 --task mol_finetune --loss $loss_func --arch unimol_base  \
                 --classification-head-name $task_name \
                 --num-classes $task_num \
                 --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
                 --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size --pooler-dropout $dropout \
                 --update-freq $update_freq --seed $seed \
                 --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
                 --tensorboard-logdir "${log_path}/${exp}/tsb" \
                 --log-interval 10 --log-format simple \
                 --validate-interval 1 --keep-last-epochs 1 \
                 --finetune-from-model $weight_path \
                 --best-checkpoint-metric $metric --patience 50 \
                 --save-dir "${log_path}/${exp}" --only-polar $only_polar \
                 --max-atoms 510 \
                 --encoder-layers ${encoder_layers} \
                 --consistent-loss ${consistent_loss} \
                 2>&1 | tee "${log_path}/${exp}/train.log"

        done
      done
    done
  done
done
