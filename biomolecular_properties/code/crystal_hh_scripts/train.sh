### gpu configs
MASTER_PORT=10068
data_path='../../data/'
save_dir="./save_finetune" 
finetune_mol_model='../../weight/pretrain/checkpoint_best.pt'

learning_rates=(0.001)
batch_sizes=(16)
layer_numbers=13 
dropout_rates=(0)
task_num=1
epoch=120
task_name='crystal_hh'
warmup=0.06
update_freq=1
log_path=.
n_gpu=8 #8


if [ "$task_name" == "qm7dft" ] || [ "$task_name" == "qm8dft" ] || [ "$task_name" == "qm9dft" ] || [ "$task_name" == "opv" ] || [ "$task_name" == "opv_rdkit_dft" ] || [ "$task_name" == "crystal_hh" ]; then
	metric="valid_mae"
    
elif [ "$task_name" == "esol" ] || [ "$task_name" == "freesolv" ] || [ "$task_name" == "lipo" ]; then
    metric="valid_agg_rmse"
else 
    metric="valid_agg_auc"
fi

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

for lr in "${learning_rates[@]}"
do
    for batch_size in "${batch_sizes[@]}"
    do
        for layers in "${layer_numbers[@]}"
        do
            for dropout in "${dropout_rates[@]}"
            do
                global_batch_size=$((batch_size * n_gpu * update_freq))
                echo "lr_${lr}_bsz_${global_batch_size}_layers_${layers}_dropout_${dropout}"
                exp=${epoch}_lr_${lr}_bsz_${global_batch_size}_dropout_${dropout}_warmup_${warmup}_gpu_${n_gpu}_layers_${layers}
                exp_dir="${log_path}/${exp}"
                mkdir -p ${exp_dir}
                torchrun  --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ../unimol --train-subset train --valid-subset valid \
                    --num-workers 8 --ddp-backend=c10d \
                    --task-name $task_name \
                    --dict-name dict.txt \
                    --task trans_mix_feature --loss trans_loss --arch trans_features  \
                    --num-classes $task_num \
                    --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
                    --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size \
                    --mol-pooler-dropout $dropout --pocket-pooler-dropout $dropout \
                    --update-freq $update_freq --seed 1 \
                    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
                    --log-interval 50 --log-format simple \
                    --patience 2000 \
                    --finetune-mol-model $finetune_mol_model \
                    --save-dir "${exp_dir}" \
                    --find-unused-parameters \
                    --validate-interval 1 --keep-last-epochs 1 \
                    --max-atoms 512 \
                    --mol-encoder-layers ${layers} \
                    --pocket-encoder-layers ${layers} \
                    --tensorboard-logdir "${exp_dir}/tsb" \
                    2>&1 | tee "${exp_dir}/train.log"
            done
        done
    done
done


