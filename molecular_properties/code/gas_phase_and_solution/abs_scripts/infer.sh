data_path="../../../data/properties_in_solution/"
n_gpu=1
layers=14

results_path="./infer"  # replace to your results path
weight_path="../../../weight/properties_in_solution/e_abs/checkpoint_best.pt"  # replace to your ckpt path
batch_size=16
task_name='e_abs' 
task_num=1
loss_func='finetune_smooth_mae_infer'
dict_name='pretrain_dict.txt'

MASTER_PORT=10086

torchrun --nproc_per_node=$n_gpu --master_port=$MASTER_PORT ../unimol/infer.py --user-dir ../unimol $data_path --task-name $task_name --valid-subset valid \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task trans_mix_feature_infer --loss finetune_smooth_mae_infer --arch trans_features5_deepchem \
       --classification-head-name 'deep4chem_abs_single_tower_v1_feature5' --num-classes $task_num \
       --dict-name $dict_name \
       --finetune-mol-model $weight_path  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --seed 1 \
       --log-interval 50 --log-format simple  --mol-encoder-layers ${layers}

# R2 for Column 0: 0.9816262012082811
# MAE for column 0: 7.084714559094001
# (3353, 4)