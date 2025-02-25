data_path='../../data/'
n_gpu=1
layers=13
results_path="./infer"   # replace to your results path
weight_path='../../weight/crystal_ll/checkpoint_best.pt'  # replace to your ckpt path
batch_size=64
task_name='crystal_ll' # data folder name 
task_num=1
dict_name='dict.txt'


MASTER_PORT=10086
torchrun --nproc_per_node=$n_gpu --master_port=$MASTER_PORT ../unimol/infer.py --user-dir ../unimol $data_path --task-name $task_name --valid-subset valid \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task trans_mix_feature_infer --loss trans_loss --arch trans_features \
       --num-classes $task_num \
       --dict-name $dict_name \
       --path $weight_path  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --seed 1 \
       --log-interval 50 --log-format simple  --mol-encoder-layers ${layers}

# R2 for Column 0: 0.9094385951995413
# MAE for column 0: 0.0022530211061879126