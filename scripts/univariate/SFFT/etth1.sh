if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=FragFM

root_path_name=../dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1


for pred_len in 48 96 192 336 720
do
for seq_len in  336 512 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --individual 0 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features S \
      --train_type Linear \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --train_epochs 50 \
      --patience 10 \
      --bias 0 \
      --enable_lowrank 1 \
      --rank 25 \
      --decomposer_depth 2 \
      --seasons 2 \
      --kernel_size 70 \
      --des 'Exp' \
      --reg 1 \
      --reg_rate 1.0 \
      --itr 1 \
      --batch_size 128 \
      --num_workers 0 \
      --lradj 7 \
      --learning_rate 0.05
done
done


