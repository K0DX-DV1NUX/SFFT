#!/bin/sh

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=SFFT

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

for pred_len in 96 192 336 720
do
for seq_len in 336 512 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --individual 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --train_type Linear \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --train_epochs 100 \
      --rank 30 \
      --bias 1 \
      --enable_lowrank 1 \
      --sym_regularizer 1 \
      --decomposer_depth 5 \
      --seasons 3 \
      --kernel_size 70 \
      --patience 20 \
      --des 'Exp' \
      --regularizer 0 \
      --itr 1 \
      --batch_size 32 \
      --learning_rate 0.01
done
done
