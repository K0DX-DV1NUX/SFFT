export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
model_name=FrNet

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1


for seq_len in 336 512 720
do
for pred_len in 48 96
do
    python -u run_longExp.py \
      --is_training 2 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --train_type Linear \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --e_layers 2 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.1\
      --fc_dropout 0.1\
      --head_dropout 0.1\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 50\
      --patience 10\
      --kernel_size 97\
      --lradj type3\
      --pred_head_type 'truncation'\
      --aggregation_type 'avg'\
      --channel_attention 0\
      --global_freq_pred 0\
      --period_list 96 48\
      --emb 96\
      --itr 1 --batch_size 128 --learning_rate 0.0004 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 192
do
    python -u run_longExp.py \
      --is_training 2 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --train_type Linear \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --e_layers 1 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.1\
      --fc_dropout 0.1\
      --head_dropout 0.1\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 50\
      --patience 10\
      --kernel_size 97\
      --lradj type1\
      --pred_head_type 'truncation'\
      --aggregation_type 'avg'\
      --channel_attention 0\
      --global_freq_pred 1\
      --period_list 96 48\
      --emb 96\
      --itr 1 --batch_size 128 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 336 512
do
    python -u run_longExp.py \
      --is_training 2 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --train_type Linear \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --e_layers 1 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.1\
      --fc_dropout 0.1\
      --head_dropout 0.1\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 50\
      --patience 10\
      --kernel_size 97\
      --lradj type3\
      --pred_head_type 'linear'\
      --aggregation_type 'avg'\
      --channel_attention 0\
      --global_freq_pred 1\
      --period_list 96 48\
      --emb 96\
      --itr 1 --batch_size 128 --learning_rate 0.0003 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 720
do
    python -u run_longExp.py \
      --is_training 2 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --train_type Linear \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --e_layers 2 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.1\
      --fc_dropout 0.1\
      --head_dropout 0.1\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 50\
      --patience 10\
      --kernel_size 97\
      --lradj type3\
      --pred_head_type 'linear'\
      --aggregation_type 'avg'\
      --channel_attention 0\
      --global_freq_pred 1\
      --period_list 96 4 24 12\
      --emb 96\
      --itr 1 --batch_size 128 --learning_rate 0.0003 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
done