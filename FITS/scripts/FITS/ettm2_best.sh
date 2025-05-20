export CUDA_VISIBLE_DEVICES=7
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_fix/ettm2_abl" ]; then
    mkdir ./logs/FITS_fix/ettm2_abl
fi
#seq_len=700
model_name=FITS

for seq_len in 336 512 720
do
for pred_len in 48 96 192 336 512 720
do
python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features S \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 1 \
  --des 'Exp' \
  --train_mode 2 \
  --H_order 14 \
  --base_T 96 \
  --gpu 0 \
  --patience 20\
  --train_epochs 50 \
  --itr 1 --batch_size 64 --learning_rate 0.0005
done
done
