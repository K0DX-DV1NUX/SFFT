export CUDA_VISIBLE_DEVICES=5
# add for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_fix/traf_abl" ]; then
    mkdir ./logs/FITS_fix/traf_abl
fi
seq_len=700
model_name=FITS

for seq_len in 336 512 720
do
for pred_len in 48 96 192 336 512 720
do
python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_'$predlen \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 1 \
  --des 'Exp' \
  --train_mode 2 \
  --H_order 10 \
  --gpu 0 \
  --patience 10 \
  --train_epochs 50 \
  --itr 1 --batch_size 64 --learning_rate 0.0005
done
done