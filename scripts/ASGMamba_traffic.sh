export CUDA_VISIBLE_DEVICES=0

model_name=ASGMamba

python -u run_longExp.py  \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1 >logs/LongForecasting/$model_name'_'traffic_$seq_len'_'96.log 

python -u run_longExp.py  \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1 >logs/LongForecasting/$model_name'_'traffic_$seq_len'_'192.log 

python -u run_longExp.py  \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 16 \
  --learning_rate 0.002 \
  --itr 1 >logs/LongForecasting/$model_name'_'traffic_$seq_len'_'336.log 

python -u run_longExp.py  \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 16 \
  --learning_rate 0.0008\
  --itr 1 >logs/LongForecasting/$model_name'_'traffic_$seq_len'_'720.log 