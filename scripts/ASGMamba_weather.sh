export CUDA_VISIBLE_DEVICES=0

model_name=ASGMamba

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --learning_rate 0.00005 \
  --train_epochs 5\
  --d_state 2 \
  --d_ff 512\
  --itr 1  >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'96.log


python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --learning_rate 0.00005 \
  --train_epochs 5\
  --d_model 512\
  --d_state 2 \
  --d_ff 512\
  --itr 1  >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'192.log


python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --learning_rate 0.00005 \
  --train_epochs 5\
  --d_model 512\
  --d_state 2 \
  --d_ff 512\
  --itr 1  >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'336.log


python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --learning_rate 0.00005 \
  --train_epochs 5\
  --d_model 512\
  --d_state 2 \
  --d_ff 512\
  --itr 1  >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'720.log