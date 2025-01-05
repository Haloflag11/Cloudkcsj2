export CUDA_VISIBLE_DEVICES=0
spark-submit \
    --master spark://autodl-container-0f0c439491-7b62fad8:7077 \
    --deploy-mode client \
    --num-executors 4 \
    --executor-memory 8G \
    --executor-cores 24 \
    --driver-memory 16G \
    run.py \
    --is_training 1 \
    --root_path /root/autodl-tmp/Dataset/BMY/ \
    --data_path BMY_data.csv \
    --model_id BMY_120_12 \
    --model S3_Mamba \
    --data custom \
    --features M \
    --seq_len 120 \
    --pred_len 12 \
    --e_layers 2 \
    --enc_in 64 \
    --dec_in 64 \
    --c_out 64 \
    --des 'Exp' \
    --d_model 256 \
    --d_ff 512 \
    --d_state 16 \
    --train_epochs 10 \
    --batch_size 16 \
    --learning_rate 0.00005 \
    --itr 1

spark-submit \
    --master spark://autodl-container-0f0c439491-7b62fad8:7077 \
    --deploy-mode client  \
    --num-executors 4 \
    --executor-memory 8G \
    --executor-cores 24 \
    --driver-memory 16G \
    run.py \
    --is_training 1 \
    --root_path /root/autodl-tmp/Dataset/BMY/ \
    --data_path BMY_data.csv \
    --model_id BMY_120_32 \
    --model S3_Mamba \
    --data custom \
    --features M \
    --seq_len 120 \
    --pred_len 24 \
    --e_layers 2 \
    --enc_in 64 \
    --dec_in 64 \
    --c_out 64 \
    --des 'Exp' \
    --d_model 256 \
    --d_ff 512 \
    --d_state 16 \
    --batch_size 16 \
    --train_epochs 15 \
    --learning_rate 0.00005 \
    --itr 1
  # python -u run.py \
  # --is_training 1 \
  # --root_path ./dataset/electricity/ \
  # --data_path electricity.csv \
  # --model_id ECL_96_336 \
  # --model $model_name \
  # --data custom \
  # --features M \
  # --seq_len 96 \
  # --pred_len 336 \
  # --e_layers 3 \
  # --enc_in 321 \
  # --dec_in 321 \
  # --c_out 321 \
  # --des 'Exp' \
  # --d_model 512 \
  # --d_ff 512 \
  # --d_state 16 \
  # --batch_size 16 \
  # --train_epochs 5 \
  # --learning_rate 0.0005 \
  # --itr 1
#  python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/electricity/ \
#  --data_path electricity.csv \
#  --model_id ECL_96_720 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 720 \
#  --e_layers 3 \
#  --enc_in 321 \
#  --dec_in 321 \
#  --c_out 321 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_state 16 \
#  --d_ff 512 \
#  --train_epochs 5 \
#  --batch_size 16 \
#  --learning_rate 0.0005 \
#  --itr 1

