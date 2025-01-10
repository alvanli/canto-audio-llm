LOG_DIR="./logs/v08"

mkdir $LOG_DIR
cp train.py "$LOG_DIR/"
cp -r model "$LOG_DIR/"

HF_HUB_ETAG_TIMEOUT=10000 CUDA_LAUNCH_BLOCKING=1 python3 train.py \
    --num_epochs 1 \
    --microbatch_size 3 \
    --batch_size 300 \
    --save_dir "$LOG_DIR" \
    --print_every 300 \
    --save_every 2400 \
    --lr 5e-05 \
    --warmup_steps 4000 \
    --total_rows 1000000 \
    --w2 1 \
    --w1 1 \
    --start_step 139200
