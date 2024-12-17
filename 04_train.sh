LOG_DIR="./logs/v07"

mkdir $LOG_DIR
cp train.py "$LOG_DIR/"
cp -r model "$LOG_DIR/"

HF_HUB_ETAG_TIMEOUT=10000 CUDA_LAUNCH_BLOCKING=1 python3 train.py \
    --num_epochs 1 \
    --microbatch_size 1 \
    --batch_size 600 \
    --save_dir "$LOG_DIR" \
    --print_every 600 \
    --save_every 2400 \
    --lr 2e-05 \
    --warmup_steps 3000 \
    --total_rows 1000000 \
    --w2 1 \
    --w1 5
