HF_HUB_ETAG_TIMEOUT=1000 python3 train.py \
    --num_epochs 1 \
    --microbatch_size 1 \
    --batch_size 600 \
    --save_dir "./logs/v03" \
    --print_every 600 \
    --save_every 2400 \
    --lr 4e-05 \
    --warmup_steps 3000 \
    --start_step 30000
