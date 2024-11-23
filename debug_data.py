from dataclasses import dataclass, field

import os
import wandb
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import torch.nn.functional as F

from transformers import HfArgumentParser, get_scheduler, AutoTokenizer
from datasets import load_from_disk, load_dataset

@dataclass
class TrainArguments:
    w1: float = field(default=0.5)
    w2: float = field(default=0.5)
    num_epochs: int = field(default=2)
    microbatch_size: int = field(default=12)
    batch_size: int = field(default=50)
    print_every: int = field(default=200)
    lr: float = field(default=1e-5)
    save_every: int = field(default=5_000)
    save_dir: str = field(default="./logs/default")
    warmup_steps: int = field(default=1000)
    start_step: int = field(default=0)
    total_rows: int = field(default=2_000_000)

def data_collator(features):
    input_ids = [feat['text'] for feat in features]
    audio = [feat['audio']['array'] for feat in features]
    return {'texts': input_ids, 'audio': audio}

if __name__ == "__main__":
    parser = HfArgumentParser((TrainArguments))
    model_args, = parser.parse_args_into_dataclasses()
    
    if not os.path.isdir(model_args.save_dir):
        os.mkdir(model_args.save_dir)

    dataset = load_dataset('alvanlii/audio-llm-train', split='train', streaming=True) # load_from_disk('./data/combined_english_canto')
    LEN_DATASET = model_args.total_rows

    train_dataloader = DataLoader(
        dataset,
        batch_size=model_args.microbatch_size,
        collate_fn=data_collator
    )

    bs = model_args.batch_size
    curr_steps = 0
    scaler = amp.GradScaler()
    curr_idx = 0

    for epoch in range(model_args.num_epochs):
        for i, batch in tqdm(enumerate(train_dataloader), total=LEN_DATASET // model_args.microbatch_size):
            curr_steps += 1
            if model_args.start_step > 0 and curr_steps < model_args.start_step:
                continue

            audio, input_ids, attention_mask = batch['audio'], batch['texts'], batch['attention_mask']

            if (139_800 < curr_steps < 145_800):
                with open(f"./debug/audio_{curr_idx}.npy", "wb") as f:
                    np.save(f, audio.cpu().detach().numpy())
                with open(f"./debug/texts_{curr_idx}.txt", "w") as f:
                    f.write(input_ids)
                curr_idx += 1

            if (curr_steps > 145_800):
                raise Exception("done")
