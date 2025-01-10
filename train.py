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

from transformers import HfArgumentParser, get_scheduler
from datasets import load_from_disk, load_dataset
from model.model import DiVAModel

def log_metrics(curr_steps, all_l2_losses, all_kl_losses, scheduler, bs):
    if len(all_l2_losses) < bs:
        l2_loss = np.mean(all_l2_losses)
        kl_loss = np.mean(all_kl_losses)
    else:
        l2_loss = np.mean(all_l2_losses[-bs:])
        kl_loss = np.mean(all_kl_losses[-bs:])

    lr = scheduler.get_last_lr()[0]
    log_dict = {
        "step": curr_steps,
        "l2_loss": l2_loss,
        "kl_loss": kl_loss,
        "lr": lr
    }
    print(log_dict)
    wandb.log(log_dict)
    return

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


def train_step(model, batch, args):
    audio, input_ids, attention_mask = batch['audio'], batch['input_ids'], batch['attention_mask']
    
    audio_embed, text_embed, audio_response, text_response, text_attention_left_pad = model.train_forward(
        audio, input_ids, attention_mask
    )
    
    diff_distill = audio_response[:, -1] - text_response[:, -1]
    kl_loss = torch.sqrt(torch.sum(diff_distill * diff_distill, dim=-1)).mean()
    

    shifted_loss_mask = torch.roll(text_attention_left_pad, shifts=1, dims=1)
    one_hot_first = torch.zeros_like(text_attention_left_pad)
    one_hot_first[:, 0] = 1
    corrected_loss_mask = shifted_loss_mask + one_hot_first
    reversed_loss_mask = torch.flip(corrected_loss_mask, [1])
    
    
    diff_contrast = audio_embed - text_embed
    contrastive_squared_diff = torch.sum(diff_contrast * diff_contrast, dim=-1)
    contrastive_loss = torch.sqrt(contrastive_squared_diff)
    
    contrastive_loss = (contrastive_loss * reversed_loss_mask).sum() / (reversed_loss_mask.sum() + 1e-6)
    
    loss = (kl_loss * args.w1 + contrastive_loss * args.w2) / args.batch_size
    loss.backward()
    
    return {
        "l2_loss": contrastive_loss.item(),
        "kl_loss": kl_loss.item(),
        "total_loss": loss.item() * args.batch_size
    }


def data_collator(features):
    input_ids = torch.stack([torch.tensor(f['input_ids']) for f in features])
    attention_mask = torch.stack([torch.tensor(f['attention_mask']) for f in features])
    audio = [feat['audio']['array'] for feat in features]
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'audio': audio}

def print_trainable_parameters(model):
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.shape))
        else:
            frozen_params.append((name, param.shape))
    
    print("Trainable parameters:")
    for name, shape in trainable_params:
        print(f"{name}: {shape}")
    
    # print("\nFrozen parameters:")
    # for name, shape in frozen_params:
    #     print(f"{name}: {shape}")
    
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_count = sum(p.numel() for p in model.parameters())
    
    print(f"\nTotal trainable parameters: {trainable_params_count:,}")
    print(f"Total parameters: {total_params_count:,}")

if __name__ == "__main__":
    parser = HfArgumentParser((TrainArguments))
    model_args, = parser.parse_args_into_dataclasses()
    
    if not os.path.isdir(model_args.save_dir):
        os.mkdir(model_args.save_dir)
        
    wandb.login()
    run = wandb.init(
        project="CantoDiVA",
        config={**model_args.__dict__},
    )
    model = DiVAModel(
        whisper_path="alvanlii/whisper-small-cantonese", llm_path="hon9kon9ize/CantoneseLLMChat-v1.0-7B",
        is_train=True, speech_encoder_device="cuda:1"
    )
    
    model = torch.compile(model)
    print_trainable_parameters(model)
    if model_args.start_step > 0:
        model.load_prev_checkpoint(f"{model_args.save_dir}/step_{model_args.start_step}")

    def tokenize_function(examples):
        return model.tokenizer(examples['text'], truncation=True, max_length=1024, return_tensors="pt", padding='max_length')

    dataset = load_dataset('alvanlii/audio-llm-train', split='train', streaming=True) # load_from_disk('./data/combined_english_canto')
    LEN_DATASET = model_args.total_rows
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=model_args.microbatch_size,
        collate_fn=data_collator
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=model_args.lr,
        betas=(0.9, 0.98),
        eps=1e-09,
        weight_decay=0.1,
        foreach=True
    )

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_training_steps=model_args.num_epochs * LEN_DATASET // model_args.microbatch_size,
        num_warmup_steps=model_args.warmup_steps
    )

    accum_samples = 0
    bs = model_args.batch_size
    curr_steps = 0
    scaler = amp.GradScaler()
    model = model.train()

    all_l2_losses = []
    all_kl_losses = []
    for epoch in range(model_args.num_epochs):
        for batch in tqdm(train_dataloader, total=model_args.total_rows // model_args.microbatch_size):
            curr_steps += 1
            if model_args.start_step > 0 and curr_steps < model_args.start_step:
                scheduler.step()
                continue
                
            accum_samples += batch['input_ids'].size(0)
            
            metrics = train_step(model, batch, model_args)
            all_l2_losses.append(metrics["l2_loss"])
            all_kl_losses.append(metrics["kl_loss"])
            
            scheduler.step()
            
            if accum_samples >= model_args.batch_size:
                optimizer.step()
                optimizer.zero_grad()
                accum_samples = 0
                
            if curr_steps % model_args.print_every == 0:
                log_metrics(curr_steps, all_l2_losses, all_kl_losses, scheduler, model_args.batch_size)
            
            if len(all_l2_losses) > 1000:
                all_l2_losses = all_l2_losses[-1000:]
                all_kl_losses = all_kl_losses[-1000:]
            
            if curr_steps % model_args.save_every == 0:
                model.save(f"{model_args.save_dir}/step_{curr_steps}")