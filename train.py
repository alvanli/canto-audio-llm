from dataclasses import dataclass, field

import os
import wandb
import numpy as np
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import torch.nn.functional as F

from transformers import HfArgumentParser
from datasets import load_from_disk, load_dataset
from model.model import DiVAModel


def log_metrics(curr_steps, all_l2_losses, all_kl_losses, optimizer, bs):
    if len(all_l2_losses) < bs:
        l2_loss = np.mean(all_l2_losses)
        kl_loss = np.mean(all_kl_losses)
    else:
        l2_loss = np.mean(all_l2_losses[-bs:])
        kl_loss = np.mean(all_kl_losses[-bs:])

    lr = optimizer.param_groups[0].get("lr", float("NaN"))
    log_dict = {
        "step": curr_steps,
        "l2_loss": l2_loss,
        "kl_loss": kl_loss,
        "lr": lr
    }
    wandb.log(log_dict)
    return


@dataclass
class TrainArguments:
    w1: float = field(default=1.0)
    w2: float = field(default=1.0)
    num_epochs: int = field(default=2)
    microbatch_size: int = field(default=12)
    batch_size: int = field(default=50)
    print_every: int = field(default=500)
    lr: float = field(default=1e-5)
    save_every: int = field(default=5_000)
    save_dir: str = field(default="./logs/default")


def l2_loss_fn(embedding_a, embedding_b):
    return torch.mean((embedding_a - embedding_b) ** 2)


def data_collator(features):
    input_ids = torch.stack([torch.tensor(f['input_ids']) for f in features])
    attention_mask = torch.stack([torch.tensor(f['attention_mask']) for f in features])
    audio = [feat['audio']['array'] for feat in features]
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'audio': audio}

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
        whisper_path="Scrya/whisper-large-v2-cantonese", llm_path="hon9kon9ize/CantoneseLLMChat-v1.0-7B",
        is_train=True, speech_encoder_device="cuda:1"
    )

    def tokenize_function(examples):
        return model.tokenizer(examples['text'], truncation=True, max_length=1024, return_tensors="pt", padding='max_length')

    dataset = load_dataset('alvanlii/audio-llm-train', split='train', streaming=True) # load_from_disk('./data/combined_english_canto')
    LEN_DATASET = 2_000_000
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
        weight_decay=0.1
    )

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=model_args.num_epochs * LEN_DATASET // model_args.batch_size,
        eta_min=0
    )

    accum_samples = 0
    current_batch_size = model_args.batch_size
    curr_steps = 0
    scaler = amp.GradScaler()
    for epoch in range(model_args.num_epochs):
        model.train()
        all_l2_losses = []
        all_kl_losses = []
        for i, batch in enumerate(train_dataloader):
            accum_samples += model_args.microbatch_size

            audio, input_ids, attention_mask = batch['audio'], batch['input_ids'], batch['attention_mask']
            optimizer.zero_grad()
            audio_embed, text_embed, audio_response, text_response = model.train_forward(audio, input_ids, attention_mask)
            
            l2_loss = l2_loss_fn(audio_embed, text_embed)
            kl_loss = F.kl_div(
                F.log_softmax(audio_response, dim=-1), 
                F.softmax(text_response, dim=-1), 
                reduction='batchmean'
            )
            loss_sum = model_args.w1 * l2_loss + model_args.w2 * kl_loss
            scaler.scale(loss_sum).backward()
            
            curr_steps += 1
            scheduler.step()
            if accum_samples >= current_batch_size:
                scaler.step(optimizer)
                scaler.update()
                optimizer.step()
                optimizer.zero_grad()
                accum_samples = 0
                                
            all_l2_losses.append(l2_loss.detach().item())
            all_kl_losses.append(kl_loss.detach().item())
            
            if curr_steps % model_args.print_every == 0:
                log_metrics(curr_steps, all_l2_losses, all_kl_losses, optimizer, current_batch_size)
                
            if curr_steps % model_args.save_every == 0:
                model.save(f"{model_args.save_dir}/step_{curr_steps}")
                
            if len(all_l2_losses) > 1000:
                all_kl_losses = all_kl_losses[-1000:]
                all_l2_losses = all_l2_losses[-1000:]