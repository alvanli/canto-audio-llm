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


def get_last_N(embeddings, N=448):
    batch_size, seq_len, dim = embeddings.size()

    valid_embeddings_list = []

    for i in range(batch_size):
        mask = attention_mask[i]  # Shape: (seq_len,)
        emb = embeddings[i]       # Shape: (seq_len, dim)
        
        # Step 1: Find indices where attention_mask != 0
        valid_indices = torch.nonzero(mask, as_tuple=True)[0]  # Shape: (num_valid_tokens,)
        
        # Step 2: Select the last N indices
        last_N_indices = valid_indices[-N:]  # If less than N, it'll take all available
        num_selected = last_N_indices.size(0)
        
        # Step 3: Extract embeddings at these indices
        selected_embeddings = emb[last_N_indices]  # Shape: (num_selected, dim)
        
        # Optional: Pad sequences to length N if necessary
        if num_selected < N:
            pad_length = N - num_selected
            padding = torch.zeros(pad_length, dim, device=emb.device)
            selected_embeddings = torch.cat((padding, selected_embeddings), dim=0)  # Shape: (N, dim)
        
        valid_embeddings_list.append(selected_embeddings)

    # Stack all embeddings to get a tensor of shape (batch_size, N, dim)
    valid_embeddings = torch.stack(valid_embeddings_list, dim=0)
    return valid_embeddings


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
    # model = DiVAModel(
    #     whisper_path="alvanlii/whisper-small-cantonese", llm_path="hon9kon9ize/CantoneseLLMChat-v1.0-7B",
    #     is_train=True, speech_encoder_device="cuda:1"
    # )
    model = DiVAModel(
        whisper_path="./logs/smaller-turbo", llm_path="hon9kon9ize/CantoneseLLMChat-v1.0-7B",
        is_train=True, speech_encoder_device="cuda:1"
    )
    model = torch.compile(model)

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
    # model = model.gradient_checkpointing_enable()

    for epoch in range(model_args.num_epochs):
        all_l2_losses = []
        all_kl_losses = []
        for i, batch in tqdm(enumerate(train_dataloader), total=LEN_DATASET // model_args.microbatch_size):
            curr_steps += 1
            if model_args.start_step > 0 and curr_steps < model_args.start_step:
                scheduler.step()
                continue
            accum_samples += model_args.microbatch_size

            audio, input_ids, attention_mask = batch['audio'], batch['input_ids'], batch['attention_mask']
            audio_embed, text_embed, audio_response, text_response, attention_mask_aud = model.train_forward(audio, input_ids, attention_mask)

            # Compute KL Proxy Loss
            diff_distill = audio_response[:,-1] - text_response[:,-1] # last_audio_logits - last_text_logits
            kl_squared_diff = diff_distill.pow(2).sum(dim=-1)
            kl_loss = kl_squared_diff.sqrt().mean()
            
            # Prepare Mask for Contrastive Loss
            shifted_loss_mask = torch.roll(attention_mask, shifts=1, dims=1)
            one_hot_first = torch.zeros_like(attention_mask)
            one_hot_first[:, 0] = 1
            corrected_loss_mask = shifted_loss_mask + one_hot_first
            reversed_loss_mask = torch.flip(corrected_loss_mask, dims=[1])
            reversed_loss_mask = (reversed_loss_mask > 0).float()

            # Compute Contrastive Loss
            # Ensure text_embed has a consistent max sequence length
            max_seq_len = text_embed.size(1)

            # Pad audio_embed to match text_embed's sequence length
            audio_embed_padded = F.pad(
                audio_embed, 
                (0, 0, 0, max_seq_len - audio_embed.size(1)),  # Pad on the sequence dimension to the right
                value=0
            )

            diff_contrast = audio_embed_padded - text_embed
            contrastive_squared_diff = diff_contrast.pow(2).sum(dim=-1)
            contrastive_loss = contrastive_squared_diff.sqrt()
            reversed_loss_mask = reversed_loss_mask.to(contrastive_loss.device)
            masked_contrastive_loss = contrastive_loss * reversed_loss_mask
            l2_loss = masked_contrastive_loss.sum() / reversed_loss_mask.sum()
            
            if (torch.isnan(kl_loss)):
                breakpoint()

            loss_sum = model_args.w1 * l2_loss + model_args.w2 * kl_loss

            scaler.scale(loss_sum / model_args.batch_size).backward()
            
            scheduler.step()
            if accum_samples >= bs:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_samples = 0
                  
            all_l2_losses.append(l2_loss.detach().item())
            all_kl_losses.append(kl_loss.detach().item())
            
            if curr_steps % model_args.print_every == 0:
                log_metrics(curr_steps, all_l2_losses, all_kl_losses, scheduler, bs)
                
            if curr_steps % model_args.save_every == 0:
                model.save(f"{model_args.save_dir}/step_{curr_steps}")
                
            if len(all_l2_losses) > 1000:
                all_kl_losses = all_kl_losses[-1000:]
                all_l2_losses = all_l2_losses[-1000:]