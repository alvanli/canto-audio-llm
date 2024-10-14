import copy
import json
import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load
from torch import nn
from .config import DiVAConfig
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel
)
from .modeling_qwen2 import Qwen2ForCausalLM
from .modeling_whisper import WhisperForConditionalGeneration

WHISPER_EMBED_DIM = 768
class QFormer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.decoder = None
        self.projection = nn.Linear(WHISPER_EMBED_DIM, 3584)
        self.query_tokens = nn.Parameter(torch.randn(448, WHISPER_EMBED_DIM))

    def forward(self, x, output_device="cuda:0"):
        bsz = x.shape[0]
        query_tokens = self.query_tokens[None, :, :].expand(bsz, -1, -1)
        virt_whisper_tokens = self.decoder(
            inputs_embeds=query_tokens, encoder_hidden_states=x
        )
        virtual_tokens = self.projection(virt_whisper_tokens[0])
        # virtual_tokens = torch.tanh(virtual_tokens)
        return virtual_tokens.to(output_device)


class DiVAModel(PreTrainedModel):
    config_class = DiVAConfig

    def __init__(
        self, via_path=None, config_dict={}, device_map=None, speech_encoder_device="cuda:0",
        whisper_path="alvanlii/whisper-small-cantonese", llm_path="hon9kon9ize/CantoneseLLMChat-v1.0-7B",
        is_train=False
    ):
        super().__init__(DiVAConfig.from_dict(config_dict))

        whisper = WhisperForConditionalGeneration.from_pretrained(
            whisper_path
        )
        connector = QFormer()
        connector.decoder = copy.deepcopy(whisper.model.decoder)
        if via_path is not None:
            with open(via_path, "rb") as f:
                sd = load(f.read())

            with torch.no_grad():
                connector.query_tokens = nn.Parameter(sd["query_tokens"])
                connector.projection.weight = nn.Parameter(sd["projection.weight"].T)
                connector.projection.bias = nn.Parameter(sd["projection.bias"])
                wsd = {
                    key.replace("connector.", ""): sd[key]
                    for key in sd
                    if key.startswith("connector.")
                }
                connector.decoder.load_state_dict(wsd)

        if device_map == None:
            num_layers = 28
            split_index = 24
            device_map = dict(
                **{"model.embed_tokens": 0, "model.norm": 0, "lm_head": 0},
                **{
                    f"model.layers.{i}": 0 if i < split_index else 0
                    for i in range(num_layers)
                },
            )

        self.qformer = connector.to(speech_encoder_device)
        self.whisper_encoder = whisper.model.encoder.to(speech_encoder_device)
        self.whisper_encoder.training = True
    
        self.llm_decoder = Qwen2ForCausalLM.from_pretrained(
            llm_path,
            device_map=device_map,
            # torch_dtype=torch.float16,
        )
        self.llm_decoder.training = True
        
        if is_train:
            for param in self.llm_decoder.parameters():
                param.requires_grad = False
        self.processor = AutoProcessor.from_pretrained(whisper_path)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        
        embed_device = self.llm_decoder.model.embed_tokens.weight.device
        self.prefix = torch.tensor(
            self.tokenizer.encode(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            )
        ).to(embed_device)
        self.pre_user_suffix = torch.tensor(
            self.tokenizer.encode(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            )
        ).to(embed_device)
        self.final_header = torch.tensor(            
            self.tokenizer.encode(
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ).to(embed_device)
        self.speech_encoder_device = speech_encoder_device
        pad_token = self.tokenizer.encode("<|endoftext|>")
        self.pad_token_embed = self.llm_decoder.model.embed_tokens(
            torch.tensor([pad_token]).to(self.llm_decoder.device)
        ) 
        torch.nn.utils.clip_grad_norm_(self.whisper_encoder.parameters(), max_norm=2.0)
        torch.nn.utils.clip_grad_norm_(self.qformer.parameters(), max_norm=2.0)

    def can_generate(cls):
        return False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config=None,
        cache_dir=None,
        **kwargs,
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            via_path = (
                pretrained_model_name_or_path + "/model-00001-of-00004.safetensors"
            )
            config_path = pretrained_model_name_or_path + "/config.json"
        else:
            # Loading from huggingface repo
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="model-00001-of-00004.safetensors",
                token=kwargs.get("token", None),
                local_dir=os.path.dirname(__file__),
            )
            hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="config.json",
                token=kwargs.get("token", None),
                local_dir=os.path.dirname(__file__),
            )
            via_path = os.path.dirname(__file__) + "/model-00001-of-00004.safetensors"
            config_path = os.path.dirname(__file__) + "/config.json"
        with open(config_path, "r") as f:
            config_dict = json.loads(f.read())
        return cls(
            via_path,
            config_dict,
            kwargs["device_map"] if "device_map" in kwargs else "auto",
            (
                kwargs["speech_encoder_device"]
                if "speech_encoder_device" in kwargs
                else None
            ),
        )

    def train_forward(self, audio, input_ids, attention_mask):
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16_000)
        input_features = inputs.input_features.to(self.speech_encoder_device)
        whisper_out = self.whisper_encoder(input_features=input_features)
        hidden_states = whisper_out["last_hidden_state"]
        if torch.isnan(hidden_states).any():
            breakpoint()
        audio_embed = self.qformer(
            hidden_states,
            output_device=self.llm_decoder.model.embed_tokens.weight.device,
        )

        decoder_device = self.llm_decoder.model.embed_tokens.weight.device
        input_ids = input_ids.to(decoder_device)
        attention_mask = attention_mask.to(decoder_device)

        # Generate text embeddings
        text_embed = self.llm_decoder.model.embed_tokens(input_ids)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                text_response = self.llm_decoder(
                    inputs_embeds=text_embed,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True
                )
                # if text_embed.size(0) == 1:
                #     audio_embed = audio_embed.unsqueeze(0)
                max_seq_len = 1024
                curr_padded_audio_embed = self.pad_token_embed.expand(
                    audio_embed.size(0), max_seq_len, audio_embed.size(2)
                ).clone()
                curr_padded_audio_embed[:, :audio_embed.size(1), :] = audio_embed

                # padded_audio_embed = torch.zeros(
                #     audio_embed.size(0), max_seq_len, audio_embed.size(2),
                #     device=decoder_device, dtype=audio_embed.dtype
                # )
                # padded_audio_embed[:, :audio_embed.size(1), :] = audio_embed

                attention_mask_aud = torch.zeros(
                    audio_embed.size(0), max_seq_len,
                    device=decoder_device, dtype=torch.long
                )
                attention_mask_aud[:, :audio_embed.size(1)] = 1
                
                audio_response = self.llm_decoder(
                    inputs_embeds=curr_padded_audio_embed,
                    attention_mask=attention_mask_aud,
                    return_dict=True,
                    output_hidden_states=True
                )
        return audio_embed, text_embed, audio_response.hidden_states[-1], text_response.hidden_states[-1]

    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(self.whisper_encoder.state_dict(), f'{path}/whisper_encoder.pth')
        torch.save(self.qformer.state_dict(), f'{path}/qformer.pth')
        
    def forward(self, audio, prefix_text_tokens, suffix_text_tokens):
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16_000)
        input_features = inputs.input_features.to(self.speech_encoder_device)
        hidden_states = self.whisper_encoder(input_features=input_features)[
            "last_hidden_state"
        ]
        virt_tokens = self.qformer(
            hidden_states,
            output_device=self.llm_decoder.model.embed_tokens.weight.device,
        ).squeeze()

        prefix_embed = self.llm_decoder.model.embed_tokens(prefix_text_tokens)
        suffix_embed = self.llm_decoder.model.embed_tokens(suffix_text_tokens)
        inputs_embeds = torch.cat(
            [prefix_embed, virt_tokens, suffix_embed], axis=0
        ).unsqueeze(0)

        outputs = self.llm_decoder(
            inputs_embeds=inputs_embeds.to(
                self.llm_decoder.model.embed_tokens.weight.device
            ).half(),
            return_dict=True,
            output_hidden_states=True
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        audio,
        text_prompt=None,
        do_sample=False,
        logits_processor=None,
        max_new_tokens=128,
    ):
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16_000)
        input_features = inputs.input_features.to(self.speech_encoder_device)
        hidden_states = self.whisper_encoder(input_features=input_features)[
            "last_hidden_state"
        ]
        virt_tokens = self.qformer(
            hidden_states,
            output_device=self.llm_decoder.model.embed_tokens.weight.device,
        )
        bsz = virt_tokens.shape[0]

        if text_prompt != None and text_prompt != "":
            user_prompt_text = torch.tensor(
                self.tokenizer(
                    text_prompt,
                    add_special_tokens=False,
                    padding=True,
                    padding_side="right",
                )["input_ids"],
                device=self.pre_user_suffix.device,
            )
            prefix = torch.cat(
                [
                    self.pre_user_suffix.expand(
                        bsz,
                        -1,
                    ),
                    user_prompt_text,
                    self.prefix.expand(
                        bsz,
                        -1,
                    ),
                ],
                axis=1,
            )
        else:
            prefix = self.prefix
        prefix_embed = self.llm_decoder.model.embed_tokens(prefix).expand(bsz, -1, -1)
        suffix = self.final_header
        suffix_embed = self.llm_decoder.model.embed_tokens(suffix).expand(bsz, -1, -1)
        inputs_embeds = torch.cat([prefix_embed, virt_tokens, suffix_embed], axis=1)
        outs = [[] for i in range(bsz)]
        complete = [False] * bsz
        outputs = None
        greedy = 1
        i = 0
        while not all(complete) and len(outs[0]) < max_new_tokens:
            past_key_values = outputs.past_key_values if outputs else None
            outputs = self.llm_decoder(
                inputs_embeds=inputs_embeds.to(
                    self.llm_decoder.model.embed_tokens.weight.device
                ).half(),
                return_dict=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            next_token_logits = outputs.logits[:, -1, :]

            if logits_processor:
                local_outs = torch.tensor(outs) if outs != [] else suffix
                local_outs = local_outs.reshape(1, -1)
                next_token_logits = logits_processor(
                    local_outs,
                    next_token_logits.reshape(1, -1),
                )
                next_token_logits = next_token_logits.flatten()
            if do_sample:
                logits = next_token_logits / self.temperature
                probs = F.softmax(logits, dim=-1)
                greedy = torch.multinomial(probs, num_samples=1)[0]
            else:
                greedy = next_token_logits.argmax(dim=-1)
            for token_index, out in enumerate(greedy.flatten().tolist()):
                if not complete[token_index]:
                    outs[token_index].append(out)
                if out == 128009:
                    complete[token_index] = True

            next_embed = self.llm_decoder.model.embed_tokens(greedy.reshape(-1, 1))
            inputs_embeds = next_embed
        return self.tokenizer.batch_decode(outs, skip_special_tokens=True)

    def generate_stream(
        self,
        audio,
        text_prompt,
        do_sample=False,
        logits_processor=None,
        max_new_tokens=128,
    ):
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16_000)
        input_features = inputs.input_features.to(self.whisper_encoder.device)
        hidden_states = self.whisper_encoder(input_features=input_features)[
            "last_hidden_state"
        ]
        virt_tokens = self.qformer(
            hidden_states,
            output_device=self.llm_decoder.model.embed_tokens.weight.device,
        ).squeeze()

        if text_prompt != None and text_prompt != "":
            user_prompt_text = torch.tensor(
                self.tokenizer(text_prompt, add_special_tokens=False)["input_ids"],
                device=self.pre_user_suffix.device,
            )
            prefix = torch.cat(
                [self.pre_user_suffix, user_prompt_text, self.prefix], axis=0
            )
        else:
            prefix = self.prefix
        prefix_embed = self.llm_decoder.model.embed_tokens(prefix)
        suffix = self.final_header
        suffix_embed = self.llm_decoder.model.embed_tokens(suffix)
        inputs_embeds = torch.cat(
            [prefix_embed, virt_tokens, suffix_embed], axis=0
        ).unsqueeze(0)
        outs = []
        outputs = None
        greedy = 1
        i = 0
        while greedy != 128009 and len(outs) < max_new_tokens:
            past_key_values = outputs.past_key_values if outputs else None
            outputs = self.llm_decoder(
                inputs_embeds=inputs_embeds.to(
                    self.llm_decoder.model.embed_tokens.weight.device
                ).half(),
                return_dict=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            next_token_logits = outputs.logits[-1, -1, :]

            if logits_processor:
                local_outs = torch.tensor(outs) if outs != [] else suffix
                local_outs = local_outs.reshape(1, -1)
                next_token_logits = logits_processor(
                    local_outs,
                    next_token_logits.reshape(1, -1),
                )
                next_token_logits = next_token_logits.flatten()
            if do_sample:
                logits = next_token_logits / self.temperature
                probs = F.softmax(logits, dim=-1)
                greedy = torch.multinomial(probs, num_samples=1)[0]
            else:
                greedy = next_token_logits.argmax()
            outs.append(greedy)
            next_embed = self.llm_decoder.model.embed_tokens(greedy.reshape(1, 1))
            inputs_embeds = next_embed
            yield self.tokenizer.decode(outs, skip_special_tokens=True).replace(
                "<|eot_id|>", ""
            )
        return self.tokenizer.decode(outs, skip_special_tokens=True).replace(
            "<|eot_id|>", ""
        )