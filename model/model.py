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
from .modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM
from .modeling_whisper import WhisperForConditionalGeneration

WHISPER_EMBED_DIM = 768
QWEN_DIM = 3584
WHISPER_MAX_LENGTH = 448

class QFormer(nn.Module):
    def __init__(
        self,
        whisper_embed_dim, qwen_dim, whisper_max_length
    ):
        super().__init__()
        self.decoder = None
        self.projection = nn.Linear(whisper_embed_dim, qwen_dim)
        self.query_tokens = nn.Parameter(torch.randn(whisper_max_length, whisper_embed_dim))

    def forward(self, x, output_device="cuda:1"):
        bsz = x.shape[0]
        query_tokens = self.query_tokens[None, :, :].expand(bsz, -1, -1)
        virt_whisper_tokens = self.decoder(
            inputs_embeds=query_tokens, encoder_hidden_states=x
        )
        if self.projection.weight.shape[-1] == 5120:
            virtual_tokens = self.projection(virt_whisper_tokens[0].reshape(112, 5120))
        else:
            virtual_tokens = self.projection(virt_whisper_tokens[0])
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
            whisper_path, attn_implementation="sdpa"
        )
        connector = QFormer(
            whisper_embed_dim=whisper.config.d_model,
            qwen_dim=3584, whisper_max_length=whisper.config.max_target_positions)
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

        if is_train:
            num_layers = 28
            split_index = 15
            device_map = dict(
                **{"embed_tokens": 0, "norm": 0},
                **{
                    f"layers.{i}": 0 if i < split_index else 1
                    for i in range(num_layers)
                },
            )

        self.qformer = connector.to(speech_encoder_device)
        self.whisper_encoder = whisper.model.encoder.to(speech_encoder_device)
        self.whisper_encoder.training = True

        if is_train:
            self.llm_decoder = Qwen2Model.from_pretrained(
                llm_path,
                device_map=device_map,
                attn_implementation="sdpa"
            )
            self.llm_decoder.requires_grad_(False)
            self.llm_decoder_device = self.llm_decoder.embed_tokens.weight.device
            for param in self.whisper_encoder.parameters():
                param.requires_grad = False
        else:
            self.llm_decoder = Qwen2ForCausalLM.from_pretrained(
                llm_path,
                device_map=device_map,
                attn_implementation="sdpa"
            )
            self.llm_decoder_device = self.llm_decoder.model.embed_tokens.weight.device

        
        self.processor = AutoProcessor.from_pretrained(whisper_path)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        prefix, suffix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "PLACEHOLDER"}],
            tokenize=False,
            add_generation_prompt=True,
        ).split("PLACEHOLDER")
        non_null = [line for line in prefix.split("\n") if line.strip()]
        prefix_tok = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tok = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.prefix = torch.tensor(prefix_tok).to(
            self.llm_decoder_device
        )

        self.pre_system = torch.tensor(
            self.tokenizer.encode(non_null[0] + "\n", add_special_tokens=False)
        ).to(self.llm_decoder_device)
        self.post_system = torch.tensor(
            self.tokenizer.encode("\n" + non_null[-1] + "\n", add_special_tokens=False)
        ).to(self.llm_decoder_device)
        self.final_header = torch.tensor(suffix_tok).to(
            self.llm_decoder_device
        )
        self.speech_encoder_device = speech_encoder_device
        self.eos_token = self.tokenizer.eos_token
        self.default_prompt = "重複返用家輸入嗰句說話"

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

        decoder_device = self.llm_decoder.embed_tokens.weight.device
        audio_embed = self.qformer(
            hidden_states,
            output_device=decoder_device,
        )

        bsz = audio_embed.shape[0]

        prefix_audio = torch.cat([
            self.pre_system.expand(bsz, -1),
            self.post_system.expand(bsz, -1),
        ], dim=1)
        prefix_audio_embed = self.get_embeddings(prefix_audio)
        suffix_audio = self.final_header.expand(bsz, -1)
        suffix_audio_embed = self.get_embeddings(suffix_audio)
        audio_inputs_embeds = torch.cat([prefix_audio_embed, audio_embed, suffix_audio_embed], dim=1)

        prefix_text = torch.cat([
            self.pre_system.expand(bsz, -1),
            self.post_system.expand(bsz, -1),
        ], dim=1).to(decoder_device)
        
        input_ids = input_ids.to(decoder_device)
        suffix_text = self.final_header.expand(bsz, -1).to(decoder_device)

        text_tokens = torch.cat([
            prefix_text, input_ids, suffix_text
        ], dim=1)
        
        text_attention = torch.cat([
            torch.ones_like(prefix_text, device=decoder_device),
            attention_mask.to(decoder_device),
            torch.ones_like(suffix_text, device=decoder_device)
        ], dim=1)

        push_back_padding = torch.argsort((text_tokens == self.tokenizer.pad_token_id).long(), dim=1)
        text_tokens_left_pad = torch.gather(text_tokens, dim=1, index=push_back_padding)
        text_attention_left_pad = torch.gather(text_attention, dim=1, index=push_back_padding)
        
        if text_tokens_left_pad.size(1) > 448:
            text_tokens_left_pad = text_tokens_left_pad[:, -448:]
            text_attention_left_pad = text_attention_left_pad[:, -448:]
        elif text_tokens_left_pad.size(1) < 448:
            pad_length = 448 - text_tokens_left_pad.size(1)
            text_tokens_left_pad = F.pad(text_tokens_left_pad, (pad_length, 0), value=self.tokenizer.pad_token_id)
            text_attention_left_pad = F.pad(text_attention_left_pad, (pad_length, 0), value=0)
        
        text_embed = self.get_embeddings(text_tokens_left_pad)

        text_response = self.llm_decoder(
            inputs_embeds=text_embed,
            attention_mask=text_attention_left_pad,
            return_dict=True,
            output_hidden_states=True
        )

        audio_response = self.llm_decoder(
            inputs_embeds=audio_inputs_embeds,
            attention_mask=torch.ones((audio_inputs_embeds.size(0), audio_inputs_embeds.size(1)), 
                                    device=audio_inputs_embeds.device),
            return_dict=True,
            output_hidden_states=True
        )

        push_forward_padding = torch.argsort((input_ids == self.tokenizer.pad_token_id).long(), dim=1, descending=True)
        input_ids_right_pad = torch.gather(input_ids, dim=1, index=push_forward_padding)
        
        if input_ids_right_pad.size(1) > 448:
            input_ids_right_pad = input_ids_right_pad[:, :448]
        elif input_ids_right_pad.size(1) < 448:
            pad_length = 448 - input_ids_right_pad.size(1)
            input_ids_right_pad = F.pad(input_ids_right_pad, (0, pad_length), value=self.tokenizer.pad_token_id)
            
        text_embeds_wo_prefix_suffix = self.get_embeddings(input_ids_right_pad)

        return (
            audio_embed,
            text_embeds_wo_prefix_suffix,
            audio_response.last_hidden_state,
            text_response.last_hidden_state,
            text_attention_left_pad
        )
    
    def get_embeddings(self, input_ids):
        if hasattr(self.llm_decoder, 'model'):
            return self.llm_decoder.model.embed_tokens(input_ids)
        return self.llm_decoder.embed_tokens(input_ids)

    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(self.qformer.state_dict(), f'{path}/qformer.pth')
        
    def load_prev_checkpoint(self, path):
        self.qformer.load_state_dict(torch.load(f"{path}/qformer.pth"))

    def forward(self, audio, prefix_text_tokens, suffix_text_tokens):
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16_000)
        input_features = inputs.input_features.to(self.speech_encoder_device)
        hidden_states = self.whisper_encoder(input_features=input_features)[
            "last_hidden_state"
        ]
        virt_tokens = self.qformer(
            hidden_states,
            output_device=self.llm_decoder_device,
        ).squeeze()

        prefix_embed = self.get_embeddings(prefix_text_tokens)
        suffix_embed = self.get_embeddings(suffix_text_tokens)
        inputs_embeds = torch.cat(
            [prefix_embed, virt_tokens, suffix_embed], axis=0
        ).unsqueeze(0)

        outputs = self.llm_decoder(
            inputs_embeds=inputs_embeds.to(
                self.llm_decoder_device
            ).half(),
            return_dict=True,
            output_hidden_states=True
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        audio,
        temperature=0.6,
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
            output_device=self.llm_decoder_device,
        )

        # virt_tokens = self.get_embeddings(
        #     torch.tensor(
        #         self.tokenizer.encode(
        #             "如果我食咗早餐，我就唔會肚餓。我今日冇肚餓，咁我今日食咗早餐未？"
        #         )
        #     ).to(self.llm_decoder_device).unsqueeze(0)
        # )

        bsz = virt_tokens.shape[0]

        if text_prompt is None:
            text_prompt = self.default_prompt
        user_prompt_text = torch.tensor(
            self.tokenizer.encode(text_prompt),
            device=self.pre_system.device,
        ).expand(bsz, -1)

        prefix = torch.cat(
            [
                self.pre_system.expand(bsz, -1),
                user_prompt_text,
                self.post_system.expand(bsz,-1),
            ],
            axis=1,
        )
        prefix_embed = self.get_embeddings(prefix).expand(bsz, -1, -1)
        suffix = self.final_header
        suffix_embed = self.get_embeddings(suffix).expand(bsz, -1, -1)
        inputs_embeds = torch.cat([prefix_embed, virt_tokens, suffix_embed], axis=1)


        # bsz=1
        # suffix = self.final_header
        # conversation = []
        # conversation.append({"role": "system", "content": "你係由 hon9kon9ize 開發嘅 CantoneseLLM，你係一個好幫得手嘅助理" })
        # conversation.append({"role": "user", "content": "如果我食咗早餐，我就唔會肚餓。我今日冇肚餓，咁我今日食咗早餐未"})
        # input_ids = self.tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        # inputs_embeds = self.get_embeddings(input_ids).expand(bsz, -1,-1)


        inputs_embeds = inputs_embeds.to(
            self.llm_decoder_device
        ).half()

        outs = [[] for i in range(bsz)]
        complete = [False] * bsz
        outputs = None
        greedy = 1
        i = 0
        past_key_values = None
        while not all(complete) and len(outs[0]) < max_new_tokens:
            outputs = self.llm_decoder(
                inputs_embeds=inputs_embeds,
                return_dict=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                use_cache=True
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
                logits = next_token_logits / temperature
                probs = F.softmax(logits, dim=-1)
                greedy = torch.multinomial(probs, num_samples=1)[0]
            else:
                greedy = next_token_logits.argmax(dim=-1)
            for token_index, out in enumerate(greedy.flatten().tolist()):
                if not complete[token_index]:
                    outs[token_index].append(out)
                if out == self.tokenizer.eos_token_id:
                    complete[token_index] = True

            next_embed = self.get_embeddings(greedy.reshape(-1, 1))
            inputs_embeds = next_embed
            past_key_values = outputs['past_key_values']

        return self.tokenizer.batch_decode(outs, skip_special_tokens=True)
