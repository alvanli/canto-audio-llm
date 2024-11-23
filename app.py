import copy
import os
import random
import sys

import xxhash
import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from accelerate import infer_auto_device_map
from datasets import Audio
from safetensors.torch import load, load_model
import spaces
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    LlamaForCausalLM,
    TextIteratorStreamer,
    WhisperForConditionalGeneration,
    AutoProcessor,
    AutoModel,
)
from transformers.generation import GenerationConfig
from model.model import DiVAModel

anonymous = False

# diva_model = AutoModel.from_pretrained(
#     "WillHeld/DiVA-llama-3-v0-8b", trust_remote_code=True
# )
diva_model = DiVAModel(
    whisper_path="alvanlii/whisper-small-cantonese", llm_path="hon9kon9ize/CantoneseLLMChat-v1.0-7B",
    is_train=False, speech_encoder_device="cuda:1"
)
diva_model.load_prev_checkpoint(f"./logs/v04/step_232800")

resampler = Audio(sampling_rate=16_000)


@spaces.GPU
@torch.no_grad
def diva_audio(audio_input, do_sample=True, temperature=0.001):
    sr, y = audio_input
    x = xxhash.xxh32(bytes(y)).hexdigest()
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    a = resampler.decode_example(
        resampler.encode_example({"array": y, "sampling_rate": sr})
    )
    yield from diva_model.generate_stream(
        a["array"], None, do_sample=do_sample, max_new_tokens=256
    )


def transcribe_wrapper(audio_input, state, model_order):
    spinner = "◒"
    d_resp = gr.Textbox(
        value="Loading",
        visible=True,
        label=model_names[0] if not anonymous else f"Model {order}",
    )
    yield (
        gr.Button(
            value="Loading Weights onto ZeroGPU...",
            interactive=False,
            variant="primary",
        ),
        d_resp,
        state,
    )

    yield from transcribe(audio_input, state, model_order)


@spaces.GPU
def transcribe(audio_input, state, model_order):
    if audio_input == None:
        return (
            "Click to run inference!",
            "",
            state,
        )

    def gen_from_diva():
        diva_resp = diva_audio(audio_input)
        for resp in diva_resp:
            d_resp = gr.Textbox(
                value=resp,
                visible=True,
                label=model_names[0] if not anonymous else f"Model {order}",
            )
            yield d_resp

    spinner_id = 0
    spinners = ["◐ ", "◓ ", "◑", "◒"]

    for response in gen_from_diva():
        spinner = spinners[spinner_id]
        spinner_id = (spinner_id + 1) % 4
        yield (
            gr.Button(
                value=spinner + " Generating Responses " + spinner,
                interactive=False,
                variant="primary",
            ),
            response,
            state,
        )
    yield (
        gr.Button(value="Click to run inference!", interactive=True, variant="primary"),
        response,
        state,
    )


def on_page_load(state, model_order):
    if state == 0:
        gr.Info(
            "Record something you'd say to an AI Assistant! Think about what you usually use Siri, Google Assistant, or ChatGPT for."
        )
        state = 1
        if anonymous:
            random.shuffle(model_order)
    return state, model_order


def recording_complete(state):
    if state == 1:
        gr.Info(
            "Once you submit your recording, DiVA will stream back a response! This might take a second as ZeroGPU needs to load model weights into vRAM!."
        )
        state = 2
    return (
        gr.Button(value="Click to run inference!", interactive=True, variant="primary"),
        state,
    )


def clear_factory(button_id):
    def clear(audio_input, model_order):
        return (
            model_order,
            gr.Button(
                value="Record Audio to Submit!",
                interactive=False,
            ),
            None,
            None,
        )

    return clear


theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c100="#82000019",
        c200="#82000033",
        c300="#8200004c",
        c400="#82000066",
        c50="#8200007f",
        c500="#8200007f",
        c600="#82000099",
        c700="#820000b2",
        c800="#820000cc",
        c900="#820000e5",
        c950="#820000f2",
    ),
    secondary_hue="rose",
    neutral_hue="stone",
)

model_names = ["DiVA Llama 3 8B"]
model_shorthand = ["diva"]
with gr.Blocks(theme=theme) as demo:
    state = gr.State(0)
    model_order = gr.State([0, 1])
    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"], streaming=False, label="Audio Input"
        )

    with gr.Row():
        btn = gr.Button(value="Record Audio to Submit!", interactive=False)

    with gr.Row():
        out1 = gr.Textbox(visible=False)

    audio_input.stop_recording(
        recording_complete,
        [state],
        [btn, state],
    )
    audio_input.start_recording(
        lambda: gr.Button(
            value="Uploading Audio to Cloud", interactive=False, variant="primary"
        ),
        None,
        btn,
    )
    btn.click(
        fn=transcribe_wrapper,
        inputs=[audio_input, state, model_order],
        outputs=[btn, out1, state],
    )
    audio_input.clear(
        clear_factory(None),
        [audio_input, model_order],
        [model_order, btn, audio_input, out1],
    )
    demo.load(
        fn=on_page_load, inputs=[state, model_order], outputs=[state, model_order]
    )

demo.launch(share=True)
