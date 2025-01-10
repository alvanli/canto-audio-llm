from model.model import DiVAModel
from datasets import Audio
import soundfile as sf
import xxhash
import numpy as np
import random, glob


if __name__ == "__main__":
    model = DiVAModel(
        whisper_path="alvanlii/whisper-small-cantonese", llm_path="hon9kon9ize/CantoneseLLMChat-v1.0-7B",
        is_train=False, speech_encoder_device="cuda:1"
    )
    model.load_prev_checkpoint(f"./logs/v10/step_388800")

    mp3_files = glob.glob("/data/download_2/mp3_direct/*.mp3")
    picked_file = random.choice(mp3_files)
    y, sr = sf.read(picked_file)
    resampler = Audio(sampling_rate=16_000)
    x = xxhash.xxh32(bytes(y)).hexdigest()
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    a = resampler.decode_example(
        resampler.encode_example({"array": y, "sampling_rate": sr})
    )['array']

    with open(picked_file.replace("mp3_direct", "bert_fixed").replace("mp3", "txt"), "r") as f:
        og_text = f.read()
    print("original text: ", og_text)

    # model.llm_decoder.use_cache = True
    out = model.generate(
        a,
        text_prompt=None,
        do_sample=False,
        temperature=0.8
    )
    print(out)

    # generator = model.generate_stream(
    #     a,
    #     text_prompt=None,
    #     do_sample=True,
    #     temperature=0.95
    # )

    # for response in generator:
    #     print(response, end='\r', flush=True)
    # print("\nDone!")    