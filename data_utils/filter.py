import regex, time
import pandas as pd
from jiwer import cer
from datasets import load_from_disk, load_dataset, concatenate_datasets, Audio

def remove_punctuations(text):
    return regex.sub(r'\p{P}+', '', text)


def compute_cer(batch):
    reference = remove_punctuations(batch['transcript_whisper'])
    hypothesis = remove_punctuations(batch['transcript_sensevoice'])
    cer_value = cer(reference, hypothesis)
    batch['cer'] = cer_value
    return batch

def filter_audio_length(example):
    return len(example['audio']['array']) < 30 * 16_000

if __name__ == "__main__":
    # ds = load_from_disk("/data/canto_labels")

    # dataset_with_cer = ds.map(compute_cer, num_proc=4)
    # filtered_dataset = dataset_with_cer.filter(lambda example: example['cer'] < 0.3 and '字幕' not in example['transcript_whisper'])
    # print(filtered_dataset[0])
    # print(len(filtered_dataset))
    # filtered_dataset.save_to_disk("./data/filtered_ds")
    # del filtered_dataset
    # import gc
    # gc.collect()

    # d1 = load_from_disk("./data/filtered_ds")

    # d1 = d1.rename_column("transcript_whisper", "text")
    # d1 = d1.remove_columns(column_names=list(set(d1.features.keys()) - {'audio', 'text'}))
    # print(d1.features.keys())

    # d2 = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="train")
    # d2 = d2.rename_column("sentence", "text")
    # d2 = d2.remove_columns(column_names=list(set(d2.features.keys()) - {'audio', 'text'}))
    # d2 = d2.cast_column("audio", Audio(sampling_rate=16_000)) 
    # d1 = d1.cast_column("audio", Audio(sampling_rate=16_000)) 
    # print(d2.features.keys())
    
    # combined_dataset = concatenate_datasets([d1, d2])
    # print(d2[0]['text'])
    # print(d2[10]['text'])
    # print(len(d1), len(d2), len(combined_dataset))
    # combined_dataset.save_to_disk("./data/combined_english_canto")
    # combined_dataset = load_from_disk("./data/combined_english_canto")
    # combined_dataset = combined_dataset.shuffle()
    # print(len(combined_dataset))
    # combined_dataset = combined_dataset.filter(filter_audio_length)
    # combined_dataset.save_to_disk("./data/combined_english_canto_filtered")
    # print(len(combined_dataset))
    combined_dataset = load_from_disk("./data/combined_english_canto_filtered")
    combined_dataset.push_to_hub("alvanlii/audio-llm-train-shuffled")

    # ds = load_from_disk("./data/combined_english_canto")
    # ds.push_to_hub("alvanlii/audio-llm-train")