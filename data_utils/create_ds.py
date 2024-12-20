import glob, json, os
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets, Audio

from canto_utils.utils import remove_emotion_and_event_tokens, convert_to_traditional, compute_cer

DOWNLOAD_PATH = "/data/download_2"

if __name__ == "__main__":
    # data = defaultdict(list)
    # all_files = sorted(glob.glob(f"{DOWNLOAD_PATH}/mp3_perturbed/*"))
    # all_files = all_files[:len(all_files)//2]

    # for audio_file in tqdm(all_files):
    #     file_id = os.path.basename(audio_file).split(".")[0]
        
    #     metadata_path = f"{DOWNLOAD_PATH}/metadata/{file_id}.json"
    #     fixed_transcript_path = f"{DOWNLOAD_PATH}/bert_fixed/{file_id}.txt"
    #     sensevoice_path = f"{DOWNLOAD_PATH}/sensevoice/{file_id}.txt"

    #     if not os.path.exists(metadata_path) or not os.path.exists(sensevoice_path) or not os.path.exists(fixed_transcript_path):
    #         continue

    #     with open(metadata_path, "r") as f:
    #         metadata = json.loads(f.read())
            
    #     try:
    #         a, b, c = metadata['channel'], metadata['labels_1'], metadata['title']
    #     except:
    #         continue

    #     with open(fixed_transcript_path, "r") as f:
    #         bert_fixed = f.read()
        
    #     with open(sensevoice_path, "r") as f:
    #         sensevoice = remove_emotion_and_event_tokens(f.read())
    #     filtered_sv = convert_to_traditional(sensevoice)

    #     if len(filtered_sv.strip()) < 4 or len(metadata['labels_1'].strip()) < 4:
    #         continue

    #     if compute_cer(filtered_sv, metadata['labels_1']) > 0.4:
    #         continue

    #     data['id'].append(file_id)
    #     data['channel'].append(metadata['channel'])
    #     data['transcript_whisper'].append(metadata['labels_1'])
    #     data['title'].append(metadata['title'])
    #     data['transcript_fixed'].append(bert_fixed)
    #     data['audio'].append(audio_file.replace("mp3_perturbed", "mp3_direct"))
    #     data['audio_perturbed'].append(audio_file)
    #     data['transcript_sensevoice'].append(filtered_sv)
            
    # ds = Dataset.from_dict(mapping=data)
    # ds.save_to_disk("./data/audio-2nd")
    
    # del ds
    # import gc
    # gc.collect()

    d1 = load_from_disk("./data/audio-2nd")

    d1 = d1.rename_column("transcript_fixed", "text")
    d1 = d1.remove_columns(column_names=list(set(d1.features.keys()) - {'audio_perturbed', 'text'}))
    d1 = d1.rename_column("audio_perturbed", "audio")
    print(d1.features.keys())

    cv_en_ds = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="train", trust_remote_code=True)
    cv_en_ds = cv_en_ds.rename_column("sentence", "text")
    cv_en_ds = cv_en_ds.remove_columns(column_names=list(set(cv_en_ds.features.keys()) - {'audio', 'text'}))
    cv_en_ds = cv_en_ds.cast_column("audio", Audio(sampling_rate=16_000)) 
    d1 = d1.cast_column("audio", Audio(sampling_rate=16_000)) 
    print(cv_en_ds.features.keys())
    
    combined_dataset = concatenate_datasets([d1, cv_en_ds])
    print(cv_en_ds[0]['text'])
    print(cv_en_ds[10]['text'])
    print(len(d1), len(cv_en_ds), len(combined_dataset))
    combined_dataset = combined_dataset.shuffle()
    combined_dataset = combined_dataset.filter(filter_audio_length)
    combined_dataset.save_to_disk("./data/combined_english_canto")

    # del combined_dataset
    # import gc
    # gc.collect()

    # combined_dataset = load_from_disk("./data/combined_english_canto")
    combined_dataset.push_to_hub("alvanlii/audio-2nd")