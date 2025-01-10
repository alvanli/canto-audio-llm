import glob, json, os
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets, Audio

from canto_utils.utils import remove_emotion_and_event_tokens, convert_to_traditional, compute_cer

DOWNLOAD_PATH = "/data/download_2"

if __name__ == "__main__":
    all_files = sorted(glob.glob(f"{DOWNLOAD_PATH}/mp3_perturbed/*"))
    all_files = all_files[:len(all_files)//2]

    count_dict = {
        "path_dne": 0,
        "bad_meta": 0,
        "short_txt": 0,
        "bad_cer": 0
    }

    for audio_file in tqdm(all_files):
        file_id = os.path.basename(audio_file).split(".")[0]
        
        metadata_path = f"{DOWNLOAD_PATH}/metadata/{file_id}.json"
        fixed_transcript_path = f"{DOWNLOAD_PATH}/bert_fixed/{file_id}.txt"
        sensevoice_path = f"{DOWNLOAD_PATH}/sensevoice/{file_id}.txt"

        if not os.path.exists(metadata_path) or not os.path.exists(sensevoice_path) or not os.path.exists(fixed_transcript_path):
            count_dict["path_dne"] += 1
            continue

        with open(metadata_path, "r") as f:
            metadata = json.loads(f.read())
            
        try:
            a, b, c = metadata['channel'], metadata['labels_1'], metadata['title']
        except:
            count_dict["bad_meta"] += 1
            continue

        with open(fixed_transcript_path, "r") as f:
            bert_fixed = f.read()
        
        with open(sensevoice_path, "r") as f:
            sensevoice = remove_emotion_and_event_tokens(f.read())
        filtered_sv = convert_to_traditional(sensevoice)

        if len(filtered_sv.strip()) < 4 or len(metadata['labels_1'].strip()) < 4:
            count_dict["short_txt"] += 1
            continue
        if compute_cer(filtered_sv, metadata['labels_1']) > 0.4:
            print("="*20)
            print(filtered_sv)
            print()
            print(metadata['labels_1'])
            print("="*20)
            count_dict["bad_cer"] += 1
            continue

    print(count_dict)