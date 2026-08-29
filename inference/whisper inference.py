!pip install torch torchvision torchaudio
!pip install datasets transformers


!pip install -U datasets --quiet
!pip install fsspec==2023.9.2 --quiet


!pip install huggingface_hub




import io
import os
import gc
import torch
import pickle
import torchaudio
import numpy as np
from datasets import load_dataset, Audio
from transformers import WhisperModel, WhisperProcessor


# === Device and Drive save path ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_ROOT = "/content/drive/MyDrive/Layer Representations"  # ✅ Save directly to Drive
os.makedirs(SAVE_ROOT, exist_ok=True)


# === Model and dataset lists ===
model_list = [
    "openai/whisper-large-v2"
]
dataset_names = [
    "PranavBhalerao/cam_assess",
    "PranavBhalerao/SAA",
    "PranavBhalerao/l2-arctic-dataset-250",
    "PranavBhalerao/sandi",
    "PranavBhalerao/CommonVoice_accent_stratified",
    "PranavBhalerao/cmu-arctic-train",
    "PranavBhalerao/ALLSSTAR_2"
]


# === Features ===
continuous_features = [
    "duration", "f0_mean", "f0_median", "f0_min", "f0_max",
    "intensity_mean", "intensity_median", "intensity_min", "intensity_max",
    "F1_mean", "F1_median", "F1_min", "F1_max",
    "F2_mean", "F2_median", "F2_min", "F2_max",
    "F3_mean", "F3_median", "F3_min", "F3_max",
    "F3_minus_F2_mean", "F3_minus_F2_median", "F3_minus_F2_min", "F3_minus_F2_max"
]
categorical_features = ["gender", "l1_background"]


def make_folder(path):
    os.makedirs(path, exist_ok=True)


def clean_gender(example):
    return {'gender': example['gender'] if example['gender'] is not None else None}


def extract_features_and_representations(model, processor, dataset, features_to_use):
    print(f"Extracting features and representations... Total examples: {len(dataset)}")
    num_layers = model.config.num_hidden_layers + 1
    reps_by_layer = [[] for _ in range(num_layers)]
    labels_dict = {feat: [] for feat in features_to_use}


    for i, example in enumerate(dataset):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing example {i + 1}/{len(dataset)}")


        audio_bytes = example["audio"].get("bytes")
        if not audio_bytes:
            continue
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, sampling_rate = torchaudio.load(audio_buffer)
        waveform = waveform.to(DEVICE)


        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000).to(DEVICE)
            waveform = resampler(waveform)


        inputs = processor(waveform.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(DEVICE)


        with torch.no_grad():
            encoder_outputs = model.encoder(input_features, output_hidden_states=True)
        hidden_states = encoder_outputs.hidden_states


        for layer_idx, hs in enumerate(hidden_states):
            mean_rep = hs.squeeze(0).mean(dim=0).cpu().numpy()
            reps_by_layer[layer_idx].append(mean_rep)


        for feat in features_to_use:
            labels_dict[feat].append(example.get(feat))


        del waveform, input_features, encoder_outputs, hidden_states
        torch.cuda.empty_cache()
        gc.collect()


    reps_by_layer = [np.stack(layer) for layer in reps_by_layer]
    return reps_by_layer, labels_dict


# === MAIN SCRIPT ===
for model_name in model_list:
    model_tag = model_name.split("/")[-1]
    model_save_dir = os.path.join(SAVE_ROOT, model_tag)  # ✅ Save under Drive
    make_folder(model_save_dir)


    for dataset_name in dataset_names:
        dataset_key = dataset_name.split("/")[-1]
        save_path = os.path.join(model_save_dir, f"{dataset_key}.pkl")


        if os.path.exists(save_path):
            print(f"✅ Already exists: {save_path}, skipping.")
            continue


        print(f"\n🔁 Loading model: {model_name}")
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperModel.from_pretrained(model_name).to(DEVICE).eval()


        print(f"📊 Processing dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.cast_column("audio", Audio(decode=False))


        if dataset_key == "sandi":
            features_to_use = continuous_features
        else:
            features_to_use = continuous_features + categorical_features


        if 'gender' in dataset.column_names:
            dataset = dataset.map(clean_gender)
            dataset = dataset.filter(lambda x: x['gender'] is not None)


        reps_by_layer, labels = extract_features_and_representations(
            model, processor, dataset, features_to_use)


        with open(save_path, "wb") as f:
            pickle.dump({
                "reps_by_layer": reps_by_layer,
                "labels": labels
            }, f)
        print(f"✅ Saved to Drive: {save_path}")


        del model, processor, dataset, reps_by_layer, labels
        torch.cuda.empty_cache()
        gc.collect()
