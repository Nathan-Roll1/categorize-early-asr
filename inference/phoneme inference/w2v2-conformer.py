# --- Mount Drive ---
from google.colab import drive
drive.mount('/content/drive')

# === RECREATE WORKING COLAB ENVIRONMENT ===

# 🧱 PyTorch core (GPU build for CUDA 12.6)
!pip install -q torch==2.8.0+cu126 torchaudio==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# 🤗 Hugging Face stack + audio utils
!pip install -q transformers==4.57.1 datasets==3.0.1 soundfile==0.13.1

# (optional, but useful for plotting/progress bars)
!pip install -q tqdm numpy matplotlib

!pip install accelerate
!pip install huggingface_hub


# =====================================================
# 🧠 Wav2Vec2-Conformer Phoneme Extractor (FP32 / RAM-safe / Streamed)
# (Phoneme-only • Cached 16k WAVs • One-PKL Output)
# =====================================================

import os, gc, io, glob, pickle, shutil, warnings
import numpy as np
import torch, torchaudio, soundfile as sf
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ConformerModel
from tqdm.auto import tqdm
import logging

# ---------------- Setup ----------------
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
os.environ["TRANSFORMERS_NO_TF"] = "1"

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
gc.set_threshold(200, 5, 5)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_ROOT = "/content/drive/MyDrive/Layer Representations/w2v2-conformer"
LOCAL_TMP = "/content/tmp_conformer"
DATA_ROOT = "/content/datasets"
os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(LOCAL_TMP, exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)

# ---------------- Config ----------------
MODEL_NAME = "facebook/wav2vec2-conformer-rel-pos-large-960h-ft"
TARGET_SR = 16000
INIT_BS = 512          # slightly smaller batch for FP32
FLUSH_EVERY = 2000
GC_EVERY = 4000
DEBUG_LIMIT = None
USE_PRELOAD = False

DATASETS = [
    #"PranavBhalerao/cmu-arctic-train_phonemes",
    #"PranavBhalerao/ALLSSTAR_2_phonemes",
    "PranavBhalerao/SAA_phonemes",
]

# =====================================================
# Helpers
# =====================================================
def make_folder(p): os.makedirs(p, exist_ok=True)

def extract_wavs_from_dataset(dataset_name, out_dir, limit=None):
    key = dataset_name.split("/")[-1]
    cache_dir = os.path.join(out_dir, key, "_16k_cache")
    if glob.glob(f"{cache_dir}/*.wav"):
        print(f"✅ Cached WAVs already exist → {cache_dir}")
        return cache_dir

    print(f"📥 Extracting + resampling {dataset_name} → {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)
    ds = load_dataset(dataset_name, split="train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    for i, ex in enumerate(tqdm(ds, desc=f"Resampling→16k ({key})")):
        try:
            wav = ex["audio"]["array"]
            sr = ex["audio"]["sampling_rate"]
            phon = ex.get("phoneme", "none")
            if sr != TARGET_SR:
                wav = torchaudio.transforms.Resample(sr, TARGET_SR)(torch.tensor(wav)).numpy()
            sf.write(os.path.join(cache_dir, f"{i:05d}_{phon}.wav"), wav, TARGET_SR)
        except Exception as e:
            tqdm.write(f"⚠️ Skipped {i}: {e}")
            continue
    print(f"✅ Extracted {len(glob.glob(f'{cache_dir}/*.wav')):,} files.")
    return cache_dir


class LocalWavDataset:
    def __init__(self, root, limit=None):
        self.files = sorted(glob.glob(f"{root}/*.wav"))
        if limit is not None:
            self.files = self.files[:limit]
        print(f"📂 Found {len(self.files):,} WAVs in {root}")
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        path = self.files[i]
        phon = os.path.basename(path).split("_", 1)[-1].replace(".wav", "")
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if sr != TARGET_SR: wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
        return {"audio": wav.squeeze(0).numpy(), "phoneme": phon}


def masked_time_mean(hidden, attn_mask):
    if attn_mask is None:
        return hidden.mean(dim=1)
    if attn_mask.shape[1] != hidden.shape[1]:
        attn_mask = torch.nn.functional.interpolate(
            attn_mask.unsqueeze(1).float(),
            size=hidden.shape[1],
            mode="nearest"
        ).squeeze(1).to(hidden.device)
    lengths = attn_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
    masked = hidden * attn_mask.unsqueeze(-1)
    return masked.sum(dim=1) / lengths


def stream_merge_and_save(final_save_path, layer_tmp_paths, labels, hidden_size):
    tmp_path = final_save_path + ".tmp"
    meta = {"labels": labels, "reps_by_layer": []}
    print(f"💾 Streaming merge → {final_save_path}")
    with open(tmp_path, "wb") as fout:
        pickle.dump(meta, fout, protocol=pickle.HIGHEST_PROTOCOL)
        fout.flush(); os.fsync(fout.fileno())

        for li, path in enumerate(layer_tmp_paths):
            gc.collect()
            if not os.path.exists(path):
                tqdm.write(f"⚠️ Missing layer {li}")
                arr = np.zeros((0, hidden_size), dtype=np.float32)
                pickle.dump(arr, fout)
                continue
            chunk_count, total_rows = 0, 0
            with open(path, "rb") as fin:
                while True:
                    try: arr = np.load(fin, allow_pickle=False)
                    except Exception: break
                    if arr.size == 0: continue
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32, copy=False)
                    pickle.dump(arr, fout)
                    total_rows += len(arr)
                    chunk_count += 1
                    if chunk_count % 3 == 0:
                        fout.flush(); os.fsync(fout.fileno())
            os.remove(path)
            tqdm.write(f"✅ Layer {li:02d}: {total_rows:,} vectors merged.")
            gc.collect()
    shutil.move(tmp_path, final_save_path)
    print(f"💾 Final PKL saved → {final_save_path}")

# =====================================================
# Core extraction
# =====================================================
def extract_and_save(model, processor, ds, save_path):
    num_layers = model.config.num_hidden_layers + 1
    hidden_size = model.config.hidden_size
    layer_tmp_paths = [os.path.join(LOCAL_TMP, f"layer_{i:02d}.npy") for i in range(num_layers)]
    for p in layer_tmp_paths:
        if os.path.exists(p): os.remove(p)

    labels = {"phoneme": []}
    cursor, total_seen, adaptive_bs = 0, 0, INIT_BS
    layer_buffers = [[] for _ in range(num_layers)]
    limit = len(ds)
    progress = tqdm(total=limit, unit="samples", desc="Extracting")

    def flush_buffers():
        for li, buf in enumerate(layer_buffers):
            if not buf: continue
            arr = np.concatenate(buf, axis=0)
            mode = "ab" if os.path.exists(layer_tmp_paths[li]) else "wb"
            with open(layer_tmp_paths[li], mode) as f: np.save(f, arr, allow_pickle=False)
            layer_buffers[li].clear(); del arr
        gc.collect()

    while cursor < limit:
        end = min(cursor + adaptive_bs, limit)
        batch = ds[cursor:end]
        waves = [b["audio"] for b in batch]
        phonemes = [b["phoneme"] for b in batch]

        proc = processor(waves, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        feats = proc.input_values.to(DEVICE)           # FP32
        attn = proc.attention_mask.to(DEVICE) if "attention_mask" in proc else None

        try:
            with torch.inference_mode():               # no autocast
                out = model(feats, attention_mask=attn, output_hidden_states=True)
            for li, hs in enumerate(out.hidden_states):
                pooled = masked_time_mean(hs, attn).cpu().numpy().astype(np.float32)
                layer_buffers[li].append(pooled)
            labels["phoneme"].extend(phonemes)
            total_seen += len(waves)
            progress.update(len(waves))
            cursor = end

            if total_seen % FLUSH_EVERY == 0 or cursor >= limit:
                flush_buffers()
            if total_seen % GC_EVERY == 0:
                gc.collect(); torch.cuda.empty_cache()

        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg:
                torch.cuda.empty_cache(); gc.collect()
                flush_buffers()
                adaptive_bs = max(int(adaptive_bs * 0.75), 4)
                tqdm.write(f"⚠️ OOM → batch ↓ {adaptive_bs}")
                continue
            else: raise e

    progress.close(); flush_buffers()
    torch.cuda.empty_cache(); gc.collect()

    from google.colab import drive
    try:
        drive.mount('/content/drive', force_remount=True)
        print("✅ Drive mounted for final save.")
    except Exception as e:
        print(f"⚠️ Drive mount failed: {e}")

    if not os.path.ismount("/content/drive"):
        local_backup = f"/content/{os.path.basename(save_path)}"
        stream_merge_and_save(local_backup, layer_tmp_paths, labels, hidden_size)
        print(f"⚠️ Saved locally: {local_backup}")
    else:
        stream_merge_and_save(save_path, layer_tmp_paths, labels, hidden_size)

# =====================================================
# Main
# =====================================================
print(f"\n🖥️ GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

print(f"🔁 Loading model: {MODEL_NAME}")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ConformerModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

for dataset_name in DATASETS:
    key = dataset_name.split("/")[-1]
    save_path = os.path.join(SAVE_ROOT, f"{key}.pkl")
    if os.path.exists(save_path):
        print(f"✅ Already done: {save_path}")
        continue

    local_wav_dir = extract_wavs_from_dataset(dataset_name, DATA_ROOT, limit=DEBUG_LIMIT)
    ds = LocalWavDataset(local_wav_dir, limit=DEBUG_LIMIT)

    print(f"🚀 Starting inference on {key} (debug={DEBUG_LIMIT}) ...")
    extract_and_save(model, processor, ds, save_path)
    print(f"✅ Done {key}")

del model, processor
gc.collect(); torch.cuda.empty_cache()
print("\n🎯 All datasets processed successfully.")
