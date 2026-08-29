# =====================================================
# Granite Speech 3.3-2b Setup for Google Colab
# =====================================================
from google.colab import drive
drive.mount('/content/drive')

# Install core dependencies with audio decode=True support
!pip install git+https://github.com/huggingface/transformers.git --quiet
!pip install -q accelerate==1.3.0

# CRITICAL: datasets[audio] enables torchcodec for decode=True
!pip install -U datasets[audio]

# Upgrade PyTorch to match datasets[audio] requirements
!pip install torch==2.9.0 torchvision torchaudio --upgrade

# Audio processing
!pip install -q soundfile==0.13.1
!pip install -q scipy==1.15.2

# Additional utilities
!pip install -q pillow==11.1.0
!pip install -q backoff==2.2.1
!pip install -q peft==0.13.2
!pip install -q huggingface_hub

print("✅ Setup complete!")
print("⚠️ IMPORTANT: Restart runtime now (Runtime → Restart runtime)")
print("Then run the extraction script.")


# =====================================================
# Granite Speech 3.3-2b Audio Layer Extractor
# (Phoneme-only, FP16, TRUE BATCHING, Stream-to-Temp, One-PKL Output, RAM-SAFE)
# =====================================================

import io, os, gc, pickle, shutil, glob, warnings
import numpy as np
import torch, torchaudio, soundfile as sf
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from tqdm.auto import tqdm
import logging

logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch.utils._sympy").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"
torch.set_float32_matmul_precision("high")
gc.set_threshold(200, 5, 5)

# ---------------- USER CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_ROOT = "/content/drive/MyDrive/Layer Representations"
LOCAL_TMP = "/content/tmp_layers"
LOCAL_DATASET_ROOT = "/content/datasets"

os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(LOCAL_TMP, exist_ok=True)
os.makedirs(LOCAL_DATASET_ROOT, exist_ok=True)

BATCH_SIZE_MAP = {
    "granite-speech-3.3-2b": 256,  # Will actually use batching now!
}

TARGET_SR = 16000
GC_EVERY = 5000  # Reduced frequency
DEBUG_LIMIT = None
USE_PRELOAD = False
FLUSH_EVERY = 5000

dataset_names = [
    "PranavBhalerao/SAA_phonemes",
]

model_list = [
    "ibm-granite/granite-speech-3.3-2b",
]

# =====================================================
# Helpers
# =====================================================
def make_folder(path):
    os.makedirs(path, exist_ok=True)

def normalize_tag(model_tag: str) -> str:
    return model_tag

def stream_merge_and_save(final_save_path, layer_tmp_paths, labels, hidden_size):
    """RAM-safe incremental merge of temp .npy chunks into one .pkl."""
    tmp_path = final_save_path + ".tmp"
    meta = {"labels": labels, "reps_by_layer": []}

    print(f"💾 Streaming merge → {final_save_path}")

    with open(tmp_path, "wb") as fout:
        pickle.dump(meta, fout, protocol=pickle.HIGHEST_PROTOCOL)
        fout.flush(); os.fsync(fout.fileno())

        for li, path in enumerate(layer_tmp_paths):
            gc.collect()
            if not os.path.exists(path):
                tqdm.write(f"⚠️ Missing layer {li} → writing empty array")
                arr = np.zeros((0, hidden_size), dtype=np.float16)
                pickle.dump(arr, fout, protocol=pickle.HIGHEST_PROTOCOL)
                continue

            chunk_count, total_rows = 0, 0
            with open(path, "rb") as fin:
                while True:
                    try:
                        chunk = np.load(fin, allow_pickle=False, mmap_mode=None)
                    except Exception:
                        break
                    if not isinstance(chunk, np.ndarray) or chunk.size == 0:
                        continue
                    if chunk.dtype != np.float16:
                        chunk = chunk.astype(np.float16, copy=False)
                    pickle.dump(chunk, fout, protocol=pickle.HIGHEST_PROTOCOL)
                    total_rows += len(chunk)
                    chunk_count += 1
                    del chunk
                    if chunk_count % 5 == 0:
                        fout.flush(); os.fsync(fout.fileno())
                        gc.collect()
            os.remove(path)
            tqdm.write(f"✅ Layer {li:02d}: {total_rows:,} rows merged.")
            gc.collect()

    shutil.move(tmp_path, final_save_path)
    print(f"💾 Safe merge complete → {final_save_path}")

# =====================================================
# Dataset extraction / preload
# =====================================================
def extract_wavs_from_dataset(dataset_name, out_dir):
    key = dataset_name.split("/")[-1]
    print(f"📥 Downloading + extracting {dataset_name} → {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset(dataset_name, split="train")
    for i, ex in enumerate(tqdm(ds, desc=f"Extracting {key}")):
        try:
            wav = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            phon = ex.get("phoneme", "none")
            path = os.path.join(out_dir, f"{i:05d}_{phon}.wav")
            sf.write(path, wav, sr)
        except Exception as e:
            tqdm.write(f"⚠️ Skipped {i}: {e}")
            continue
    print(f"✅ Extracted {len(glob.glob(f'{out_dir}/*.wav')):,} files.")
    return out_dir

def preload_dataset(root):
    files = sorted(glob.glob(f"{root}/*.wav"))
    print(f"🧠 Preloading {len(files):,} audio files into RAM …")
    all_audio, all_phonemes = [], []
    for path in tqdm(files, desc="Preloading"):
        try:
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
            if sr != TARGET_SR: wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
            all_audio.append(wav.squeeze(0).numpy())
            all_phonemes.append(os.path.basename(path).split("_", 1)[-1].replace(".wav", ""))
        except Exception as e:
            tqdm.write(f"⚠️ Skipped {path}: {e}")
            continue
    print(f"✅ Preloaded {len(all_audio):,} clips (~{sum(len(x) for x in all_audio)/TARGET_SR/3600:.1f} h total).")
    return [{"audio": a, "phoneme": p} for a, p in zip(all_audio, all_phonemes)]

class LocalWavDataset:
    def __init__(self, root):
        self.files = sorted(glob.glob(f"{root}/*.wav"))
        print(f"📂 Found {len(self.files):,} audio files in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        path = self.files[i]
        phoneme = os.path.basename(path).split("_", 1)[-1].replace(".wav", "")
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if sr != TARGET_SR: wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
        return {"audio": wav.squeeze(0).numpy(), "phoneme": phoneme}

# =====================================================
# Core extraction function (TRUE BATCHING)
# =====================================================
def extract_and_save_one_pkl(model, processor, ds, init_batch_size, final_save_path, debug_limit=None):
    from google.colab import drive
    device = next(model.parameters()).device

    # Granite: Get number of encoder layers
    n_layers = len(model.encoder.layers)

    tqdm.write(f"📐 Config: {n_layers} encoder layers")

    layer_tmp_paths = [os.path.join(LOCAL_TMP, f"layer_{i:02d}.npy") for i in range(n_layers)]
    for p in layer_tmp_paths:
        if os.path.exists(p): os.remove(p)

    labels = {"phoneme": []}
    N = len(ds)
    limit = N if debug_limit is None else min(N, debug_limit)
    progress = tqdm(total=limit, unit="samples", desc="Extracting", leave=True)
    cursor, total_seen, adaptive_bs = 0, 0, init_batch_size
    hidden_size = None

    system_prompt = (
        "Knowledge Cutoff Date: April 2024.\n"
        "Today's Date: November 10, 2025.\n"
        "You are Granite, developed by IBM. You are a helpful AI assistant"
    )

    # OOM handling
    oom_count = 0

    def _gentle_shrink(bs):
        if bs > 64:  return max(int(bs * 0.75), 4)
        if bs > 16:  return max(int(bs * 0.80), 4)
        return max(bs - 2, 4)

    def _hard_backoff(bs):
        return max(int(bs * 0.5), 4)

    # Process in TRUE batches
    while cursor < limit:
        end = min(cursor + adaptive_bs, limit)
        batch = ds[cursor:end]

        waves = [b["audio"] for b in batch]
        phonemes = [b["phoneme"] for b in batch]

        try:
            # Create chat prompts for ENTIRE batch (like your working code)
            chats = []
            for phoneme in phonemes:
                user_prompt = f"<|audio|>{phoneme}"
                chats.append([
                    dict(role="system", content=system_prompt),
                    dict(role="user", content=user_prompt),
                ])

            # Apply chat template to all prompts at once
            prompts = [
                processor.tokenizer.apply_chat_template(
                    c, tokenize=False, add_generation_prompt=True
                ) for c in chats
            ]

            # TRUE BATCHING: Pass LISTS to processor (key difference!)
            model_inputs = processor(prompts, waves, return_tensors="pt", padding=True)
            model_inputs = {k: v.to(device, non_blocking=True) for k, v in model_inputs.items()}

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                outputs = model(**model_inputs, output_hidden_states=True)

            oom_count = 0  # Reset on success

            # Extract hidden states (skip embedding layer)
            hidden_states = outputs.hidden_states[1:1 + n_layers]

            for li in range(n_layers):
                layer = hidden_states[li]
                pooled = layer.mean(dim=1).cpu().numpy()

                if pooled.dtype != np.float16:
                    pooled = pooled.astype(np.float16, copy=False)

                # Write to temp file immediately (streaming)
                mode = "ab" if os.path.exists(layer_tmp_paths[li]) else "wb"
                with open(layer_tmp_paths[li], mode) as f:
                    np.save(f, pooled, allow_pickle=False)

                if hidden_size is None and li == 0:
                    hidden_size = pooled.shape[-1]
                    tqdm.write(f"📐 Inferred hidden_size={hidden_size}")

            labels["phoneme"].extend(phonemes)
            total_seen += len(batch)
            progress.update(len(batch))
            cursor = end

            # Periodic cleanup
            if total_seen % GC_EVERY == 0:
                gc.collect()
                torch.cuda.empty_cache()

        except RuntimeError as e:
            msg = str(e).lower()
            if ("out of memory" in msg) or ("cuda error: out of memory" in msg):
                oom_count += 1
                torch.cuda.empty_cache()
                gc.collect()

                if oom_count >= 3:
                    adaptive_bs = _hard_backoff(adaptive_bs)
                    oom_count = 0
                else:
                    adaptive_bs = _gentle_shrink(adaptive_bs)

                tqdm.write(f"⚠️ OOM → reducing batch size to {adaptive_bs}")
                continue
            else:
                raise e

    progress.close()
    torch.cuda.empty_cache()
    gc.collect()

    if hidden_size is None:
        raise RuntimeError("No samples processed!")

    # Remount Drive
    try:
        print("\n🔄 Re-mounting Google Drive...")
        drive.mount('/content/drive', force_remount=True)
        if os.path.ismount("/content/drive"):
            print("✅ Drive mounted")
    except Exception as e:
        print(f"⚠️ Drive error: {e}")

    # Final save
    if not os.path.ismount("/content/drive"):
        local_backup = f"/content/{os.path.basename(final_save_path)}"
        stream_merge_and_save(local_backup, layer_tmp_paths, labels, hidden_size)
        print(f"⚠️ Saved locally: {local_backup}")
    else:
        stream_merge_and_save(final_save_path, layer_tmp_paths, labels, hidden_size)

    del labels
    gc.collect()
    torch.cuda.empty_cache()

# =====================================================
# MAIN
# =====================================================
print(f"\n🖥️ GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

for dataset_name in tqdm(dataset_names, desc="Datasets", unit="ds"):
    dataset_key = dataset_name.split("/")[-1]
    local_dir = os.path.join(LOCAL_DATASET_ROOT, dataset_key)

    models_needed = []
    for model_name in model_list:
        model_tag = model_name.split("/")[-1]
        short_tag = normalize_tag(model_tag)
        save_path = os.path.join(SAVE_ROOT, short_tag, "phonemes", f"{dataset_key}.pkl")
        if not os.path.exists(save_path):
            models_needed.append(model_name)

    if not models_needed:
        print(f"✅ All models already processed for {dataset_key} — skipping dataset.")
        continue

    if not glob.glob(f"{local_dir}/*.wav"):
        print(f"📥 Extracting dataset from HF: {dataset_name}")
        local_dir = extract_wavs_from_dataset(dataset_name, out_dir=local_dir)
    else:
        print(f"✅ Local WAVs for {dataset_key} already exist — using cached files.")

    ds = preload_dataset(local_dir) if USE_PRELOAD else LocalWavDataset(local_dir)

    for model_name in tqdm(models_needed, desc=f"Models for {dataset_key}", unit="model", leave=False):
        model_tag = model_name.split("/")[-1]
        short_tag = normalize_tag(model_tag)

        print("\n" + "="*80)
        print(f"🚀 STARTING MODEL: {model_tag}  ×  DATASET: {dataset_key}")
        print("="*80)

        model_dir = os.path.join(SAVE_ROOT, short_tag)
        phoneme_dir = os.path.join(model_dir, "phonemes")
        make_folder(phoneme_dir)
        save_path = os.path.join(phoneme_dir, f"{dataset_key}.pkl")

        tqdm.write(f"🔁 Loading {short_tag} for {dataset_key}")

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        ).eval()

        INIT_BS = BATCH_SIZE_MAP.get(short_tag, 32)
        tqdm.write(f"📊 Extracting features (FP16, TRUE BATCHING, batch_size={INIT_BS})…")

        extract_and_save_one_pkl(
            model=model,
            processor=processor,
            ds=ds,
            init_batch_size=INIT_BS,
            final_save_path=save_path,
            debug_limit=DEBUG_LIMIT,
        )

        del model, processor
        torch.cuda.empty_cache()
        gc.collect()

    del ds
    gc.collect()
    torch.cuda.empty_cache()

print("\n🎯 Done. All datasets processed for all models.")
print("⚡ FP16 TRUE BATCHING with conditional dataset extraction.")
