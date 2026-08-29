# =====================================================
# Phi-4-Multimodal Setup for Google Colab
# =====================================================
from google.colab import drive
drive.mount('/content/drive')

# Install exact versions that work best as of Nov 2025
!pip install -q transformers==4.48.2
!pip install -q accelerate==1.3.0
!pip install -U datasets[audio]
!pip install torch==2.9.0 torchvision torchaudio --upgrade
!pip install -q soundfile==0.13.1
!pip install -q scipy==1.15.2
!pip install -q pillow==11.1.0
!pip install -q backoff==2.2.1
!pip install -q peft==0.13.2
!pip install -q huggingface_hub

print("✅ Setup complete!")


# =====================================================
# Phi-4-Multimodal Audio Layer Extractor
# (Phoneme-only, FP16, Micro-batched, Stream-to-Temp, One-PKL Output, RAM-SAFE)
# =====================================================

import io, os, gc, pickle, shutil, glob, warnings
import numpy as np
import torch, torchaudio, soundfile as sf
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
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
    "Phi-4-multimodal-instruct": 128,  # Conservative for multimodal
}

TARGET_SR = 16000
GC_EVERY = 1000
DEBUG_LIMIT = None
USE_PRELOAD = False
FLUSH_EVERY = 5000

dataset_names = [
    "PranavBhalerao/SAA_phonemes",
    "PranavBhalerao/cmu-arctic-train_phonemes",
]

model_list = [
    "microsoft/Phi-4-multimodal-instruct",
]

# =====================================================
# Helpers (unchanged from original)
# =====================================================
def make_folder(path):
    os.makedirs(path, exist_ok=True)

def normalize_tag(model_tag: str) -> str:
    return model_tag

def masked_time_mean(hidden, attn_mask):
    """Mean-pool over time; skip attn_mask if mismatch."""
    if hidden.dim() != 3:
        raise ValueError(f"Unexpected hidden shape: {hidden.shape}")
    if hidden.shape[1] > hidden.shape[2]:
        hidden = hidden.transpose(1, 2)

    if attn_mask is None or attn_mask.shape[1] != hidden.shape[1]:
        return hidden.mean(dim=1)

    lengths = attn_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
    masked = hidden * attn_mask.unsqueeze(-1)
    return masked.sum(dim=1) / lengths

def stream_merge_and_save(final_save_path, layer_tmp_paths, labels, hidden_size):
    """
    RAM-safe incremental merge of temp .npy chunks into one .pkl.
    Each layer is streamed directly from disk, never fully loaded into RAM.
    """
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
# Dataset extraction / preload (unchanged)
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
# Phi-4-Multimodal Collator (TRUE BATCHING)
# =====================================================
class Phi4MultimodalCollator:
    def __init__(self, processor, target_sr=16000):
        self.processor = processor
        self.target_sr = target_sr

    def __call__(self, examples):
        waves = [ex["audio"] for ex in examples]
        phonemes = [ex.get("phoneme", "unknown") for ex in examples]

        # Create a unique prompt for each audio in the batch
        prompts = [
            f"<|user|><|audio_{i+1}|>Transcribe this audio.<|end|><|assistant|>"
            for i in range(len(waves))
        ]

        # Format all audios
        audios_with_sr = [
            (wav if isinstance(wav, np.ndarray) else np.array(wav), self.target_sr)
            for wav in waves
        ]

        # Process batch with multiple prompts
        inputs = self.processor(
            text=prompts,  # List of prompts
            audios=audios_with_sr,  # List of audios
            return_tensors="pt",
            padding=True
        )

        return {
            "inputs": inputs,
            "attention_mask": getattr(inputs, "attention_mask", None),
            "phonemes": phonemes,
            "count": len(waves)
        }


# =====================================================
# Core extraction function
# =====================================================
def extract_and_save_one_pkl(model, processor, ds, init_batch_size, final_save_path, debug_limit=None):
    from google.colab import drive
    device = next(model.parameters()).device

    # Phi-4-multimodal: 32 transformer layers + 1 embedding = 33 hidden states
    num_layers = model.config.num_hidden_layers + 1
    hidden_size = model.config.hidden_size  # 3072

    tqdm.write(f"📐 Config: {num_layers} layers, hidden_size={hidden_size}")

    layer_tmp_paths = [os.path.join(LOCAL_TMP, f"layer_{i:02d}.npy") for i in range(num_layers)]
    for p in layer_tmp_paths:
        if os.path.exists(p): os.remove(p)

    labels = {"phoneme": []}
    collator = Phi4MultimodalCollator(processor, target_sr=TARGET_SR)
    N = len(ds)
    limit = N if debug_limit is None else min(N, debug_limit)
    progress = tqdm(total=limit, unit="samples", desc="Extracting", leave=True)
    cursor, total_seen, adaptive_bs = 0, 0, init_batch_size
    layer_buffers = [[] for _ in range(num_layers)]

    def _flush_buffers_to_temp():
        for li, buf in enumerate(layer_buffers):
            if not buf: continue
            arr = np.concatenate(buf, axis=0)
            mode = "ab" if os.path.exists(layer_tmp_paths[li]) else "wb"
            with open(layer_tmp_paths[li], mode) as f:
                np.save(f, arr, allow_pickle=False)
            layer_buffers[li].clear()
            del arr
        gc.collect()

    # OOM-optimized adaptive batching (same as original)
    oom_count, since_last_grow = 0, 0
    grow_every_samples = max(FLUSH_EVERY // 2, 1000)

    def _gentle_shrink(bs):
        if bs > 64:  return max(int(bs * 0.75), 4)
        if bs > 16:  return max(int(bs * 0.80), 4)
        return max(bs - 2, 4)

    def _hard_backoff(bs):
        return max(int(bs * 0.5), 4)

    while cursor < limit:
        end = min(cursor + adaptive_bs, limit)
        batch = collator(ds[cursor:end])

        inputs = batch["inputs"]
        attn = batch.get("attention_mask")

        # Move to device
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        if attn is not None:
            attn = attn.to(device, non_blocking=True)

        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                outputs = model(**inputs, output_hidden_states=True)

            oom_count = 0
            since_last_grow += batch["count"]

            # Extract all hidden states (skip first which is embedding)
            for li, hs in enumerate(outputs.hidden_states):
                pooled = masked_time_mean(hs, attn).cpu().numpy()
                if pooled.dtype != np.float16:
                    pooled = pooled.astype(np.float16, copy=False)
                layer_buffers[li].append(pooled)

            labels["phoneme"].extend(batch["phonemes"])
            total_seen += batch["count"]
            progress.update(batch["count"])
            cursor = end

            # Adaptive batch size recovery
            if adaptive_bs < init_batch_size and since_last_grow >= grow_every_samples:
                new_bs = min(int(adaptive_bs * 1.25), init_batch_size)
                if new_bs != adaptive_bs:
                    adaptive_bs = new_bs
                    tqdm.write(f"📈 Recovering batch size → {adaptive_bs}")
                since_last_grow = 0

        except RuntimeError as e:
            msg = str(e).lower()
            if ("out of memory" in msg) or ("cuda error: out of memory" in msg):
                oom_count += 1
                torch.cuda.empty_cache()
                gc.collect()
                _flush_buffers_to_temp()
                gc.collect()
                torch.cuda.empty_cache()

                if oom_count >= 3:
                    adaptive_bs = _hard_backoff(adaptive_bs)
                    oom_count = 0
                else:
                    adaptive_bs = _gentle_shrink(adaptive_bs)

                tqdm.write(f"⚠️ OOM → reducing batch size to {adaptive_bs}")
                continue
            else:
                raise e

        # Periodic flushing
        if total_seen % FLUSH_EVERY == 0 or cursor >= limit:
            _flush_buffers_to_temp()
        if total_seen % GC_EVERY == 0:
            gc.collect()
            torch.cuda.empty_cache()

    progress.close()
    _flush_buffers_to_temp()
    torch.cuda.empty_cache()
    gc.collect()

    # Force remount Google Drive (same as original)
    try:
        print("\n🔄 Re-mounting Google Drive before final save...")
        drive.mount('/content/drive', force_remount=True)
        if os.path.ismount("/content/drive"):
            print("✅ Drive mount confirmed. Proceeding to final save.")
        else:
            print("⚠️ Drive mount failed. Will save locally and copy manually later.")
    except Exception as e:
        print(f"⚠️ Drive remount error: {e}")
        print("Saving locally to /content as fallback.")

    # Final save
    if not os.path.ismount("/content/drive"):
        local_backup = f"/content/{os.path.basename(final_save_path)}"
        stream_merge_and_save(local_backup, layer_tmp_paths, labels, hidden_size)
        print(f"⚠️ Saved locally: {local_backup}\nManually copy this to Drive later.")
    else:
        stream_merge_and_save(final_save_path, layer_tmp_paths, labels, hidden_size)

    del labels
    gc.collect()
    torch.cuda.empty_cache()

# =====================================================
# MAIN (same structure as original)
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

        # Load Phi-4-multimodal with optimal settings for Colab Nov 2025
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Try flash attention, fallback to eager if unavailable
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation='flash_attention_2',
            ).eval()
            tqdm.write("⚡ Using flash_attention_2 for optimal speed.")
        except Exception as e:
            tqdm.write(f"⚠️ Flash attention failed ({e}), using eager mode.")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                _attn_implementation="eager",
            ).eval()

        # Try torch.compile for additional optimization
        #try:
            #model = torch.compile(model, mode="reduce-overhead")
            #tqdm.write("🧩 Model compiled for optimized inference.")
        #except Exception as e:
            #tqdm.write(f"⚠️ Could not compile model: {e}")

        INIT_BS = BATCH_SIZE_MAP.get(short_tag, 12)
        tqdm.write(f"📊 Extracting features (FP16, micro-batched, compiled)…")

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
print("⚡ FP16 micro-batched inference with conditional dataset extraction.")
