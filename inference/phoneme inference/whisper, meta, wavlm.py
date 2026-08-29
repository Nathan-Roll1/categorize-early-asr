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


# =====================================================
# Whisper / Wav2Vec2 / HuBERT / WavLM Layer Extractor
# (Phoneme-only, FP16, Micro-batched, Stream-to-Temp, One-PKL Output, RAM-SAFE)
# =====================================================


import io, os, gc, pickle, shutil, glob, warnings, pickletools
import numpy as np
import torch, torchaudio, soundfile as sf
from datasets import load_dataset
from transformers import WhisperModel, WhisperProcessor, AutoModel, AutoFeatureExtractor
from tqdm.auto import tqdm
import logging
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch.utils._sympy").setLevel(logging.ERROR)


warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
os.environ["TRANSFORMERS_NO_TF"] = "1"
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
    # Whisper
    "whisper-tiny.en": 512, "whisper-tiny": 512,
    "whisper-base.en": 256, "whisper-base": 256,
    "whisper-small.en": 128, "whisper-small": 128,
    "whisper-medium.en": 128, "whisper-medium": 128,
    "whisper-large": 64, "whisper-large-v2": 64,
    "whisper-large-v3": 64, "whisper-large-v3-turbo": 96,
    # Wav2Vec2 / HuBERT / WavLM
    "wav2vec2-large-960h-lv60": 24,
    "hubert-xlarge-ls960-ft": 12,
    "hubert-large-ls960-ft": 24,
    "wavlm-large": 64,
}


TARGET_SR = 16000
GC_EVERY = 1000
DEBUG_LIMIT = None
USE_PRELOAD = False
FLUSH_EVERY = 5000


dataset_names = [
    "PranavBhalerao/cam_assess_phonemes",
    "PranavBhalerao/SAA_phonemes",
    "PranavBhalerao/l2-arctic-dataset-250_phonemes",
    "PranavBhalerao/CommonVoice_accent_stratified_phonemes",
    "PranavBhalerao/cmu-arctic-train_phonemes",
    "PranavBhalerao/ALLSSTAR_2_phonemes",
]


# 🔁 Models
model_list = [
    "microsoft/wavlm-large",
]


# =====================================================
# Helpers
# =====================================================
def make_folder(path): os.makedirs(path, exist_ok=True)


def normalize_tag(model_tag: str) -> str:
    return model_tag  # clean folder name (e.g. wavlm-large)


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
    tmp_path = final_save_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump({"labels": labels, "reps_by_layer": []}, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush(); os.fsync(f.fileno())
        for li, path in enumerate(layer_tmp_paths):
            gc.collect()
            if not os.path.exists(path):
                arr = np.zeros((0, hidden_size), dtype=np.float32)
            else:
                all_chunks = []
                with open(path, "rb") as f_in:
                    while True:
                        try:
                            all_chunks.append(np.load(f_in, allow_pickle=False))
                        except Exception:
                            break
                os.remove(path)
                arr = np.concatenate(all_chunks, axis=0) if all_chunks else np.zeros((0, hidden_size), dtype=np.float32)
                del all_chunks
            f.write(pickletools.optimize(pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)))
            f.flush(); os.fsync(f.fileno())
            del arr; gc.collect()
    shutil.move(tmp_path, final_save_path)
    print(f"💾 Incrementally saved → {final_save_path}")


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
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        if isinstance(i, slice): return [self[j] for j in range(*i.indices(len(self)))]
        path = self.files[i]
        phoneme = os.path.basename(path).split("_", 1)[-1].replace(".wav", "")
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if sr != TARGET_SR: wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
        return {"audio": wav.squeeze(0).numpy(), "phoneme": phoneme}


# =====================================================
# Core extraction
# =====================================================
def detect_backend(model_name: str):
    m = model_name.lower()
    if m.startswith("openai/whisper"):
        return "whisper"
    if "wavlm" in m:
        return "wavlm"
    return "wav2vec2like"


class ManualCollator:
    def __init__(self, processor, backend: str, target_sr=16000):
        self.processor = processor
        self.backend = backend
        self.target_sr = target_sr


    def __call__(self, examples):
        waves = [ex["audio"] for ex in examples]
        phonemes = [ex.get("phoneme") for ex in examples]


        if self.backend == "whisper":
            proc = self.processor(waves, sampling_rate=self.target_sr, return_tensors="pt", padding=True, truncation=True)
            inputs = proc.input_features
            attn = getattr(proc, "attention_mask", None)
        else:
            proc = self.processor(waves, sampling_rate=self.target_sr, return_tensors="pt", padding=True)
            inputs = proc.input_values
            attn = getattr(proc, "attention_mask", None)


        return {"inputs": inputs, "attention_mask": attn, "phonemes": phonemes, "count": len(waves)}


def run_forward(model, backend, inputs, attention_mask):
    if backend == "whisper":
        return model.encoder(inputs, attention_mask=attention_mask, output_hidden_states=True)
    else:
        return model(inputs, attention_mask=attention_mask, output_hidden_states=True)


def extract_and_save_one_pkl(model, processor, backend, ds, init_batch_size, final_save_path, debug_limit=None):
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers + 1
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "d_model", None)
    assert hidden_size is not None, "Could not determine model hidden size."


    layer_tmp_paths = [os.path.join(LOCAL_TMP, f"layer_{i:02d}.npy") for i in range(num_layers)]
    for p in layer_tmp_paths:
        if os.path.exists(p): os.remove(p)


    labels = {"phoneme": []}
    collator = ManualCollator(processor, backend=backend, target_sr=TARGET_SR)
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
            with open(layer_tmp_paths[li], mode) as f: np.save(f, arr, allow_pickle=False)
            layer_buffers[li].clear(); del arr
        gc.collect()


    while cursor < limit:
        end = min(cursor + adaptive_bs, limit)
        batch = collator(ds[cursor:end])
        feats, attn = batch["inputs"], batch.get("attention_mask")


        if backend == "whisper":
            pad_len = 3000 - feats.shape[-1]
            if pad_len > 0:
                feats = torch.nn.functional.pad(feats, (0, pad_len))
                if attn is not None:
                    attn = torch.nn.functional.pad(attn, (0, pad_len))


        feats = feats.to(device, dtype=torch.float16, non_blocking=True)
        attn = attn.to(device, non_blocking=True) if attn is not None else None


        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                enc_out = run_forward(model, backend, feats, attn)
            for li, hs in enumerate(enc_out.hidden_states):
                pooled = masked_time_mean(hs, attn).cpu().numpy()
                layer_buffers[li].append(pooled)
            labels["phoneme"].extend(batch["phonemes"])
            total_seen += batch["count"]
            progress.update(batch["count"])
            cursor = end
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache(); gc.collect()
                adaptive_bs = max(int(adaptive_bs * 0.5), 4)
                tqdm.write(f"⚠️ OOM → reducing batch size to {adaptive_bs}")
                continue
            else:
                raise e


        if total_seen % FLUSH_EVERY == 0 or cursor >= limit: _flush_buffers_to_temp()
        if total_seen % GC_EVERY == 0: gc.collect(); torch.cuda.empty_cache()


    progress.close(); _flush_buffers_to_temp()
    torch.cuda.empty_cache(); gc.collect()
    stream_merge_and_save(final_save_path, layer_tmp_paths, labels, hidden_size)
    del labels; gc.collect(); torch.cuda.empty_cache()


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


        backend = detect_backend(model_name)
        tqdm.write(f"🔁 Loading {short_tag} ({backend}) for {dataset_key}")


        if backend == "whisper":
            processor = WhisperProcessor.from_pretrained(model_name)
            model = WhisperModel.from_pretrained(model_name).to(DEVICE).half().eval()
            to_compile = getattr(model, "encoder", model)
        else:
            # Wav2Vec2 / HuBERT / WavLM → raw waveform via AutoFeatureExtractor (no vocab)
            processor = AutoFeatureExtractor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(DEVICE).half().eval()
            to_compile = model


        try:
            to_compile = torch.compile(to_compile)
            if backend == "whisper":
                model.encoder = to_compile
            else:
                model = to_compile
            tqdm.write("🧩 Model compiled for optimized inference.")
        except Exception as e:
            tqdm.write(f"⚠️ Could not compile model: {e}")


        INIT_BS = BATCH_SIZE_MAP.get(short_tag, BATCH_SIZE_MAP.get(model_tag, 16))
        tqdm.write(f"📊 Extracting features (FP16, micro-batched, compiled)…")


        extract_and_save_one_pkl(
            model=model,
            processor=processor,
            backend=backend,
            ds=ds,
            init_batch_size=INIT_BS,
            final_save_path=save_path,
            debug_limit=DEBUG_LIMIT,
        )


        del model, processor
        torch.cuda.empty_cache(); gc.collect()


    del ds
    gc.collect(); torch.cuda.empty_cache()


print("\n🎯 Done. All datasets processed for all models.")
print("⚡ FP16 micro-batched inference with conditional dataset extraction.")
