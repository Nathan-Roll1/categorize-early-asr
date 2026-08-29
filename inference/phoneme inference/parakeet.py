# =====================================================
# 🎯 Mount Google Drive
# =====================================================
from google.colab import drive
drive.mount('/content/drive')

# =====================================================
# ⚙️ Core PyTorch (CUDA 12.1)
# =====================================================
!pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121 -q

# =====================================================
# 🧰 Hugging Face + Datasets
# =====================================================
!pip install datasets==2.19.1 fsspec==2023.9.2 soundfile \
  huggingface_hub hf_transfer backoff peft --quiet

# =====================================================
# 🧩 Transformers (latest main branch)
# =====================================================
!pip uninstall -y transformers > /dev/null
!pip install https://github.com/huggingface/transformers/archive/main.zip --quiet

# =====================================================
# 🧠 NVIDIA NeMo (for Parakeet / Canary / ASR)
# =====================================================
!pip install nemo_toolkit[asr]==2.0.0 pytorch-lightning==2.3.1 \
  hydra-core==1.3.2 omegaconf==2.3.0 --quiet

# (Optional but recommended)
!pip install accelerate==0.33.0 --quiet



# =====================================================
# ⚡ Parakeet-TDT-0.6B-v2 Phoneme Extractor (FINAL RAM-SAFE STREAMING)
#   FP16 • Adaptive Batch • True Append • Stream-to-PKL • Drive-Safe
# =====================================================

import os, gc, glob, pickle, warnings, psutil, time, shutil
import numpy as np
import torch, torchaudio, soundfile as sf
from tqdm.auto import tqdm
from datasets import load_dataset
from google.colab import drive
from nemo.collections.asr.models import ASRModel

# ---------------- ENV ----------------
try:
    drive.mount('/content/drive', force_remount=False)
except Exception as e:
    print("⚠️ Drive mount issue:", e)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
gc.set_threshold(200, 5, 5)

# ---------------- CONFIG ----------------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR   = 16000
INIT_BS     = 512
GC_STEPS    = 200
FLUSH_EVERY = 5
RAM_THRESHOLD_GB = 5
DEBUG_LIMIT = None

SAVE_ROOT = "/content/drive/MyDrive/Layer Representations"
TMP_DIR   = "/content/tmp_layers"
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

datasets = [
    "PranavBhalerao/cmu-arctic-train_phonemes",
    "PranavBhalerao/ALLSSTAR_2_phonemes",
    "PranavBhalerao/SAA_phonemes",
]

# ---------------- LOAD MODEL ----------------
print(f"🔁 Loading NeMo model: {MODEL_NAME}")
nemo_model = ASRModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
encoder, preproc = nemo_model.encoder, nemo_model.preprocessor
num_layers = len(encoder.layers)
print(f"✅ Loaded Parakeet-TDT-0.6B-v2 with {num_layers} encoder layers")

# ---------------- DATA HELPERS ----------------
def extract_wavs_from_dataset(name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset(name, split="train")
    for i, ex in enumerate(tqdm(ds, total=len(ds), desc=f"Extracting {name}")):
        audio = ex["audio"]
        wav, sr = audio["array"], audio["sampling_rate"]
        phon = ex.get("phoneme", "none")
        sf.write(f"{out_dir}/{i:06d}_{phon}.wav", wav, sr)
    return out_dir

class LocalWavDataset:
    def __init__(self, root, limit=None):
        files = sorted(glob.glob(f"{root}/*.wav"))
        self.files = files if (limit is None) else files[:limit]
        print(f"📂 Found {len(self.files):,} WAVs in {root}")
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        p = self.files[idx]
        phon = os.path.basename(p).split("_",1)[-1].rsplit(".",1)[0]
        wav, sr = torchaudio.load(p)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if sr != TARGET_SR: wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
        return wav.squeeze(0), phon

# ---------------- ENCODER HOOKS ----------------
def hook_encoder_layers(enc):
    layer_outs = [[] for _ in range(num_layers)]
    handles = []
    def make_hook(li):
        def hook(_, __, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            if x.dim() == 3 and x.shape[1] > x.shape[0]:
                x = x.transpose(0, 1)
            elif x.dim() == 3 and x.shape[0] == 1:
                x = x.squeeze(0)
            pooled = x.mean(dim=1).detach().float().cpu().numpy()
            layer_outs[li].append(pooled)
        return hook
    for i, m in enumerate(enc.layers):
        handles.append(m.register_forward_hook(make_hook(i)))
    return handles, layer_outs

# ---------------- STREAM HELPERS ----------------
def append_to_npy(layer_idx, arr):
    """True append (no reload)"""
    path = os.path.join(TMP_DIR, f"layer_{layer_idx:02d}.npy")
    mode = "ab" if os.path.exists(path) else "wb"
    with open(path, mode) as f:
        np.save(f, arr, allow_pickle=False)
        f.flush(); os.fsync(f.fileno())

def stream_load_layer(path):
    """Yield appended np.save(ab) chunks"""
    with open(path, "rb") as f:
        while True:
            try: yield np.load(f, allow_pickle=False)
            except Exception: break

def flush_buffers(layer_buffers):
    for li, buf in enumerate(layer_buffers):
        if buf:
            arr = np.concatenate(buf, axis=0).astype(np.float32)
            append_to_npy(li, arr)
            buf.clear()
    torch.cuda.empty_cache(); gc.collect()

# ---------------- STREAM-TO-PKL MERGE ----------------
def merge_to_pkl(save_path, phonemes):
    """
    Stream each layer sequentially into one PKL.
    Keeps RAM usage <3 GB even for 80 GB total data.
    """
    tmp_pkl = save_path + ".tmp"
    print(f"💾 Stream-merging layers → {save_path}")
    with open(tmp_pkl, "wb") as f:
        meta = {"labels": {"phoneme": phonemes}, "num_layers": num_layers}
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush(); os.fsync(f.fileno())

        for li in range(num_layers):
            layer_path = os.path.join(TMP_DIR, f"layer_{li:02d}.npy")
            total = 0
            if not os.path.exists(layer_path):
                pickle.dump(np.zeros((0,512), dtype=np.float32), f)
                continue
            for chunk in stream_load_layer(layer_path):
                pickle.dump(chunk.astype(np.float32, copy=False), f, protocol=pickle.HIGHEST_PROTOCOL)
                total += len(chunk)
                if total % 10000 == 0:
                    f.flush(); os.fsync(f.fileno())
            print(f"✅ Layer {li:02d} merged ({total:,} samples)")
            gc.collect()

    try:
        shutil.move(tmp_pkl, save_path)
        os.sync()
        print(f"✅ Final PKL saved → {save_path}")
    except Exception as e:
        print(f"⚠️ Drive move failed: {e}")
        local_bak = f"/content/{os.path.basename(save_path)}"
        shutil.move(tmp_pkl, local_bak)
        print(f"💾 Local backup at: {local_bak}")

# ---------------- EXTRACTION LOOP ----------------
@torch.inference_mode()
def extract_batched(ds, save_path):
    handles, layer_outs = hook_encoder_layers(encoder)
    phonemes = []
    layer_buffers = [[] for _ in range(num_layers)]

    adaptive_bs = INIT_BS
    total = len(ds)
    progress = tqdm(total=total, desc="Extracting", unit="samples")

    i = 0
    while i < total:
        end = min(i + adaptive_bs, total)
        batch = [ds[j] for j in range(i, end)]
        wavs, phs = zip(*batch)
        phonemes.extend(phs)

        maxlen = max(w.shape[-1] for w in wavs)
        wav_tensor = torch.zeros(len(wavs), maxlen, dtype=torch.float32)
        for j, w in enumerate(wavs):
            wav_tensor[j, :w.shape[-1]] = w
        wav_tensor = wav_tensor.to(DEVICE)
        lengths = torch.tensor([w.shape[-1] for w in wavs], device=DEVICE)

        try:
            with torch.autocast("cuda", dtype=torch.float16):
                feats, feat_lens = preproc(input_signal=wav_tensor, length=lengths)
                _ = encoder(audio_signal=feats, length=feat_lens)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                adaptive_bs = max(int(adaptive_bs * 0.7), 4)
                torch.cuda.empty_cache(); gc.collect()
                tqdm.write(f"⚠️ OOM → reduced batch to {adaptive_bs}")
                continue
            else:
                raise e

        for li in range(num_layers):
            if layer_outs[li]:
                layer_buffers[li].extend(layer_outs[li])
                layer_outs[li].clear()

        if ((i // adaptive_bs + 1) % FLUSH_EVERY == 0) or (psutil.virtual_memory().used/1e9 > RAM_THRESHOLD_GB) or (end >= total):
            flush_buffers(layer_buffers)
        if (i // adaptive_bs + 1) % GC_STEPS == 0:
            torch.cuda.empty_cache(); gc.collect()

        progress.update(len(batch))
        i = end

    for h in handles: h.remove()
    flush_buffers(layer_buffers)
    progress.close()
    print("🧮 Merging all chunks …")
    merge_to_pkl(save_path, phonemes)

# ---------------- RUN ----------------

out_dir = os.path.join(SAVE_ROOT, "parakeet-tdt-0.6b-v2", "phonemes")
os.makedirs(out_dir, exist_ok=True)

for ds_name in datasets:
    ds_key = ds_name.split("/")[-1]
    local = f"/content/datasets/{ds_key}"
    if not glob.glob(f"{local}/*.wav"):
        print(f"📥 Extracting WAVs for {ds_key}")
        extract_wavs_from_dataset(ds_name, local)
    else:
        print(f"✅ Using cached WAVs for {ds_key}")

    ds = LocalWavDataset(local, limit=DEBUG_LIMIT)
    save_path = os.path.join(out_dir, f"{ds_key}.pkl")
    if os.path.exists(save_path):
        print(f"✅ Already exists: {save_path} — skipping")
        continue

    print(f"\n🚀 MODEL: {MODEL_NAME} × DATASET: {ds_key}")
    extract_batched(ds, save_path)

    print("🧹 Cleaning temp files …")
    for f in glob.glob(os.path.join(TMP_DIR, "layer_*.npy")):
        os.remove(f)

print("🎯 Done.")
try: nemo_model.to("cpu")
except: pass
del nemo_model, preproc, encoder
torch.cuda.empty_cache(); gc.collect()
