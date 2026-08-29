# =====================================================
# 🎯 Mount Google Drive
# =====================================================
from google.colab import drive
drive.mount('/content/drive')

# =====================================================
# 🧹 Clean any old/conflicting installs
# =====================================================
!pip uninstall -y nemo_toolkit pytorch-lightning lightning_fabric lightning transformers > /dev/null

# =====================================================
# ⚙️ Core PyTorch (CUDA 12.1, works great on L4)
# =====================================================
!pip install -q torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121

# =====================================================
# 🧰 Hugging Face + datasets (stable)
# =====================================================
!pip install -q datasets==2.19.1 fsspec==2023.9.2 soundfile \
  huggingface_hub hf_transfer backoff peft accelerate==0.33.0

# =====================================================
# 🧩 Transformers (latest main branch)
#  - Ensures compatibility with NeMo Canary2 prompt formatter
# =====================================================
!pip install -q https://github.com/huggingface/transformers/archive/main.zip

# =====================================================
# 🧠 NVIDIA NeMo (for Canary / Canary2 / Parakeet)
#  - v2.1.0 adds prompt_formatter="canary2"
# =====================================================
!pip install -q nemo_toolkit[asr]==2.1.0 pytorch-lightning==2.3.1 \
  hydra-core==1.3.2 omegaconf==2.3.0

print("✅ Canary2 environment ready — NeMo 2.1.0, Transformers (main), Torch 2.3.1")


  # =====================================================
# ⚡ Canary-1B / Canary-1B-Flash Phoneme Extractor (High-Efficiency)
#    Lazy Load • Float16 • Memmap-Safe • Double Drive Save
# =====================================================

import os, gc, glob, pickle, warnings, psutil, shutil, time
import numpy as np
import torch, torchaudio, soundfile as sf
from tqdm.auto import tqdm
from datasets import load_dataset
from nemo.collections.asr.models import EncDecMultiTaskModel

# ---------------- ENV ----------------
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

# ---------------- CONFIG ----------------
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR    = 16000
BATCH_SIZE   = 512
GC_STEPS     = 200
DEBUG_LIMIT  = None

FLUSH_EVERY       = 10
MAX_RAM_PCT       = 0.8
SAVE_ROOT         = "/content/drive/MyDrive/Layer Representations"
TMP_ROOT          = "/content/tmp_layers"
LOCAL_SAVE_ROOT   = "/content/tmp_save"

os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(TMP_ROOT, exist_ok=True)
os.makedirs(LOCAL_SAVE_ROOT, exist_ok=True)

MODELS = [
    "nvidia/canary-1b",
    "nvidia/canary-1b-flash",
]

DATASETS = [

    #"PranavBhalerao/cmu-arctic-train_phonemes",
    #"PranavBhalerao/ALLSSTAR_2_phonemes",
    "PranavBhalerao/SAA_phonemes",
    "PranavBhalerao/CommonVoice_accent_stratified_phonemes",
]

# =====================================================
# 🔹 Data helpers
# =====================================================
def extract_wavs_from_dataset(name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset(name, split="train")
    for i, ex in enumerate(tqdm(ds, total=len(ds), desc=f"Extracting {name}")):
        audio = ex.get("audio")
        if not audio:
            continue
        wav, sr = audio["array"], audio["sampling_rate"]
        phon = ex.get("phoneme", "none")
        sf.write(f"{out_dir}/{i:06d}_{phon}.wav", wav, sr)
    return out_dir

class LocalWavDataset:
    def __init__(self, root, limit=None):
        files = sorted(glob.glob(f"{root}/*.wav"))
        self.files = files if (limit is None) else files[:limit]
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        p = self.files[idx]
        phon = os.path.basename(p).split("_", 1)[-1].rsplit(".", 1)[0]
        wav, sr = torchaudio.load(p)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != TARGET_SR:
            wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
        return wav.squeeze(0), phon

# =====================================================
# 🔹 Encoder hooks
# =====================================================
def hook_encoder_layers(enc):
    blocks = list(enc.layers) if hasattr(enc, "layers") else list(enc.children())
    num_layers = len(blocks)
    layer_outs = [[] for _ in range(num_layers)]
    handles = []

    def make_hook(li):
        def hook(_, __, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            if x.dim() == 3 and x.shape[1] > x.shape[0]:
                x = x.transpose(0, 1)
            elif x.dim() == 3 and x.shape[0] == 1:
                x = x.squeeze(0)
            pooled = x.mean(dim=1).detach().to(torch.float16).cpu().numpy()
            layer_outs[li].append(pooled)
        return hook

    for i, m in enumerate(blocks):
        handles.append(m.register_forward_hook(make_hook(i)))
    return handles, layer_outs, num_layers

# =====================================================
# 🔹 Memmap-safe incremental writing
# =====================================================
def append_to_npy(layer_idx, arr):
    path = os.path.join(TMP_ROOT, f"layer_{layer_idx:02d}.npy")
    if not os.path.exists(path):
        np.save(path, arr)
        return
    old = np.load(path, mmap_mode="r")
    total = old.shape[0] + arr.shape[0]
    tmp_path = path + ".tmp"
    mm = np.lib.format.open_memmap(tmp_path, mode='w+', dtype=old.dtype,
                                   shape=(total, arr.shape[1]))
    mm[:old.shape[0]] = old
    mm[old.shape[0]:] = arr
    del mm, old
    os.replace(tmp_path, path)

def flush_buffers(layer_buffers):
    for li, buf in enumerate(layer_buffers):
        if buf:
            chunk = np.concatenate(buf, axis=0)
            append_to_npy(li, chunk)
            buf.clear()
    torch.cuda.empty_cache(); gc.collect()

def merge_to_pkl(local_save, phonemes, num_layers):
    reps_by_layer = []
    for i in range(num_layers):
        layer_path = os.path.join(TMP_ROOT, f"layer_{i:02d}.npy")
        reps_by_layer.append(np.load(layer_path, mmap_mode="r"))
    payload = {"reps_by_layer": reps_by_layer, "labels": {"phoneme": phonemes}}
    with open(local_save, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"💾 Local PKL saved: {local_save}")

def safe_drive_save(local_path, drive_path):
    """Copy twice with verification."""
    for attempt in range(2):
        try:
            shutil.copy2(local_path, drive_path)
            time.sleep(2)
            if os.path.getsize(local_path) == os.path.getsize(drive_path):
                print(f"✅ Drive save verified on attempt {attempt+1}")
                return True
        except Exception as e:
            print(f"⚠️ Drive save failed ({e}), retrying …")
            time.sleep(5)
    print("❌ Drive save verification failed; please check manually.")
    return False

# =====================================================
# 🔹 Batched extraction
# =====================================================
@torch.inference_mode()
def extract_batched(preproc, encoder, ds, drive_save_path):
    handles, layer_outs, num_layers = hook_encoder_layers(encoder)
    phonemes = []
    layer_buffers = [[] for _ in range(num_layers)]
    total = len(ds)

    for bi in tqdm(range(0, total, BATCH_SIZE), desc="Batched Extraction"):
        batch = [ds[j] for j in range(bi, min(bi + BATCH_SIZE, total))]
        wavs, phs = zip(*batch)
        phonemes.extend(phs)

        maxlen = max(w.shape[-1] for w in wavs)
        wav_tensor = torch.zeros(len(wavs), maxlen, dtype=torch.float32)
        for j, w in enumerate(wavs):
            wav_tensor[j, :w.shape[-1]] = w
        wav_tensor = wav_tensor.to(DEVICE)
        lengths = torch.tensor([w.shape[-1] for w in wavs], device=DEVICE)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            feats, feat_lens = preproc(input_signal=wav_tensor, length=lengths)
            _ = encoder(audio_signal=feats, length=feat_lens)

        for li in range(num_layers):
            if layer_outs[li]:
                layer_buffers[li].extend(layer_outs[li])
                layer_outs[li].clear()

        ram_used = psutil.virtual_memory().percent / 100
        if ram_used > MAX_RAM_PCT or (bi // BATCH_SIZE + 1) % FLUSH_EVERY == 0 or (bi + BATCH_SIZE >= total):
            flush_buffers(layer_buffers)

        if (bi // BATCH_SIZE + 1) % GC_STEPS == 0:
            torch.cuda.empty_cache(); gc.collect()

    for h in handles: h.remove()
    flush_buffers(layer_buffers)

    print("🧮 Merging all chunks …")
    local_save = os.path.join(LOCAL_SAVE_ROOT, os.path.basename(drive_save_path))
    merge_to_pkl(local_save, phonemes, num_layers)
    print("📤 Copying to Drive …")
    safe_drive_save(local_save, drive_save_path)
    print("🧹 Cleaning local tmp")
    for f in glob.glob(os.path.join(TMP_ROOT, "layer_*.npy")):
        os.remove(f)
    gc.collect()

# =====================================================
# 🔹 MAIN LOGIC — Lazy-load only missing combos
# =====================================================
todo = []
for ds_name in DATASETS:
    ds_key = ds_name.split("/")[-1]
    for model_name in MODELS:
        short_name = model_name.split("/")[-1]
        out_dir = os.path.join(SAVE_ROOT, short_name, "phonemes")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{ds_key}.pkl")
        if not os.path.exists(save_path):
            todo.append((ds_name, ds_key, model_name, short_name))

if not todo:
    print("✅ All model–dataset combos already exist.")
else:
    print(f"🧾 Pending combinations: {len(todo)}")
    for (d, dk, m, sm) in todo:
        print(f" - {sm} × {dk}")

for ds_name, ds_key, model_name, short_name in todo:
    local_ws = f"/content/datasets/{ds_key}"
    if not glob.glob(f"{local_ws}/*.wav"):
        print(f"📥 Extracting WAVs for {ds_key}")
        extract_wavs_from_dataset(ds_name, local_ws)
    else:
        print(f"✅ Using cached WAVs for {ds_key}")
    ds = LocalWavDataset(local_ws, limit=DEBUG_LIMIT)

    print(f"\n🔁 Loading NeMo model: {model_name}")
    nemo_model = EncDecMultiTaskModel.from_pretrained(model_name).to(DEVICE).eval()
    preproc, encoder = nemo_model.preprocessor, nemo_model.encoder
    n_layers = len(encoder.layers) if hasattr(encoder, "layers") else sum(1 for _ in encoder.children())
    print(f"✅ Loaded {short_name} with {n_layers} layers")

    out_dir = os.path.join(SAVE_ROOT, short_name, "phonemes")
    drive_save_path = os.path.join(out_dir, f"{ds_key}.pkl")

    for f in glob.glob(os.path.join(TMP_ROOT, "layer_*.npy")):
        os.remove(f)

    print(f"\n🚀 MODEL: {short_name} × DATASET: {ds_key}")
    extract_batched(preproc, encoder, ds, drive_save_path)

    nemo_model.to("cpu")
    del nemo_model, preproc, encoder, ds
    torch.cuda.empty_cache(); gc.collect()

print("🎯 All pending extractions complete.")
