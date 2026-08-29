# =====================================================
# Canary-Qwen Complete Setup (All Dependencies)
# =====================================================
from google.colab import drive
drive.mount('/content/drive')

print("📦 Installing dependencies...")

# Install transformers FIRST (stable version)
!pip install -q transformers==4.48.2

# Install accelerate
!pip install -q accelerate==1.3.0

# CRITICAL: datasets[audio] enables torchcodec for decode=True
!pip install -U datasets[audio]

# Upgrade PyTorch AFTER datasets (this is the key!)
!pip install torch==2.9.0 torchvision torchaudio --upgrade

# Audio processing
!pip install -q soundfile==0.13.1
!pip install -q scipy==1.15.2

# Install Cython before NeMo
!pip install -q Cython packaging

# Install lhotse (required for Canary-Qwen)
!pip install -q lhotse

# Install NeMo
!pip install -q "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"

# NeMo ASR dependencies
!pip install -q sacrebleu whisper-normalizer librosa webdataset

# Additional utilities
!pip install -q huggingface_hub pyyaml

print("\n✅ Setup complete!")
print("⚠️ IMPORTANT: Restart runtime now (Runtime → Restart runtime)")





# =====================================================
# Canary-Qwen Layer Extractor - FIXED
# (No disk I/O, proper in-memory Lhotse handling)
# =====================================================

import io, os, gc, pickle, shutil, glob, warnings
import numpy as np
import torch, torchaudio, soundfile as sf
from datasets import load_dataset
from tqdm.auto import tqdm
import logging
from nemo.collections.speechlm2.models import SALM
from lhotse import Recording, AudioSource, CutSet
from lhotse.dataset import DynamicCutSampler
import tempfile

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
AUDIO_TMP = "/content/tmp_audio"  # Temp storage for batch audio

os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(LOCAL_TMP, exist_ok=True)
os.makedirs(AUDIO_TMP, exist_ok=True)

TARGET_SR = 16000
GC_EVERY = 1000
DEBUG_LIMIT = None
FLUSH_EVERY = 5000

dataset_names = [
    #"PranavBhalerao/cam_assess_phonemes",
    "PranavBhalerao/l2-arctic-dataset-250_phonemes",
    "PranavBhalerao/CommonVoice_accent_stratified_phonemes",
    "PranavBhalerao/cmu-arctic-train_phonemes",
    "PranavBhalerao/ALLSSTAR_2_phonemes",
    "PranavBhalerao/SAA_phonemes",
]

model_name = "nvidia/canary-qwen-2.5b"
INIT_BATCH_SIZE = 512

# =====================================================
# Helpers (same as before)
# =====================================================
def make_folder(path):
    os.makedirs(path, exist_ok=True)

def normalize_tag(model_tag: str) -> str:
    return model_tag

def masked_time_mean(hidden, attn_mask=None):
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
# Dataset with decode=True
# =====================================================
class StreamingAudioDataset:
    def __init__(self, dataset_name, split="train"):
        print(f"📥 Loading dataset {dataset_name} with decode=True...")
        self.ds = load_dataset(dataset_name, split=split)
        print(f"✅ Loaded {len(self.ds):,} samples")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]

        ex = self.ds[i]
        audio_array = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        phoneme = ex.get("phoneme", "unknown")

        audio_tensor = torch.from_numpy(audio_array).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        if audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(0, keepdim=True)

        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            audio_tensor = resampler(audio_tensor)

        return {
            "audio": audio_tensor.squeeze(0).numpy(),
            "phoneme": phoneme,
            "sr": TARGET_SR
        }

# =====================================================
# Hidden State Extractor
# =====================================================
class HiddenStateExtractor:
    def __init__(self, model):
        self.model = model
        self.all_hiddens = []
        self.hooks = []

    def _hook(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        if isinstance(hidden, torch.Tensor) and hidden.dim() == 3:
            self.all_hiddens.append(hidden.detach().cpu())

    def register_hooks(self):
        encoder_count = 0
        if hasattr(self.model, 'perception') and hasattr(self.model.perception, 'encoder'):
            encoder = self.model.perception.encoder
            if hasattr(encoder, 'layers'):
                for layer in encoder.layers:
                    h = layer.register_forward_hook(self._hook)
                    self.hooks.append(h)
                    encoder_count = len(encoder.layers)
            elif hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layers'):
                for layer in encoder.encoder.layers:
                    h = layer.register_forward_hook(self._hook)
                    self.hooks.append(h)
                    encoder_count = len(encoder.encoder.layers)

        decoder_count = 0
        if hasattr(self.model, 'llm'):
            llm = self.model.llm
            if hasattr(llm, 'model') and hasattr(llm.model, 'layers'):
                for layer in llm.model.layers:
                    h = layer.register_forward_hook(self._hook)
                    self.hooks.append(h)
                    decoder_count = len(llm.model.layers)

        print(f"✅ Registered {encoder_count} encoder + {decoder_count} decoder = {len(self.hooks)} hooks")
        return encoder_count, decoder_count

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def reset(self):
        self.all_hiddens.clear()

# =====================================================
# Batch Processing Helper
# =====================================================
def create_batch_cuts(batch_items, batch_idx):
    """
    Create Lhotse cuts for a batch, keeping temp files until batch is processed.
    Returns (cuts, temp_files_to_cleanup).
    """
    cuts = []
    temp_files = []

    for local_idx, item in enumerate(batch_items):
        audio = item["audio"]
        phoneme = item["phoneme"]
        sr = item["sr"]

        # Create temp file that persists for this batch
        tmp_path = os.path.join(AUDIO_TMP, f"batch_{batch_idx}_sample_{local_idx}.wav")
        sf.write(tmp_path, audio, sr)
        temp_files.append(tmp_path)

        # Create Lhotse Recording from temp file
        rec = Recording.from_file(tmp_path, recording_id=f"batch_{batch_idx}_sample_{local_idx}")
        cut = rec.resample(sr).to_cut()
        if cut.num_channels > 1:
            cut = cut.to_mono(mono_downmix=True)
        cut.custom = {"phoneme": phoneme}
        cuts.append(cut)

    return cuts, temp_files

# =====================================================
# Extraction
# =====================================================
def extract_and_save_one_pkl(model, ds, init_batch_size, final_save_path, debug_limit=None):
    from google.colab import drive

    device = next(model.parameters()).device
    model.eval()

    extractor = HiddenStateExtractor(model)
    encoder_layers, decoder_layers = extractor.register_hooks()

    audio_tag = model.audio_locator_tag
    print(f"🔊 Audio locator: {audio_tag}")

    num_layers = encoder_layers + decoder_layers + 1
    hidden_size = 1024
    if hasattr(model, 'llm') and hasattr(model.llm, 'config'):
        hidden_size = model.llm.config.hidden_size

    print(f"📊 Layers: {num_layers} (E:{encoder_layers} + D:{decoder_layers} + 1)")
    print(f"📊 Hidden: {hidden_size}")

    layer_tmp_paths = [os.path.join(LOCAL_TMP, f"layer_{i:02d}.npy") for i in range(num_layers)]
    for p in layer_tmp_paths:
        if os.path.exists(p): os.remove(p)

    labels = {"phoneme": []}
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

    N = len(ds)
    limit = N if debug_limit is None else min(N, debug_limit)
    progress = tqdm(total=limit, unit="samples", desc="Extracting", leave=True)
    cursor, total_seen, adaptive_bs = 0, 0, init_batch_size
    oom_count, since_last_grow = 0, 0
    grow_every_samples = max(FLUSH_EVERY // 2, 1000)

    def _gentle_shrink(bs):
        if bs > 16: return max(int(bs * 0.75), 4)
        if bs > 4: return max(int(bs * 0.80), 2)
        return max(bs - 1, 1)

    def _hard_backoff(bs):
        return max(int(bs * 0.5), 1)

    batch_counter = 0

    while cursor < limit:
        end = min(cursor + adaptive_bs, limit)
        temp_files_to_cleanup = []

        try:
            # Load batch from streaming dataset
            batch_items = [ds[i] for i in range(cursor, end)]
            batch_phonemes = [item["phoneme"] for item in batch_items]
            batch_size = len(batch_items)

            # Create Lhotse cuts with temp files
            cuts, temp_files_to_cleanup = create_batch_cuts(batch_items, batch_counter)
            batch_counter += 1

            # Use DynamicCutSampler for batching
            cutset = CutSet.from_cuts(cuts)
            sampler = DynamicCutSampler(cutset, max_cuts=batch_size)
            batch = next(iter(sampler))

            extractor.reset()

            # Load audio with Lhotse collation
            audio, audio_lens = batch.load_audio(collate=True)

            audio_tensor = torch.as_tensor(audio).to(device, non_blocking=True)
            audio_lens_tensor = torch.as_tensor(audio_lens).to(device, non_blocking=True)

            prompts = [[{"role": "user", "content": f"Transcribe the following: {audio_tag}"}]] * batch_size

            with torch.inference_mode():
                _ = model.generate(
                    prompts=prompts,
                    audios=audio_tensor,
                    audio_lens=audio_lens_tensor,
                    max_new_tokens=32,
                )

            oom_count = 0
            since_last_grow += batch_size

            # Collect hidden states
            if len(extractor.all_hiddens) > 0:
                for li in range(min(num_layers, len(extractor.all_hiddens))):
                    hs = extractor.all_hiddens[li]
                    if hs.dim() == 3 and hs.shape[0] == batch_size:
                        pooled = masked_time_mean(hs, None).float().numpy().astype(np.float16)
                        layer_buffers[li].append(pooled)

                labels["phoneme"].extend(batch_phonemes)
                total_seen += batch_size
                progress.update(batch_size)
                cursor = end

                if adaptive_bs < init_batch_size and since_last_grow >= grow_every_samples:
                    new_bs = min(int(adaptive_bs * 1.25), init_batch_size)
                    if new_bs != adaptive_bs:
                        adaptive_bs = new_bs
                        tqdm.write(f"📈 Recovering batch size → {adaptive_bs}")
                    since_last_grow = 0
            else:
                tqdm.write(f"⚠️ No hiddens captured, skipping batch")
                cursor = end

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

        finally:
            # Clean up temp files for this batch
            for tmp_file in temp_files_to_cleanup:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)

        if total_seen % FLUSH_EVERY == 0 or cursor >= limit:
            _flush_buffers_to_temp()
        if total_seen % GC_EVERY == 0:
            gc.collect()
            torch.cuda.empty_cache()

    progress.close()
    _flush_buffers_to_temp()
    extractor.remove_hooks()
    torch.cuda.empty_cache()
    gc.collect()

    # Clean up any remaining temp audio files
    for f in glob.glob(f"{AUDIO_TMP}/*.wav"):
        os.remove(f)

    # Drive remount and save (same as before)
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
# MAIN
# =====================================================
print(f"\n🖥️ GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

for dataset_name in tqdm(dataset_names, desc="Datasets", unit="ds"):
    dataset_key = dataset_name.split("/")[-1]
    model_tag = "canary-qwen-2.5b"
    short_tag = normalize_tag(model_tag)

    model_dir = os.path.join(SAVE_ROOT, short_tag)
    phoneme_dir = os.path.join(model_dir, "phonemes")
    make_folder(phoneme_dir)
    save_path = os.path.join(phoneme_dir, f"{dataset_key}.pkl")

    if os.path.exists(save_path):
        print(f"✅ All models already processed for {dataset_key} — skipping dataset.")
        continue

    ds = StreamingAudioDataset(dataset_name, split="train")

    print("\n" + "="*80)
    print(f"🚀 STARTING MODEL: {model_tag}  ×  DATASET: {dataset_key}")
    print("="*80)

    print(f"🔁 Loading {model_tag}...")
    model = SALM.from_pretrained(model_name).bfloat16().eval().to(DEVICE)
    print("✅ Model loaded in bfloat16")

    print(f"📊 Extracting features (BFloat16, micro-batched, adaptive)…")

    extract_and_save_one_pkl(
        model=model,
        ds=ds,
        init_batch_size=INIT_BATCH_SIZE,
        final_save_path=save_path,
        debug_limit=DEBUG_LIMIT,
    )

    del model
    torch.cuda.empty_cache()
    gc.collect()

    del ds
    gc.collect()
    torch.cuda.empty_cache()

print("\n🎯 Done. All datasets processed for all models.")
print("⚡ BFloat16 micro-batched inference with streaming dataset.")
