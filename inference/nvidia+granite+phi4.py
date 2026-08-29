from google.colab import drive
drive.mount('/content/drive')


# Core PyTorch and audio libraries
!pip install torch torchvision torchaudio --quiet


# Datasets and cache management
!pip install datasets fsspec==2023.9.2 --quiet


# Uninstall old transformers and install latest main branch (critical for granite_speech model)
!pip uninstall transformers -y
!pip install https://github.com/huggingface/transformers/archive/main.zip --quiet


# HuggingFace hub and transfer utils
!pip install huggingface_hub hf_transfer --quiet


# Additional utilities
!pip install backoff peft soundfile --quiet


# NeMo toolkit for Nvidia speech models
!pip install nemo_toolkit nemo_toolkit[all] hydra-core omegaconf pytorch-lightning --quiet






# ==== Put env vars BEFORE importing HF libraries ====
import os
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"          # metadata check
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"     # file downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"     # speed-up if hf_transfer is installed


# If you want persistent Drive cache, uncomment:
# os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_home"
# os.environ["HF_HUB_CACHE"] = "/content/drive/MyDrive/hf_home/hub"


import io
import gc
import pickle
import warnings
import time
import numpy as np
import torch
import torchaudio
from datasets import load_dataset, Audio, DownloadConfig
from requests.exceptions import ReadTimeout, ConnectionError as ReqConnError


try:
    from huggingface_hub.utils import HfHubHTTPError
except Exception:
    class HfHubHTTPError(Exception):
        pass


warnings.filterwarnings("ignore", category=UserWarning)
torch.set_grad_enabled(False)  # inference only


# == Save root ==
SAVE_ROOT = "/content/drive/MyDrive/Layer Representations"
os.makedirs(SAVE_ROOT, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# == Model type mappings ==
NEMO_MODELS = {
    "nvidia/parakeet-tdt-0.6b-v2": "ASRModel",
    "nvidia/canary-1b-flash": "EncDecMultiTaskModel",
    "nvidia/canary-1b": "EncDecMultiTaskModel",
    "nvidia/canary-qwen-2.5b": "SALM",   # Special SALM loader
}


TRANSFORMERS_AUDIO_LLM = {
    "ibm-granite/granite-speech-3.3-2b": True,
    "microsoft/Phi-4-multimodal-instruct": True,
}


continuous_features = [
    "duration","f0_mean","f0_median","f0_min","f0_max",
    "intensity_mean","intensity_median","intensity_min","intensity_max",
    "F1_mean","F1_median","F1_min","F1_max",
    "F2_mean","F2_median","F2_min","F2_max",
    "F3_mean","F3_median","F3_min","F3_max",
    "F3_minus_F2_mean","F3_minus_F2_median","F3_minus_F2_min","F3_minus_F2_max"
]
categorical_features = ["gender", "l1_background"]


def make_folder(p):
    os.makedirs(p, exist_ok=True)


def clean_gender(example):
    return {'gender': example['gender'] if example['gender'] is not None else None}


def load_audio_16k(example):
    audio_bytes = example["audio"].get("bytes")
    if not audio_bytes:
        return None
    wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    return wav.squeeze(0)


# -------- NeMo extractor (unchanged) --------
def extract_with_nemo(model, encoder, dataset, features_to_use):
    blocks = []
    if hasattr(encoder, "layers") and hasattr(encoder.layers, "__iter__"):
        for idx, m in enumerate(encoder.layers):
            blocks.append((f"layer_{idx}", m))
    else:
        for idx, m in enumerate(encoder.children()):
            blocks.append((f"layer_{idx}", m))
    if not blocks:
        raise RuntimeError("No encoder blocks found to hook.")
    print(f"    ↳ Hooking {len(blocks)} encoder blocks; pooling mean over time *inside* the hook.")
    pooled_by_layer = [[] for _ in range(len(blocks))]
    handles = []
    def make_hook(layer_idx):
        def hook(module, inp, out):
            x = out[0] if isinstance(out, (tuple, list)) else out  # (B,T,C) or (T,B,C)
            if x.dim() == 3 and x.shape[0] > 4 and x.shape[1] < 8:  # likely (T,B,C)
                x = x.transpose(0, 1)  # -> (B,T,C)
            if x.dim() == 3:
                v = x.mean(dim=1)  # (B,C)
            else:
                v = x
            v = v.squeeze(0).detach().float().cpu().numpy()
            pooled_by_layer[layer_idx].append(v)
        return hook
    for i, (_, m) in enumerate(blocks):
        handles.append(m.register_forward_hook(make_hook(i)))
    labels_dict = {feat: [] for feat in features_to_use}
    try:
        with torch.no_grad():
            for i, ex in enumerate(dataset):
                if (i + 1) % 50 == 0 or i == 0:
                    print(f"  NeMo example {i+1}/{len(dataset)}")
                wav = load_audio_16k(ex)
                if wav is None:
                    continue
                sig = wav.unsqueeze(0).to(DEVICE)
                length = torch.tensor([sig.shape[-1]], device=DEVICE, dtype=torch.long)


                # Conditional preprocessor usage:
                if hasattr(model, "perception") and hasattr(model.perception, "preprocessor"):
                    # For SALM model "canary-qwen-2.5b"
                    feats, feat_lens = model.perception.preprocessor(input_signal=sig, length=length)
                else:
                    # For other NeMo models (EncDecMultiTaskModel, ASRModel)
                    feats, feat_lens = model.preprocessor(input_signal=sig, length=length)


                _ = encoder(audio_signal=feats, length=feat_lens)


                for feat in features_to_use:
                    labels_dict[feat].append(ex.get(feat))
                del sig, length, feats, feat_lens
                torch.cuda.empty_cache()
                gc.collect()
    finally:
        for h in handles:
            h.remove()
    if sum(len(v) for v in pooled_by_layer) == 0:
        raise RuntimeError("No examples after filtering; nothing to save.")
    reps_by_layer = [np.stack(vecs) for vecs in pooled_by_layer]
    return reps_by_layer, labels_dict


# -------- Granite Speech encoder-only extractor --------
def extract_granite_audio_encoder(model, processor, dataset, features_to_use):
    model.eval()
    n_layers = len(model.encoder.layers)
    print(f"    ↳ Granite encoder has {n_layers} layers.")


    outputs_by_layer = [[] for _ in range(n_layers)]
    labels_dict = {feat: [] for feat in features_to_use}


    system_prompt = (
        "Knowledge Cutoff Date: April 2024.\n"
        "Today's Date: April 9, 2025.\n"
        "You are Granite, developed by IBM. You are a helpful AI assistant"
    )


    for i, ex in enumerate(dataset):
        if i % 50 == 0 or i == 0:
            print(f"    Granite example {i+1}/{len(dataset)}")


        wav = load_audio_16k(ex)
        if wav is None:
            continue


        # Be tolerant to datasets lacking a 'text' column
        user_txt = ex.get("text") or ""
        if user_txt == "":
            print(f"⚠️ Warning: text missing for example {i+1}")
        user_prompt = f"<|audio|>{user_txt}"


        # Prepare chat-like input for Granite
        chat = [
            dict(role="system", content=system_prompt),
            dict(role="user", content=user_prompt),
        ]
        prompt = processor.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )


        # Build inputs: real prompt + audio
        model_inputs = processor(prompt, wav.cpu().numpy(), return_tensors="pt")
        model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}


        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True)


        hidden_states = outputs.hidden_states  # tuple: [embedding, layer1, layer2, ...]
        for layer_idx in range(n_layers):
            layer_hidden_state = hidden_states[layer_idx + 1]  # skip embedding
            pooled = layer_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            outputs_by_layer[layer_idx].append(pooled)


        for feat in features_to_use:
            labels_dict[feat].append(ex.get(feat))


        torch.cuda.empty_cache()
        gc.collect()


    reps_by_layer = [np.stack(layer_out) for layer_out in outputs_by_layer]
    return reps_by_layer, labels_dict




# -------- Transformers extractor (for non-Granite audio LLMs) --------
def extract_with_transformers_audio_llm(model, processor, dataset, features_to_use):
    n_layers = getattr(model.config, "num_hidden_layers", 32)
    print(f"    ↳ Model has {n_layers} encoder layers; extracting all encoder layers.")
    hidden_collector = [[] for _ in range(n_layers)]
    labels_dict = {feat: [] for feat in features_to_use}


    with torch.no_grad():
        for i, ex in enumerate(dataset):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"   HF audio-LLM example {i+1}/{len(dataset)}")
            wav = load_audio_16k(ex)
            if wav is None:
                continue


            prompt = "<|user|><|audio_1|>Transcribe the audio.<|end|><|assistant|>"
            inputs = processor(
                text=prompt,
                audios=[(wav.cpu().numpy(), 16000)],
                return_tensors="pt"
            ).to(DEVICE)


            out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states


            for li in range(n_layers):
                vec = hs[li + 1].mean(dim=1).squeeze(0).float().cpu().numpy()
                hidden_collector[li].append(vec)


            for feat in features_to_use:
                labels_dict[feat].append(ex.get(feat))


            del inputs, out, hs
            torch.cuda.empty_cache()
            gc.collect()


    reps_by_layer = [np.stack(layer_list) for layer_list in hidden_collector]
    return reps_by_layer, labels_dict




# -------- Driver --------
def run_one(model_obj, processor_or_encoder, model_name, dataset_name, is_nemo):
    model_tag = model_name.split("/")[-1]
    dataset_key = dataset_name.split("/")[-1]
    model_dir = os.path.join(SAVE_ROOT, model_tag); make_folder(model_dir)
    save_path = os.path.join(model_dir, f"{dataset_key}.pkl")


    if os.path.exists(save_path):
        print(f"✅ Already exists: {save_path} (skipping)")
        return


    print(f"\n📊 Dataset: {dataset_name}")
    dcfg = DownloadConfig(max_retries=10)
    ds = load_dataset(dataset_name, split="train", download_config=dcfg)
    ds = ds.cast_column("audio", Audio(decode=False))


    features_to_use = continuous_features if dataset_key == "sandi" else (continuous_features + categorical_features)


    if 'gender' in ds.column_names:
        ds = ds.map(clean_gender)
        ds = ds.filter(lambda x: x['gender'] is not None)


    if is_nemo:
        reps_by_layer, labels = extract_with_nemo(model_obj, processor_or_encoder, ds, features_to_use)
    else:
        if model_name == "ibm-granite/granite-speech-3.3-2b":
            reps_by_layer, labels = extract_granite_audio_encoder(model_obj, processor_or_encoder, ds, features_to_use)
        else:
            reps_by_layer, labels = extract_with_transformers_audio_llm(model_obj, processor_or_encoder, ds, features_to_use)


    with open(save_path, "wb") as f:
        pickle.dump({"reps_by_layer": reps_by_layer, "labels": labels}, f)
    print(f"✅ Saved: {save_path}")


    del ds, reps_by_layer, labels
    torch.cuda.empty_cache()
    gc.collect()


# -------- Main --------
if __name__ == "__main__":
    model_list = [
        "microsoft/Phi-4-multimodal-instruct"
    ]
    dataset_names = [
        "PranavBhalerao/ALLSSTAR_2",
        "PranavBhalerao/cmu-arctic-train"
    ]
    # Ensure offload folder exists for accelerate
    os.makedirs("/content/accelerate_offload", exist_ok=True)


    for mn in model_list:
        all_done = all(
            os.path.exists(os.path.join(SAVE_ROOT, mn.split("/")[-1], f"{dn.split('/')[-1]}.pkl"))
            for dn in dataset_names
        )
        if all_done:
            print(f"✅ All datasets already processed for {mn}, skipping model load.")
            continue


        print(f"\n🔁 Loading model: {mn}")


        if mn in NEMO_MODELS:
            # Only NeMo models here (Granite removed from this mapping)
            if NEMO_MODELS[mn] == "EncDecMultiTaskModel":
                from nemo.collections.asr.models import EncDecMultiTaskModel as NemoModel
                model_obj = NemoModel.from_pretrained(model_name=mn).to(DEVICE).eval()
                processor_or_encoder = model_obj.encoder
                is_nemo = True
            elif NEMO_MODELS[mn] == "ASRModel":
                from nemo.collections.asr.models import ASRModel as NemoModel
                model_obj = NemoModel.from_pretrained(model_name=mn).to(DEVICE).eval()
                processor_or_encoder = model_obj.encoder
                is_nemo = True
            elif NEMO_MODELS[mn] == "SALM":
                from nemo.collections.speechlm2.models import SALM
                model_obj = SALM.from_pretrained(mn).to(DEVICE).eval()
                processor_or_encoder = model_obj.perception.encoder  # SALM audio encoder
                is_nemo = True
            else:
                raise ValueError(f"Unknown NeMo model type for {mn}")


        elif mn == "ibm-granite/granite-speech-3.3-2b":
            # Load Granite via Transformers (seq2seq); offload to keep VRAM sane
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
            processor = AutoProcessor.from_pretrained(mn, trust_remote_code=True)
            model_obj = AutoModelForSpeechSeq2Seq.from_pretrained(
                mn,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder="/content/accelerate_offload",
                trust_remote_code=True
            ).eval()
            processor_or_encoder = processor
            is_nemo = False


        elif mn in TRANSFORMERS_AUDIO_LLM:
            # Other multimodal audio LLMs (e.g., Phi-4-multimodal)
            from transformers import AutoProcessor, AutoModelForCausalLM
            # Correct place to set use_fast=True (only for processor)
            processor = AutoProcessor.from_pretrained(mn, trust_remote_code=True, use_fast=True)
            model_obj = AutoModelForCausalLM.from_pretrained(
                mn,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                _attn_implementation="eager" # disable flash attn if missing
            ).eval()
            processor_or_encoder = processor
            is_nemo = False


        else:
            raise ValueError(f"Model {mn} not supported.")


        for dn in dataset_names:
            for attempt in range(3):
                try:
                    run_one(model_obj, processor_or_encoder, mn, dn, is_nemo)
                    break
                except (ReadTimeout, ReqConnError, HfHubHTTPError) as e:
                    wait = 5 * (attempt + 1)
                    print(f"⚠️ {type(e).__name__} on {dn}. Retrying in {wait}s ({attempt + 1}/3)...")
                    time.sleep(wait)
            else:
                print(f"❌ Skipping {dn} after 3 failed attempts.")


        print(f"🧹 Clearing model: {mn}")
        try:
            model_obj.to("cpu")
        except Exception:
            pass
        del model_obj, processor_or_encoder
        torch.cuda.empty_cache()
        gc.collect()
