"""
analysis_toolkit.py

This module provides a centralized toolkit for running ASR model analysis,
including model loading, representation extraction, and linear probing.
"""

import os
import gc
import torch
import torchaudio
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Audio
import librosa

# Scikit-learn imports
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Transformers and other model libraries
from transformers import WhisperModel, WhisperProcessor, Wav2Vec2ConformerModel, Wav2Vec2Processor # Add more as needed

# ================================
# Part 1: Model Loading
# ================================

def load_model_and_processor(model_name, device):
    """
    Loads a model and its corresponding processor from Hugging Face.
    """
    print(f"Loading model: {model_name}...")
    if "whisper" in model_name.lower():
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperModel.from_pretrained(model_name).to(device).eval()
    elif "wav2vec2-conformer" in model_name.lower():
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ConformerModel.from_pretrained(model_name).to(device).eval()
    elif "phi" in model_name.lower():
        # Phi-4 is a multimodal model that requires trust_remote_code
        from transformers import AutoModelForCausalLM, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
        ).to(device).eval()
    else:
        raise ValueError(f"Model family for {model_name} is not supported yet.")

    return model, processor

# ================================
# Part 2: Representation Extraction
# ================================

def extract_representations(model, processor, dataset, device):
    """
    Extracts hidden state representations from all layers of a model for a given dataset.
    Handles Hugging Face model architectures.
    Supports both regular and streaming datasets.
    """
    # For Hugging Face models, we get N+1 layers (N blocks + embedding layer)
    num_layers = model.config.num_hidden_layers + 1

    reps_by_layer = [[] for _ in range(num_layers)]

    # Handle both regular and streaming datasets for column names
    try:
        column_names = dataset.column_names
    except (AttributeError, TypeError):
        # For streaming datasets, we'll get column names from the first example
        column_names = None

    labels_dict = {}

    print(f"Extracting representations from dataset...")
    example_count = 0

    # Convert dataset to list to avoid iterator issues that cause crashes
    # This loads all examples into memory but avoids the crash during iteration
    print("Loading dataset examples into memory...")
    try:
        dataset_list = list(dataset)
        print(f"Loaded {len(dataset_list)} examples into memory")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return [], {}

    for example in tqdm(dataset_list, desc="Extracting Reps"):
        try:
            if not example.get("audio"):
                continue

            example_count += 1
            if example_count % 10 == 1:
                print(f"Processing example {example_count}...")

            # Manually decode audio to bypass datasets Audio feature (which causes crashes)
            try:
                audio_data = example["audio"]

                if example_count == 1:
                    print(f"  Debug: audio_data type: {type(audio_data)}")
                    if isinstance(audio_data, dict):
                        print(f"  Debug: audio_data keys: {audio_data.keys()}")

                # Handle different audio formats from HuggingFace datasets
                if isinstance(audio_data, dict):
                    # If already decoded by datasets library
                    if "array" in audio_data:
                        audio_array = audio_data["array"]
                        sampling_rate = audio_data.get("sampling_rate", 16000)
                    # If it has bytes (from parquet storage)
                    elif "bytes" in audio_data:
                        import io
                        import soundfile as sf
                        audio_bytes = audio_data["bytes"]
                        target_sr = audio_data.get("sampling_rate", 16000)
                        # Decode from bytes
                        audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
                        # Resample if needed
                        if sampling_rate != target_sr:
                            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=target_sr)
                            sampling_rate = target_sr
                    # If it's a path (shouldn't happen with parquet, but handle it)
                    elif "path" in audio_data:
                        # Try to use datasets library's internal path resolution
                        # For now, skip files with paths as they need special handling
                        raise ValueError(f"Audio path found but not accessible: {audio_data.get('path')}")
                    else:
                        raise ValueError(f"Unknown audio dict format: {audio_data.keys()}")
                elif isinstance(audio_data, bytes):
                    # Audio stored directly as bytes
                    import io
                    import soundfile as sf
                    audio_array, sampling_rate = sf.read(io.BytesIO(audio_data))
                    # Resample to 16kHz if needed
                    if sampling_rate != 16000:
                        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
                        sampling_rate = 16000
                elif isinstance(audio_data, (list, np.ndarray)):
                    # Already decoded array
                    audio_array = np.array(audio_data, dtype=np.float32)
                    sampling_rate = 16000  # Default assumption
                elif isinstance(audio_data, str):
                    # If it's just a path string (unlikely with parquet)
                    raise ValueError(f"Audio path string not supported: {audio_data}")
                else:
                    raise ValueError(f"Unknown audio format: {type(audio_data)}")

                # Convert to numpy if needed
                if not isinstance(audio_array, np.ndarray):
                    audio_array = np.array(audio_array, dtype=np.float32)

                # Ensure it's float32
                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)

                # Create tensor on CPU first, then move to device
                waveform = torch.tensor(audio_array, dtype=torch.float32)
                waveform = waveform.to(device)

                if example_count % 10 == 1:
                    print(f"  Waveform shape: {waveform.shape}, dtype: {waveform.dtype}, device: {waveform.device}, sr: {sampling_rate}")
            except Exception as e:
                print(f"Error creating waveform tensor for example {example_count}: {e}")
                import traceback
                traceback.print_exc()
                continue

            hidden_states = []

            # --- Hugging Face Model Logic ---
            try:
                # Handle Whisper-like processors with feature_extractor
                if hasattr(processor, 'feature_extractor') and hasattr(processor.feature_extractor, 'sampling_rate'):
                    target_sampling_rate = processor.feature_extractor.sampling_rate
                    if sampling_rate != target_sampling_rate:
                        # Resample on CPU to avoid CUDA kernel issues
                        waveform_cpu = waveform.cpu()
                        resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
                        waveform = resampler(waveform_cpu).to(device)

                    inputs = processor(waveform.squeeze().cpu().numpy(), sampling_rate=target_sampling_rate, return_tensors="pt")
                    input_features = inputs.input_features.to(device)

                # Handle other processors (like for Phi-4)
                else:
                    # Assume the processor can handle the raw waveform
                    # This may need adjustment depending on the model's requirements
                    inputs = processor(audios=[waveform.squeeze().cpu().numpy()], sampling_rate=sampling_rate, return_tensors="pt")
                    input_features = inputs.get("input_features") or inputs.get("pixel_values") # Handle different input names
                    if input_features is not None:
                        input_features = input_features.to(device)
                    else:
                        raise ValueError("Could not find input_features or pixel_values in processor output")

                with torch.no_grad():
                    # Use a generic 'forward' or 'encoder' call
                    if hasattr(model, 'encoder'):
                        outputs = model.encoder(input_features, output_hidden_states=True)
                    else:
                        outputs = model(input_features, output_hidden_states=True)
                hidden_states = outputs.hidden_states
            except Exception as e:
                print(f"Error in model forward pass for example {example_count}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # --- Common Logic for Appending Representations ---
            if len(reps_by_layer) != len(hidden_states):
                print(f"[Warning] Mismatch between expected layers ({len(reps_by_layer)}) and extracted states ({len(hidden_states)}). Adjusting.")
                # Adjust reps_by_layer if there's a mismatch
                if len(reps_by_layer) > len(hidden_states):
                    reps_by_layer = reps_by_layer[:len(hidden_states)]
                else: # This case is less likely
                    hidden_states = hidden_states[:len(reps_by_layer)]

            for layer_idx, hs in enumerate(hidden_states):
                mean_rep = hs.squeeze(0).mean(dim=0).cpu().numpy()
                reps_by_layer[layer_idx].append(mean_rep)

            # Dynamically collect labels from example keys (excluding 'audio')
            if column_names is None:
                # Initialize labels_dict from first example if needed
                column_names = [k for k in example.keys() if k != 'audio']

            for feat in column_names:
                if feat != 'audio':
                    if feat not in labels_dict:
                        labels_dict[feat] = []
                    labels_dict[feat].append(example.get(feat))

            del waveform, hidden_states
            if 'input_features' in locals(): del input_features
            if 'encoder_outputs' in locals(): del encoder_outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Unexpected error processing example {example_count}: {e}")
            import traceback
            traceback.print_exc()
            continue

    reps_by_layer = [np.stack(layer) for layer in reps_by_layer if layer]
    return reps_by_layer, labels_dict

# ================================
# Part 3: Linear Probing
# ================================

def _stratified_split_robust(y, test_size=0.2, random_state=42):
    y = np.asarray(y)
    idx_all = np.arange(len(y))
    unique, counts = np.unique(y, return_counts=True)
    if np.min(counts) < 2: # Cannot stratify if a class has only one sample
        return train_test_split(idx_all, test_size=test_size, random_state=random_state)

    return train_test_split(idx_all, test_size=test_size, random_state=random_state, stratify=y)


def run_probing_analysis(reps_by_layer, labels, is_categorical_map):
    """
    Runs the full probing analysis for a set of representations and labels.
    """
    results = []

    for feat, y_raw in labels.items():
        if not is_categorical_map.get(feat, False): # Skip non-feature columns
            continue

        y = np.array(y_raw, dtype=object)
        is_categorical = is_categorical_map.get(feat, False)

        # Handle splitting
        if is_categorical:
            train_idx, test_idx = _stratified_split_robust(y)
        else:
            idx_all = np.arange(len(y))
            train_idx, test_idx = train_test_split(idx_all, test_size=0.2, random_state=42)

        y_train, y_test = y[train_idx], y[test_idx]

        for layer_idx, X_full in enumerate(tqdm(reps_by_layer, desc=f"Probing {feat}")):
            X = np.asarray(X_full)
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]

            # Scale and PCA
            scaler = StandardScaler().fit(X_train_raw)
            X_train = scaler.transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            pca = PCA(n_components=10, random_state=42)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            # Fit probe
            score = np.nan
            if is_categorical:
                if len(np.unique(y_train)) < 2: continue
                le = LabelEncoder().fit(y_train)
                y_train_enc = le.transform(y_train)
                y_test_enc = le.transform(y_test)

                probe = LogisticRegression(max_iter=1000, class_weight="balanced")
                probe.fit(X_train_pca, y_train_enc)
                score = accuracy_score(y_test_enc, probe.predict(X_test_pca))
            else:
                probe = LinearRegression()
                probe.fit(X_train_pca, y_train)
                score = r2_score(y_test, probe.predict(X_test_pca))

            results.append({
                "layer": layer_idx,
                "feature": feat,
                "score": score
            })

    return pd.DataFrame(results)
