# Copyright (c) 2025 Jerônimo Augusto Soares
# Licensed under the BSD 3-Clause License. See LICENSE file in the project root for full license information.

import os
import time
import random
import torch
import torchaudio
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import tarfile
import requests

__all__ = [
    "download_gsc_dataset",
    "create_gsc_dataframes",
    "SpeechCommandsDataset",
    "PreprocessingDataLoader",
]


def download_gsc_dataset(
    destination_dir, version=2, keep_archive=False, force_download=False
):
    """
    Downloads and extracts the Google Speech Commands dataset (v1 or v2).
    Uses a marker file to ensure extraction completion and avoid redundant downloads.

    Args:
        destination_dir (str): Root path where the dataset folder will be created.
        version (int): Dataset version (1 or 2). Default is 2.
        keep_archive (bool): If True, keeps the .tar.gz file after extraction.
                             If False, removes it. Default is False.
        force_download (bool): If True, ignores existing files/markers and forces
                               a fresh download and extraction. Default is False.

    Returns:
        str: The path to the extracted dataset directory.
    """

    if version == 1:
        url = "https://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
    elif version == 2:
        url = "https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    else:
        raise ValueError("Version must be 1 or 2.")

    filename = url.split("/")[-1]

    # Adjust destination to include the specific version subfolder
    extract_path = os.path.join(destination_dir, filename.split(".tar.gz")[0])

    # Sentinel file to mark successful extraction
    marker_file = os.path.join(extract_path, ".dataset_ready")
    tar_path = os.path.join(destination_dir, filename)

    # Check if the dataset is already fully extracted
    if not force_download and os.path.exists(marker_file):
        print(f"Dataset already fully extracted at {extract_path}. Skipping download.")
        return extract_path

    # If forced, remove the old marker to reset state
    if force_download and os.path.exists(marker_file):
        try:
            os.remove(marker_file)
        except OSError:
            pass

    os.makedirs(destination_dir, exist_ok=True)

    # Download if file is missing or forced
    if force_download or not os.path.exists(tar_path):
        print(f"Downloading Speech Commands v{version} to {destination_dir}...")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 * 1024  # 1MB

            with open(tar_path, "wb") as file, tqdm(
                desc=filename,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    bar.update(size)
        except Exception as e:
            print(f"Download failed: {e}")
            # Clean up partial file to prevent corruption issues on retry
            if os.path.exists(tar_path):
                os.remove(tar_path)
            raise e
    else:
        print(f"Archive {filename} already exists. Skipping download.")

    print(f"Extracting files to {extract_path}...")
    try:
        os.makedirs(extract_path, exist_ok=True)

        with tarfile.open(tar_path, "r:gz") as tar:
            # Security fix for Python 3.12+ regarding 'data' filter deprecation
            if hasattr(tarfile, "data_filter"):
                tar.extractall(path=extract_path, filter="data")
            else:
                tar.extractall(path=extract_path)

    except tarfile.ReadError:
        print("Error: The archive is corrupted. Deleting it.")
        if os.path.exists(tar_path):
            os.remove(tar_path)
        raise RuntimeError(
            "Dataset archive was corrupted. Please rerun with force_download=True."
        )
    except Exception as e:
        print(f"Extraction failed: {e}")
        raise e

    if not keep_archive:
        print("Cleaning up: Removing compressed file...")
        if os.path.exists(tar_path):
            os.remove(tar_path)
    else:
        print(f"Keeping compressed file at {tar_path}")

    # Create marker file to indicate success
    with open(marker_file, "w") as f:
        f.write("Download and extraction completed successfully.")

    print("Download and extraction complete.")

    return extract_path


def create_gsc_dataframes(root_dir):
    """
    Generates Train, Validation, Test, and Noise DataFrames based on the
    Google Speech Commands Dataset structure (v1 or v2) and official split lists.

    Args:
        root_dir (str): Path to the dataset root folder (e.g., './data/speech_commands_v0.02').

    Returns:
        tuple: (train_df, valid_df, test_df, noise_df)
    """

    # Load official split lists provided with the dataset.
    val_list_path = os.path.join(root_dir, "validation_list.txt")
    test_list_path = os.path.join(root_dir, "testing_list.txt")

    valid_files = set()
    if os.path.exists(val_list_path):
        with open(val_list_path, "r") as f:
            valid_files = set(x.strip().replace("\\", "/") for x in f.readlines())

    test_files = set()
    if os.path.exists(test_list_path):
        with open(test_list_path, "r") as f:
            test_files = set(x.strip().replace("\\", "/") for x in f.readlines())

    data_train = []
    data_valid = []
    data_test = []
    data_noise = []

    file_id_counter = 0

    print(f"Scanning dataset at: {root_dir} ...")

    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # Special case: Background Noise
        if folder_name == "_background_noise_":
            for filename in os.listdir(folder_path):
                if filename.endswith(".wav"):
                    # file_id < 0 indicates noise in the Dataset class
                    data_noise.append(
                        {
                            "file_name": filename,
                            "label": "_background_noise_",
                            "file_id": -1,
                        }
                    )
            continue

        # Standard case: Command words (up, down, sheila, left, etc.)
        for filename in os.listdir(folder_path):
            if not filename.endswith(".wav"):
                continue

            # Relative path used in official lists (e.g., "bed/00176480_nohash_0.wav")
            relative_path = f"{folder_name}/{filename}"

            entry = {
                "file_name": relative_path,
                "label": folder_name,
                "file_id": file_id_counter,
            }
            file_id_counter += 1

            # Split logic consistent with the official GSC paper/implementation
            if relative_path in valid_files:
                data_valid.append(entry)
            elif relative_path in test_files:
                data_test.append(entry)
            else:
                # If not in validation or test list, it defaults to training.
                data_train.append(entry)

    train_df = pd.DataFrame(data_train)
    valid_df = pd.DataFrame(data_valid)
    test_df = pd.DataFrame(data_test)
    noise_df = pd.DataFrame(data_noise)

    return train_df, valid_df, test_df, noise_df


class SpeechCommandsDataset(Dataset):
    def __init__(
        self,
        metadata_df,
        target_labels,
        audio_dir,
        noise_dir=None,
        mode="train",
        cache_ram=True,
        sr=16000,
        audio_len_samples=16000,
        limit_unknown=None,
        limit_silence=None,
        random_state=None,
    ):
        """
        Custom Dataset for Google Speech Commands with support for specific target
        classes, "Unknown" class balancing, and dynamic "Silence" generation.

        Args:
            metadata_df (pd.DataFrame or str): DataFrame or path to CSV containing columns
                                               ['file_name', 'label', 'file_id'].
                                               file_id < 0 indicates background noise.
            target_labels (list): List of specific target classes (e.g., ['up', 'down']).
            audio_dir (str): Root directory containing the audio files.
            noise_dir (str): Directory containing background noise files.
                             Defaults to audio_dir/_background_noise_ if None.
            mode (str): Sampling mode ('train', 'valid', 'test').
            cache_ram (bool): If True, loads all speech samples into RAM.
            sr (int): Desired sample rate (default: 16000).
            audio_len_samples (int): Fixed audio length in samples (default: 1s = 16000).
            limit_unknown (int): Forced number of Unknown samples (None = automatic).
            limit_silence (int): Forced number of Silence samples (None = automatic).
            random_state (int): Seed for reproducibility of file selection/splitting.
        """
        self.audio_dir = audio_dir
        self.noise_dir = (
            noise_dir if noise_dir else os.path.join(audio_dir, "_background_noise_")
        )
        self.mode = mode
        self.cache_ram = cache_ram
        self.sr = sr
        self.audio_len = audio_len_samples
        self.random_state = random_state

        # Local RNG to ensure reproducible splits without affecting global training augmentation
        if random_state is None:
            random_state = int(time.time())

        self.rng = random.Random(random_state)

        # --- Label Configuration ---
        self.target_labels = sorted(target_labels)
        self.label_to_idx = {label: i + 2 for i, label in enumerate(self.target_labels)}

        # Reserved indices: 0 for Silence, 1 for Unknown (Standard GSC/BCResNet convention)
        self.label_to_idx["_silence_"] = 0
        self.label_to_idx["_unknown_"] = 1

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        # --- Metadata Processing ---
        df = (
            metadata_df
            if isinstance(metadata_df, pd.DataFrame)
            else pd.read_csv(metadata_df)
        )

        # Separate words and noise based on file_id convention (file_id < 0 is noise)
        if "file_id" in df.columns:
            df_noise = df[df["file_id"] < 0].copy()
            df_words = df[df["file_id"] >= 0].copy()
        else:
            # Fallback: try to infer noise if file_id column is missing
            df_noise = pd.DataFrame()
            df_words = df.copy()

        # Filter Targets and Unknowns
        df_targets = df_words[df_words["label"].isin(self.target_labels)]
        df_unknowns = df_words[~df_words["label"].isin(self.target_labels)]

        self.samples = []

        # --- Sample List Construction (Balancing) ---

        # A) Targets (Add all)
        for _, row in df_targets.iterrows():
            self.samples.append((row["file_name"], self.label_to_idx[row["label"]]))

        # B) Unknowns (Sampling)
        if limit_unknown is not None:
            n_unknown = limit_unknown
        else:
            # Default: Match average size of target classes or roughly 10% of total
            n_unknown = (
                len(df_targets) // len(self.target_labels)
                if len(self.target_labels) > 0
                else 0
            )

        unknown_files = df_unknowns["file_name"].tolist()
        if len(unknown_files) > n_unknown:
            selected_unknowns = self.rng.sample(unknown_files, n_unknown)
        else:
            selected_unknowns = unknown_files

        for fname in selected_unknowns:
            self.samples.append((fname, self.label_to_idx["_unknown_"]))

        # C) Silence (Quantity Definition)
        # Silence is not added to self.samples list; it is generated dynamically in __getitem__
        if limit_silence is not None:
            self.num_silence = limit_silence
        else:
            self.num_silence = n_unknown  # Keep balanced with Unknowns

        # --- RAM Caching ---
        self.cached_speech = None
        self.cached_noise = []

        if self.cache_ram:
            print(
                f"[{mode.upper()}] Loading {len(self.samples)} speech samples into RAM..."
            )
            self.cached_speech = []

            for fname, label_idx in tqdm(self.samples, desc="Load Speech"):
                file_path = os.path.join(self.audio_dir, fname)
                waveform = self._load_and_fix_waveform(file_path, fixed_length=True)
                self.cached_speech.append((waveform, label_idx))

            print(f"[{mode.upper()}] Loading background noise...")
            # Load noise from folder if not present in DataFrame
            if df_noise.empty and os.path.isdir(self.noise_dir):
                noise_files = [
                    f for f in os.listdir(self.noise_dir) if f.endswith(".wav")
                ]
            else:
                noise_files = df_noise["file_name"].tolist()

            for nf in noise_files:
                n_path = os.path.join(self.noise_dir, nf)
                if os.path.exists(n_path):
                    # IMPORTANT: Load FULL noise (fixed_length=False) to allow random cropping later
                    waveform = self._load_and_fix_waveform(n_path, fixed_length=False)
                    self.cached_noise.append(waveform)

            if not self.cached_noise:
                print("WARNING: No noise files found. Using digital silence (zeros).")
                self.cached_noise.append(torch.zeros(1, self.audio_len))

    def _load_and_fix_waveform(self, file_path, fixed_length=True):
        """
        Loads audio and adjusts length ONLY if fixed_length=True.
        Noise files are kept in original length for random cropping.
        """
        waveform, sample_rate = torchaudio.load(file_path)

        if sample_rate != self.sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sr)
            waveform = resampler(waveform)

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if fixed_length:
            # Pad or Crop to ensure exact self.audio_len
            if waveform.shape[1] < self.audio_len:
                diff = self.audio_len - waveform.shape[1]
                waveform = F.pad(waveform, (0, diff))
            elif waveform.shape[1] > self.audio_len:
                # Center/Start crop for standardization during caching
                # (In real training, random crop is handled by augmentation, here we ensure consistent size)
                waveform = waveform[:, : self.audio_len]

        return waveform

    def __len__(self):
        """Returns the total number of samples (Speech + Silence)."""
        return len(self.samples) + self.num_silence

    def __getitem__(self, idx):
        """
        Retrieves a sample. If index is out of bounds of speech samples,
        generates a Silence sample dynamically.
        """
        # A) Speech Samples (Targets and Unknown)
        if idx < len(self.samples):
            if self.cache_ram:
                waveform, label = self.cached_speech[idx]
                # Return clone to avoid reference modification
                return waveform.clone(), label
            else:
                # On-the-fly loading
                fname, label = self.samples[idx]
                file_path = os.path.join(self.audio_dir, fname)
                waveform = self._load_and_fix_waveform(file_path, fixed_length=True)
                return waveform, label

        # B) Silence Sample (Dynamically Generated)
        else:
            label = self.label_to_idx["_silence_"]

            if self.cache_ram and self.cached_noise:
                noise = random.choice(self.cached_noise)
            else:
                # Fallback / On-the-fly logic placeholder
                noise = torch.zeros(1, self.audio_len)

            # Random Crop of 1 second from long noise file
            if noise.shape[1] > self.audio_len:
                max_start = noise.shape[1] - self.audio_len
                start = random.randint(0, max_start)
                waveform = noise[:, start : start + self.audio_len]
            else:
                # Padding if noise is too short
                diff = self.audio_len - noise.shape[1]
                waveform = F.pad(noise, (0, diff))

            return waveform.clone(), label

    def get_class_map(self):
        """Returns the dictionary mapping class names to indices."""
        return self.label_to_idx

    def set_random_state(self, random_state):
        """Updates the local random state for reproducible splits."""
        self.random_state = random_state
        self.rng = random.Random(random_state)


class PreprocessingDataLoader:
    """
    Wraps a standard DataLoader to apply preprocessing on the target device (GPU/CPU)
    transparently during the training loop.
    """

    def __init__(self, loader, preprocessor, is_train=True):
        """
        Args:
            loader (DataLoader): The standard PyTorch DataLoader.
            preprocessor (nn.Module): The AudioPreprocessor instance (already on device).
            is_train (bool): Flag to enable/disable augmentation logic in the preprocessor.
        """
        self.loader = loader
        self.dataset = loader.dataset
        self.preprocessor = preprocessor
        self.is_train = is_train

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for inputs, labels in self.loader:
            # Ensure tensors are on the same device as the preprocessor (e.g., GPU)
            inputs = inputs.to(self.preprocessor.device)
            labels = labels.to(self.preprocessor.device)

            # Apply transformation (Augmentation + LogMel)
            # The preprocessor handles noise logic internally based on the is_train flag
            processed_inputs = self.preprocessor(
                inputs,
                labels,
                augment=self.is_train,  # Enable augmentation only for training
                is_train=self.is_train,
            )

            yield processed_inputs, labels
