# Copyright (c) 2025 Jerônimo Augusto Soares
# Licensed under the BSD 3-Clause License. See LICENSE file in the project root for full license information.

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import random
from glob import glob

__all__ = ["AudioPreprocessor"]


class AudioPreprocessor(nn.Module):
    def __init__(
        self,
        device,
        noise_dir=None,
        sample_rate=16000,
        hop_length=160,
        win_length=480,
        n_fft=512,
        n_mels=40,
        specaug=False,
        frequency_masking_para=7,
        time_masking_para=20,
        frequency_mask_num=2,
        time_mask_num=2,
    ):
        """
        Handles GPU-accelerated audio preprocessing including:
        1. Waveform Augmentation (Background Noise Injection + Time Shift)
        2. Feature Extraction (Log-Mel Spectrogram)
        3. Frequency Domain Augmentation (SpecAugment)

        Args:
            device (torch.device): Compute device (CPU or CUDA).
            noise_dir (str, optional): Directory containing background noise .wav files.
            sample_rate (int): Target sample rate.
            hop_length (int): STFT hop length.
            win_length (int): STFT window length.
            n_fft (int): Number of FFT bins.
            n_mels (int): Number of Mel filterbanks.
            specaug (bool): Whether to apply SpecAugment during training.
            frequency_masking_para (int): Max frequency bands to mask.
            time_masking_para (int): Max time frames to mask.
            frequency_mask_num (int): Number of frequency masks to apply.
            time_mask_num (int): Number of time masks to apply.
        """
        super().__init__()
        self.device = device
        self.sample_rate = sample_rate
        self.specaug = specaug

        # --- SpecAugment Configuration ---
        self.frequency_masking_para = frequency_masking_para
        self.time_masking_para = time_masking_para
        self.frequency_mask_num = frequency_mask_num
        self.time_mask_num = time_mask_num

        # --- Background Noise Loading ---
        # Used for Signal-to-Noise (SNR) mixing.
        # Note: This is independent of the "Silence" class generation in the Dataset.
        self.background_noise = []
        if noise_dir:
            noise_files = glob(f"{noise_dir}/*.wav")
            if not noise_files:
                print(f"WARNING: No .wav files found in {noise_dir}")

            for file_name in noise_files:
                waveform, sr = torchaudio.load(file_name)
                # Ensure sample rate consistency
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)

                # Keep in CPU memory to save VRAM, move to GPU on-the-fly
                self.background_noise.append(waveform)

        # --- Feature Extractor (LogMel) ---
        self.feature_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            n_mels=n_mels,
        )

    def forward(self, x, labels=None, augment=False, noise_prob=0.8, is_train=True):
        """
        Args:
            x (Tensor): Batch of waveforms [Batch, 1, Time].
            labels (Tensor): Batch of labels [Batch] (used to determine noise amplitude).
            augment (bool): If True, applies noise and time shift.
            noise_prob (float): Probability of adding noise to speech samples.
            is_train (bool): If True, enables time shifting.

        Returns:
            Tensor: Log-Mel Spectrograms [Batch, 1, n_mels, Frames].
        """
        x = x.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        # 1. Time-Domain Augmentation (Waveform)
        if augment and self.background_noise:
            # Iterating over batch is required as each sample gets unique augmentation
            for idx in range(x.shape[0]):

                # A) Noise Injection Logic
                # - If Target Class (label != 0): Add noise with probability `noise_prob`.
                # - If Silence Class (label == 0): ALWAYS add noise (ignoring `noise_prob`).
                current_label = labels[idx].item() if labels is not None else -1

                if current_label != 0 and (
                    not is_train or random.random() > noise_prob
                ):
                    pass  # Skip noise injection
                else:
                    # Amplitude: Low (0.1) for speech, High (1.0) for silence to vary texture
                    noise_amp = (
                        np.random.uniform(0, 0.1)
                        if current_label != 0
                        else np.random.uniform(0, 1)
                    )

                    # Select random noise
                    noise_waveform = random.choice(self.background_noise).to(
                        self.device
                    )

                    # Random crop or tile to match input length
                    if noise_waveform.shape[-1] >= x.shape[-1]:
                        sample_loc = random.randint(
                            0, noise_waveform.shape[-1] - x.shape[-1]
                        )
                        noise_segment = noise_waveform[
                            :, sample_loc : sample_loc + x.shape[-1]
                        ]
                    else:
                        repeats = 1 + x.shape[-1] // noise_waveform.shape[-1]
                        noise_segment = noise_waveform.repeat(1, repeats)[
                            :, : x.shape[-1]
                        ]

                    # Mix: Original + (Noise * Amplitude)
                    x[idx] = x[idx] + (noise_amp * noise_segment)

                # B) Time Shift (Rolling)
                if is_train:
                    # Shift up to 10% of sample rate
                    shift_amt = int(np.random.uniform(-0.1, 0.1) * self.sample_rate)
                    if shift_amt != 0:
                        x[idx] = torch.roll(x[idx], shifts=shift_amt, dims=-1)
                        # Zero out the wrapped part to simulate linear shift instead of circular roll
                        if shift_amt > 0:
                            x[idx, :, :shift_amt] = 0
                        else:
                            x[idx, :, shift_amt:] = 0

                # Clip to valid audio range [-1, 1]
                x[idx] = torch.clamp(x[idx], -1.0, 1.0)

        # 2. Feature Extraction
        self.feature_extractor = self.feature_extractor.to(self.device)
        x = self.feature_extractor(x)

        # Logarithmic scale (add epsilon for numerical stability)
        x = (x + 1e-6).log()

        # 3. Frequency-Domain Augmentation (SpecAugment)
        if self.specaug and is_train:
            for i in range(x.shape[0]):
                x[i] = self.apply_spec_augment(
                    x[i],
                    self.frequency_masking_para,
                    self.time_masking_para,
                    self.frequency_mask_num,
                    self.time_mask_num,
                )

        return x

    @staticmethod
    def apply_spec_augment(
        x, freq_mask_para, time_mask_para, freq_mask_num, time_mask_num
    ):
        """Applies SpecAugment to a single spectrogram (not the entire batch)."""
        # x shape: [Channels, Freq, Time]
        c, lenF, lenT = x.shape

        # Frequency masking
        for _ in range(freq_mask_num):
            f = int(np.random.uniform(low=0.0, high=freq_mask_para))
            if f > lenF:
                continue
            f0 = random.randint(0, lenF - f)
            x[:, f0 : f0 + f, :] = 0

        # Time masking
        for _ in range(time_mask_num):
            t = int(np.random.uniform(low=0.0, high=time_mask_para))
            if t > lenT:
                continue
            t0 = random.randint(0, lenT - t)
            x[:, :, t0 : t0 + t] = 0

        return x
