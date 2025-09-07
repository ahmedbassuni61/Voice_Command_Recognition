"""
augment_audio.py

Creates augmented versions of 2-second WAV files stored in dataset/<class>/*.wav
Outputs saved to dataset_augmented/<class>/.

Dependencies:
    pip install librosa soundfile numpy
"""

import os
import random
import numpy as np
import librosa
import soundfile as sf

# -------- CONFIG --------
DATASET_PATH = "dataset"                  # input: dataset/on/, dataset/off/
OUT_PATH = "dataset_augmented"            # output
SR = 16000                                # sample rate
TARGET_LEN = 1 * SR                       # 2 seconds = 32000 samples
AUG_PER_FILE = 6                          # how many augmented files per original
RANDOM_SEED = 42
# ------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------- AUGMENTATION FUNCTIONS ----------

def add_noise(y, snr_db):
    """Add gaussian noise to achieve approximate SNR in dB."""
    rms = np.sqrt(np.mean(y**2))
    snr = 10 ** (snr_db / 20.0)
    noise_rms = rms / snr
    noise = np.random.normal(0, noise_rms, y.shape[0])
    return y + noise

def time_shift(y, shift_max=0.15):
    """Shift audio by up to shift_max seconds (positive = right shift)."""
    shift_amt = int(random.uniform(-shift_max, shift_max) * SR)
    if shift_amt > 0:
        y_shift = np.concatenate([np.zeros(shift_amt), y])[:len(y)]
    elif shift_amt < 0:
        y_shift = np.concatenate([y[-shift_amt:], np.zeros(-shift_amt)])
    else:
        y_shift = y
    return y_shift

def change_gain(y, db):
    """Apply gain in dB."""
    factor = 10 ** (db / 20.0)
    return y * factor

def pitch_shift(y, n_steps):
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps)

def time_stretch(y, rate):
    try:
        y_st = librosa.effects.time_stretch(y, rate)
    except Exception:
        new_len = int(len(y) / rate)
        y_st = librosa.resample(y, orig_sr=SR, target_sr=int(SR*rate))
    # enforce TARGET_LEN
    if len(y_st) > TARGET_LEN:
        y_st = y_st[:TARGET_LEN]
    elif len(y_st) < TARGET_LEN:
        y_st = np.pad(y_st, (0, TARGET_LEN - len(y_st)), mode='constant')
    return y_st

def augment_clip(y):
    """Return a list of augmented variants for clip y (length TARGET_LEN)."""
    variants = []

    # 1) small pitch shifts
    variants.append(pitch_shift(y, n_steps=random.uniform(-0.5, 0.5)))

    # 2) speed changes (time stretch)
    variants.append(time_stretch(y, rate=random.uniform(0.92, 1.08)))

    # 3) time shift
    variants.append(time_shift(y, shift_max=0.12))

    # 4) add noise with different SNRs
    variants.append(add_noise(y, snr_db=random.uniform(14, 20)))

    # 5) gain change
    variants.append(change_gain(y, db=random.uniform(-6, 6)))

    # 6) combination: small pitch + noise
    y_combo = pitch_shift(y, n_steps=random.uniform(-0.5, 0.5))
    y_combo = add_noise(y_combo, snr_db=random.uniform(14, 18))
    variants.append(y_combo)

    return variants

def ensure_mono_and_length(y, sr):
    """Convert to mono, resample, and pad/trim to 2 seconds (TARGET_LEN)."""
    if y.ndim > 1:
        y = librosa.to_mono(y)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) > TARGET_LEN:
        y = y[:TARGET_LEN]
    elif len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)), mode='constant')
    return y

# ---------- MAIN PROCESSING ----------

def process_dataset():
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH, exist_ok=True)

    for label in os.listdir(DATASET_PATH):
        class_dir = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(class_dir):
            continue

        out_class_dir = os.path.join(OUT_PATH, label)
        os.makedirs(out_class_dir, exist_ok=True)

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(".wav"):
                continue

            path_in = os.path.join(class_dir, fname)
            try:
                y, sr = librosa.load(path_in, sr=SR, mono=True)
            except Exception as e:
                print(f"⚠️ Could not load {path_in}: {e}")
                continue

            y = ensure_mono_and_length(y, sr)
            base = os.path.splitext(fname)[0]

            # save original
            out_orig = os.path.join(out_class_dir, f"{base}_orig.wav")
            sf.write(out_orig, y, SR)

            # augment
            aug_variants = augment_clip(y)
            chosen = aug_variants[:AUG_PER_FILE] if len(aug_variants) >= AUG_PER_FILE else aug_variants
            while len(chosen) < AUG_PER_FILE:
                y2 = time_shift(y, shift_max=0.12)
                y2 = pitch_shift(y2, n_steps=random.uniform(-1,1))
                y2 = add_noise(y2, snr_db=random.uniform(8,18))
                chosen.append(y2)

            for i, aug in enumerate(chosen, 1):
                if len(aug) > TARGET_LEN:
                    aug = aug[:TARGET_LEN]
                elif len(aug) < TARGET_LEN:
                    aug = np.pad(aug, (0, TARGET_LEN - len(aug)), mode='constant')
                out_file = os.path.join(out_class_dir, f"{base}_aug{i}.wav")
                sf.write(out_file, aug, SR)

            print(f"Saved {len(chosen)} augmentations for {path_in} -> {out_class_dir}")

if __name__ == "__main__":
    process_dataset()
