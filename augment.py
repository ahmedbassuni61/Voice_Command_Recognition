"""
augment_audio.py

Creates augmented versions of 1-second WAV files stored in dataset/<class>/*.wav
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
AUG_PER_FILE = 6                          # how many augmented files per original
RANDOM_SEED = 42
# ------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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
    # librosa.time_stretch expects at least length of window; handle short arrays
    try:
        y_st = librosa.effects.time_stretch(y, rate)
    except Exception:
        # fallback: resample to simulate small rate change
        new_len = int(len(y) / rate)
        y_st = librosa.resample(y, orig_sr=SR, target_sr=int(SR*rate))
        if len(y_st) > len(y):
            y_st = y_st[:len(y)]
        else:
            y_st = np.pad(y_st, (0, max(0, len(y)-len(y_st))), mode='constant')
    # ensure length exactly SR (1 sec)
    if len(y_st) > SR:
        y_st = y_st[:SR]
    elif len(y_st) < SR:
        y_st = np.pad(y_st, (0, SR - len(y_st)), mode='constant')
    return y_st

def augment_clip(y):
    """Return a list of augmented variants for clip y (length SR)."""
    variants = []

    # 1) small pitch shifts
    variants.append(pitch_shift(y, n_steps=random.uniform(-2, 2)))

    # 2) speed changes (time stretch)
    variants.append(time_stretch(y, rate=random.uniform(0.92, 1.08)))

    # 3) time shift
    variants.append(time_shift(y, shift_max=0.12))

    # 4) add noise with different SNRs
    variants.append(add_noise(y, snr_db=random.uniform(5, 20)))  # quieter noise (~5dB) to mild noise (20dB)

    # 5) gain change
    variants.append(change_gain(y, db=random.uniform(-6, 6)))

    # 6) combination: small pitch + noise
    y_combo = pitch_shift(y, n_steps=random.uniform(-1, 1))
    y_combo = add_noise(y_combo, snr_db=random.uniform(8, 18))
    variants.append(y_combo)

    return variants

def ensure_mono_and_length(y, sr):
    # convert to mono if stereo
    if y.ndim > 1:
        y = librosa.to_mono(y)
    # resample if needed (shouldn't be if we load with sr=SR)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    # trim/pad to exactly 1 second
    if len(y) > SR:
        y = y[:SR]
    elif len(y) < SR:
        y = np.pad(y, (0, SR - len(y)), mode='constant')
    return y

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

            # save original (copied) to augmented folder as well
            out_orig = os.path.join(out_class_dir, f"{base}_orig.wav")
            sf.write(out_orig, y, SR)

            aug_variants = augment_clip(y)
            # we may produce more variants than AUG_PER_FILE, so choose randomly
            chosen = aug_variants[:AUG_PER_FILE] if len(aug_variants) >= AUG_PER_FILE else aug_variants
            # if fewer, add random combinations to reach target
            while len(chosen) < AUG_PER_FILE:
                # create a random compound augment
                y2 = time_shift(y, shift_max=0.12)
                y2 = pitch_shift(y2, n_steps=random.uniform(-1,1))
                y2 = add_noise(y2, snr_db=random.uniform(8,18))
                chosen.append(y2)

            for i, aug in enumerate(chosen, 1):
                # final safety: trim/pad to SR
                if len(aug) > SR:
                    aug = aug[:SR]
                elif len(aug) < SR:
                    aug = np.pad(aug, (0, SR - len(aug)), mode='constant')
                out_file = os.path.join(out_class_dir, f"{base}_aug{i}.wav")
                sf.write(out_file, aug, SR)

            print(f"Saved {len(chosen)} augmentations for {path_in} -> {out_class_dir}")

if __name__ == "__main__":
    process_dataset()
