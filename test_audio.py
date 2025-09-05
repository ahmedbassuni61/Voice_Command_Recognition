import numpy as np
import librosa
import sounddevice as sd
import joblib

# -----------------------------
# Load trained model + encoder
# -----------------------------
model = joblib.load("on_off_model.pkl")  # trained sklearn model
encoder = joblib.load("label_encoder.pkl")  # label encoder


# -----------------------------
# Real-time prediction
# -----------------------------
def predict_from_mic(seconds=2, n_mfcc=13, max_len=88, rms_threshold=0.01):
    audio = sd.rec(int(seconds * 16000), samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()

    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Force length = 2 seconds (32k samples)
    if len(audio) > seconds * 16000:
        audio = audio[: seconds * 16000]
    elif len(audio) < seconds * 16000:
        audio = np.pad(audio, (0, int(seconds * 16000) - len(audio)), mode="constant")

    # Check RMS energy (silence detection)
    rms = np.sqrt(np.mean(audio**2))
    if rms < rms_threshold:
        return "silence"

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=n_mfcc)

    # Pad or truncate MFCC frames
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    # Flatten for sklearn model (same as training)
    mfcc = mfcc.flatten().reshape(1, -1)

    # Predict
    prediction = model.predict(mfcc)[0]
    label = encoder.inverse_transform([prediction])[0]

    return label


# -----------------------------
# Continuous testing loop
# -----------------------------
print("🎤 Starting live detection. Press Ctrl+C to stop.")
try:
    while True:
        print("Speak now...")
        detected = predict_from_mic()
        print(f"🔊 Detected: {detected}\n")
except KeyboardInterrupt:
    print("\n🛑 Stopped live detection.")
