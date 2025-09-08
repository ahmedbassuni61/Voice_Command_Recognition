import numpy as np
import librosa
import sounddevice as sd
import joblib

# -----------------------------
# Load trained model + encoder
# -----------------------------
clf = joblib.load("on_off_rf_model.pkl")   # Random Forest model
encoder = joblib.load("label_encoder.pkl") # saved during training

# -----------------------------
# Parameters
# -----------------------------
SAMPLE_RATE = 16000
SAMPLES_PER_FILE = 32000  # 2 seconds
N_MFCC = 13
MAX_LEN = 88

# -----------------------------
# Prediction function
# -----------------------------
def predict_from_mic(seconds=2, rms_threshold=0.01):
    # Record audio
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()

    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Force fixed length
    if len(audio) > SAMPLES_PER_FILE:
        audio = audio[:SAMPLES_PER_FILE]
    elif len(audio) < SAMPLES_PER_FILE:
        audio = np.pad(audio, (0, SAMPLES_PER_FILE - len(audio)), mode="constant")

    # Check RMS energy (detect silence)
    rms = np.sqrt(np.mean(audio**2))
    if rms < rms_threshold:
        return "silence"

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

    # Pad or truncate
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]

    # Flatten for Random Forest
    mfcc = mfcc.flatten().reshape(1, -1)

    # Predict
    prediction = clf.predict(mfcc)[0]
    prob = np.max(clf.predict_proba(mfcc))

    print(prob)

    # Optional: ignore low-confidence predictions
    # if prob < 0.5:
    #     return "silence"

    return encoder.inverse_transform([prediction])[0]

# -----------------------------
# Continuous testing loop
# -----------------------------
print("🎤 Starting live detection (Random Forest). Press Ctrl+C to stop.")
try:
    while True:
        print("Speak now...")
        detected = predict_from_mic()
        print(f"🔊 Detected: {detected}\n")
except KeyboardInterrupt:
    print("\n🛑 Stopped live detection.")
