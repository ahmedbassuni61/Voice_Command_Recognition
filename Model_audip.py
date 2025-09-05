import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# -----------------------------
# Parameters
# -----------------------------
DATASET_PATH = "dataset_augmented"  # dataset/on/, dataset/off/, dataset/noise/
SAMPLES_PER_FILE = 32000  # 2 second audio at 16kHz


# -----------------------------
# Feature Extraction (MFCCs)
# -----------------------------
def extract_features(file_path, n_mfcc=13, max_len=88):
    audio, sr = librosa.load(file_path, sr=16000)
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Force length = 2 seconds (32k samples)
    if len(audio) > SAMPLES_PER_FILE:
        audio = audio[:SAMPLES_PER_FILE]
    elif len(audio) < SAMPLES_PER_FILE:
        audio = np.pad(audio, (0, SAMPLES_PER_FILE - len(audio)), mode="constant")

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # Pad or truncate to fixed length
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    # Flatten to 1D (since sklearn models expect 1D features)
    return mfcc.flatten()


# -----------------------------
# Load Dataset
# -----------------------------
labels = []
features = []

for label in os.listdir(DATASET_PATH):
    class_dir = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_dir):
        continue
    for file in os.listdir(class_dir):
        if file.endswith(".wav"):
            file_path = os.path.join(class_dir, file)
            mfcc = extract_features(file_path)
            features.append(mfcc)
            labels.append(label)

X = np.array(features)
y = np.array(labels)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Choose Model
# -----------------------------
# You can swap models here
# model = RandomForestClassifier(n_estimators=200, random_state=42)
# model = SVC(kernel="rbf", probability=True)
model = XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
acc = model.score(X_test, y_test)
print(f"✅ Test Accuracy: {acc:.2f}")

# -----------------------------
# Save Model & Encoder
# -----------------------------
joblib.dump(model, "on_off_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")
print("✅ Model trained and saved as on_off_model.pkl")
