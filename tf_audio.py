import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# -----------------------------
# Parameters
# -----------------------------
DATASET_PATH = "dataset_augmented"   # folder structure: dataset/on/, dataset/off/, dataset/noise/
SAMPLES_PER_FILE = 32000   # 2 second audio at 16kHz

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
        audio = np.pad(audio, (0, SAMPLES_PER_FILE - len(audio)), mode='constant')
        
    # Extract MFCC        
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # Pad or truncate to fixed length
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

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
y = to_categorical(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN (add channel dimension)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# -----------------------------
# Build Model
# -----------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                    validation_data=(X_test, y_test))

# -----------------------------
# Save Model
# -----------------------------
model.save("on_off_model.h5")
print("✅ Model trained and saved as on_off_model.h5")
joblib.dump(encoder, "label_encoder.pkl")