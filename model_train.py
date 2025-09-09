import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV

# -----------------------------
# Parameters
# -----------------------------
DATASET_PATH = "dataset_augmented"
SAMPLES_PER_FILE = 32000   # 2 sec @16kHz

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(file_path, n_mfcc=13, max_len=88):
    audio, sr = librosa.load(file_path, sr=16000)
    audio, _ = librosa.effects.trim(audio, top_db=20)

    if len(audio) > SAMPLES_PER_FILE:
        audio = audio[:SAMPLES_PER_FILE]
    elif len(audio) < SAMPLES_PER_FILE:
        audio = np.pad(audio, (0, SAMPLES_PER_FILE - len(audio)), mode='constant')

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc.flatten()   #Flatten

# -----------------------------
# Load Dataset
# -----------------------------
labels, features = [], []

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# -----------------------------
# Model
# -----------------------------

# base_model  =   ExtraTreesClassifier(n_estimators=140, random_state=42)
# base_model  =   SVC(kernel="poly",
#                 degree=3,        # try 2 or 3
#                 C=0.8,            # regularization strength
#                 gamma="scale",   # kernel coefficient
#                 probability=True,
#                 random_state=42)
base_model  =     RandomForestClassifier(n_estimators=150,random_state=42)
# base_model  =   XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=15, random_state=42)
#calibrated_model = CalibratedClassifierCV(base_model, cv=10)  # Platt scaling

mypipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("Classifier",base_model)
    ])

mypipeline.fit(X_train, y_train)

# Evaluate
y_pred = mypipeline.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display with labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=encoder.classes_)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix -")
plt.show()

acc = accuracy_score(y_test, y_pred)
print(f"✅ test accuracy: {acc:.2f}")

# -----------------------------
# Save Model + Encoder
# -----------------------------
joblib.dump(mypipeline, "on_off_rf_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")
print("✅ model saved as on_off_rf_model.pkl")
