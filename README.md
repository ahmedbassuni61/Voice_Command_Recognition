# 📘 Ground Station GUI Documentation

This Python script builds a **Tkinter-based dashboard** that integrates:

- 🎙️ **Real-time voice waveform visualization**
- 🤖 **Voice command recognition with ML**
- 🌡️ **Arduino sensor readings (temperature & light)**
- 🚨 **Popup alerts when objects are detected**
- 📊 **Data logging into CSV + history viewer**

---

## 📂 Main Components

### 1. **Imports**

```python
import tkinter as tk
from tkinter import ttk
import threading, time
from collections import deque
import serial, numpy as np, sounddevice as sd, librosa, joblib
```

- **tkinter** → GUI framework
- **threading** → background tasks for sensors & audio
- **serial** → communicates with Arduino (COM port)
- **numpy, librosa, sounddevice** → audio processing
- **joblib** → loads ML models

---

### 2. **Serial & Model Setup**

```python
arduino_port = "COM11"
baud_rate = 9600
ser = serial.Serial("COM11", 9600, timeout=1)
pause_reading = threading.Event()
model = joblib.load("on_off_rf_model.pkl")
encoder = joblib.load("label_encoder.pkl")
```

- Connects to Arduino (`COM11`, 9600 baud).
- Loads a **Random Forest ML model** (`on_off_rf_model.pkl`) and label encoder.
- `pause_reading` allows safe pausing during command sending.
- Sensor data is saved in **`sensor_data.csv`**.

---

### 3. **Tkinter Main Window**

```python
root = tk.Tk()
root.title("Voice + Sensors Dashboard")
root.geometry("800x650")
root.configure(bg="#2e2e2e")
```

- Main dashboard window.
- Modern dark theme.

---

## 🗂️ Features

### 4. **Logs & History**

- **`history()`** → Displays in-memory log (`log` list).
- **`show_all_data()`** → Loads **all sensor history** from CSV and displays it in a scrollable text box.

---

### 5. **Voice Waveform Display**

- **Canvas (`600x100`)** → Shows **real-time microphone waveform**.
- Functions:

  - `audio_callback()` → Processes audio samples.
  - `draw_waveform()` → Draws on canvas continuously.
  - `start_voice()` → Starts microphone + ML prediction.
  - `stop_voice()` → Stops microphone & resets.

---

### 6. **Sensor Display**

Two labels for **Temperature 🌡️** and **Light 💡**:

```python
lbl_temp_val.config(text=f"{temp_val} °C")
lbl_light_val.config(text=f"{light_val} Unit")
```

- Updated dynamically by `update_sensors()`.

---

### 7. **Reading Sensor Data**

- **`read_sensor_data()`** → Reads Arduino serial line (`temperature,light,alert`).
- **`read_sensors()`** → Runs in background thread, logging data to `sensor_data.csv`.

---

### 8. **Alert Popup**

- Variables:

  ```python
  signal_active = False
  popup = None
  ```

- Functions:

  - `trigger_signal()` → Creates red alert popup if **alert=1**.
  - `clear_signal()` → Closes popup.
  - `monitor_sensors()` → Continuously checks for alerts (every 500ms).

---

### 9. **Sending Commands to Arduino**

```python
def send_data(command, pause_time=1):
    pause_reading.set()
    ser.write(f"{command}\n".encode())
    time.sleep(pause_time)
    pause_reading.clear()
```

- Sends **ON (1)** / **OFF (0)** command to Arduino.
- Prevents data conflicts by pausing sensor reading temporarily.

---

### 10. 🎙️ **Voice Command Prediction**

```python
def predict_from_mic(seconds=2, n_mfcc=13, max_len=88, rms_threshold=0.01):
    ...
```

Workflow:

1. Records microphone input.
2. Removes silence + normalizes.
3. Extracts **MFCC features**.
4. Detects **silence** if RMS below threshold.
5. Passes features → ML model → predicts `"on" / "off"`.
6. Sends command to Arduino + logs action.
7. Runs continuously in background thread.

---

### 11. **Requesting Data**

- **`request_latest_data()`** → Reads last line from `sensor_data.csv` and updates GUI.

---

### 12. **Control Buttons**

Modern styled buttons:

1. ▶ **Start Listening** → `start_voice()`
2. ⏹ **Stop Listening** → `stop_voice()`
3. 📡 **Request Latest Data** → `request_latest_data()`
4. 📊 **Show All Data** → `show_all_data()`
5. 📜 **Show Log** → `history()`

---

### 13. **Background Threads**

- **Sensor Reading** → `threading.Thread(target=read_sensors, daemon=True)`
- **Alert Monitor** → `monitor_sensors()`
- **UI Updates** → `update_sensors()`
- **Main Loop** → `root.mainloop()`

---

## ⚡ Feature Summary

✅ **Voice waveform visualization**
✅ **Sensor data logging (Temperature + Light)**
✅ **CSV history tracking**
✅ **Popup alerts (alert=1 from Arduino)**
✅ **Real-time ON/OFF voice recognition with ML**
✅ **Command sending to Arduino**
✅ **Manual control via buttons**
✅ **Multithreaded background tasks**
