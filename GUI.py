# GUI.py
import tkinter as tk
import sounddevice as sd
import numpy as np
from collections import deque
import serial
import numpy as np
import librosa
import joblib
import time
import threading

# ----------------- Serial Port Config and load model ------------------
model = joblib.load("on_off_model.pkl")  # trained sklearn model
encoder = joblib.load("label_encoder.pkl")  # label encoder
ser = serial.Serial("COM11", 9600)  # Update with your Arduino port
pause_reading = threading.Event()
# ---- Tkinter window ----
root = tk.Tk()
root.title("Voice + Sensors Dashboard")
root.geometry("700x500")
root.configure(bg="#f0f0f0")

# ---- Title ----
title = tk.Label(
    root, text="Voice Recorder Dashboard", font=("Arial", 16, "bold"), bg="#f0f0f0"
)
title.pack(pady=10)

# ---- Log storage ----
log = []


def history():
    tp = tk.Toplevel()
    tp.title("Log")
    tp.geometry("300x250")

    txt = tk.Text(tp, wrap="word")
    txt.pack(expand=True, fill="both")

    for item in log:
        txt.insert("end", item + "\n")


# --------------History ----------------
history_log = "sensor_data.csv"


# ---- Helper: Read sensor data ----
def read_sensor_data():
    if not pause_reading.is_set():
        try:
            line = ser.readline().decode().strip()
            parts = line.split(",")  # expecting "temp,light,alert"
            if len(parts):
                temp_val = float(parts[0])
                light_val = int(parts[1])
                alert_val = int(parts[2])
                return temp_val, light_val, alert_val
        except:
            print("Serial read error")
    return None, None, None


# ---------------- read and save data in csv form ----------------
def read_sensors():
    while True:
        line = read_sensor_data()  # e.g., "25.6,300"
        if line and all(v is not None for v in line):
            try:
                temp, light, alert = line
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(history_log, "a") as f:
                    f.write(f"{timestamp},{temp},{light},{alert}\n")
            except Exception as e:
                print("Error parsing:", line, e)


# ---- Menu ----
menu = tk.Menu(root)
menu.add_cascade(label="Log", command=history)
root.config(menu=menu)

# ---- Voice Waveform ----
frame_voice = tk.Frame(root, bg="white", relief="sunken", bd=2)
frame_voice.pack(pady=10)

lbl_voice = tk.Label(frame_voice, text="🎤 Voice (5 seconds)", font=("Arial", 12))
lbl_voice.pack()

canvas = tk.Canvas(frame_voice, width=600, height=100, bg="white")
canvas.pack(padx=10, pady=5)

WIDTH, HEIGHT = 600, 100
buffer_size = 1024
waveform = deque([0] * WIDTH, maxlen=WIDTH)
stream = None


# --------------------model--------------------


# helper: send data
def send_data(command, pause_time=1):
    pause_reading.set()  # pause sensor reading
    ser.write(f"{command}\n".encode())
    time.sleep(pause_time)  # pause for specific time
    pause_reading.clear()


def predict_from_mic(seconds=2, n_mfcc=13, max_len=88, rms_threshold=0.01):
    def worker():
        # Record audio
        audio = sd.rec(
            int(seconds * 16000), samplerate=16000, channels=1, dtype="float32"
        )
        sd.wait()
        audio = audio.flatten()

        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # Force length = 2 seconds (32k samples)
        if len(audio) > seconds * 16000:
            audio = audio[: seconds * 16000]
        elif len(audio) < seconds * 16000:
            audio = np.pad(
                audio, (0, int(seconds * 16000) - len(audio)), mode="constant"
            )

        # Check RMS energy (silence detection)
        rms = np.sqrt(np.mean(audio**2))
        if rms < rms_threshold:
            label = "silence"
        else:
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

        # -----------------------------
        # Send signal to Arduino
        # -----------------------------
        if label == "on":
            # Send ON signal
            send_data("1", pause_time=1)
            log.append("Command: ON → Sent '1'")
        elif label == "off":
            # Send OFF signal
            send_data("0", pause_time=1)
            log.append("Command: OFF → Sent '0'")
        else:
            pass  # silence or unknown

        # Schedule next prediction
        root.after(2000, lambda: threading.Thread(target=predict_from_mic).start())

    # Run heavy work in a background thread
    threading.Thread(target=worker, daemon=True).start()


def audio_callback(indata, frames, time, status):
    global waveform
    samples = indata[:, 0]
    samples = np.interp(samples, (-1, 1), (-HEIGHT // 2, HEIGHT // 2))
    for s in samples:
        waveform.append(s)


def draw_waveform():
    canvas.delete("wave")
    mid = HEIGHT // 2
    points = [(x, mid - int(y)) for x, y in enumerate(waveform)]
    for i in range(len(points) - 1):
        canvas.create_line(
            points[i][0],
            points[i][1],
            points[i + 1][0],
            points[i + 1][1],
            fill="lime",
            tags="wave",
        )
    root.after(30, draw_waveform)


def start_voice():
    global stream
    lbl_voice.config(text="🎙️ Listening...")
    stream = sd.InputStream(
        callback=audio_callback, channels=1, samplerate=44100, blocksize=buffer_size
    )
    stream.start()
    draw_waveform()
    predict_from_mic()  # start prediction loop


def stop_voice():
    global stream
    if stream:
        stream.stop()
        stream.close()
        lbl_voice.config(text="⏹ Stopped")
        stream = None


# ---- Sensor Bars (Temp & Light) ----
frame_sensors = tk.Frame(root, bg="#f0f0f0")
frame_sensors.pack(pady=15)

lbl_temp = tk.Label(frame_sensors, text="🌡️ Temp", font=("Arial", 12), bg="#f0f0f0")
lbl_temp.grid(row=0, column=0, padx=30)

lbl_light = tk.Label(frame_sensors, text="💡 Light", font=("Arial", 12), bg="#f0f0f0")
lbl_light.grid(row=0, column=1, padx=30)

canvas_temp = tk.Canvas(frame_sensors, width=200, height=30, bg="gray")
canvas_temp.grid(row=1, column=0, padx=30, pady=5)

canvas_light = tk.Canvas(frame_sensors, width=200, height=30, bg="gray")
canvas_light.grid(row=1, column=1, padx=30, pady=5)


def update_sensors(temp_val=0, light_val=0):
    # Clear previous drawings
    canvas_temp.delete("all")
    canvas_light.delete("all")

    # Draw bar backgrounds
    canvas_temp.create_rectangle(0, 0, 200, 30, fill="gray")
    canvas_light.create_rectangle(0, 0, 200, 30, fill="gray")

    # Write text values in the middle
    canvas_temp.create_text(
        100, 15, text=f"{temp_val} °C", fill="white", font=("Arial", 12, "bold")
    )
    canvas_light.create_text(
        100, 15, text=f"{light_val} Unit", fill="black", font=("Arial", 12, "bold")
    )


# --- Request Button ---
def request_sensors(timeout=2):
    _, temp, light, _ = None, None, None, None
    with open("sensor_data.csv", "r") as f:
        _, temp, light, _ = f.readlines()[-1].strip().split(",")
    if temp is not None and light is not None:  # check if data is valid
        update_sensors(temp_val=temp, light_val=light)
        log.append(f"🌡️ Temperature requested: {temp} °C")
        log.append(f"💡 Light requested: {light} Unit")
        return  # exit once data is received

    # If data not valid
    log.append("⚠️ No readings received.")


btn_temp_req = tk.Button(
    frame_sensors, text="Request readings", command=request_sensors
)
btn_temp_req.grid(row=2, column=0, columnspan=2, pady=10)  # center under both bars

# ---------- Alert system ----------------
signal_active = False
popup = None


def trigger_signal():
    global signal_active, popup
    if not signal_active:
        signal_active = True
        log.append("⚠ Object Detected!")  # log it
        popup = tk.Toplevel(root)
        popup.title("🚨 Alert")
        popup.geometry("250x120")
        popup.configure(bg="red")

        lbl = tk.Label(
            popup,
            text="Signal Received!",
            font=("Arial", 14, "bold"),
            fg="white",
            bg="red",
        )
        lbl.pack(pady=20)

        btn_close = tk.Button(popup, text="Close Alert", command=clear_signal)
        btn_close.pack(pady=5)


def clear_signal():
    global signal_active, popup
    if popup:
        popup.destroy()
        popup = None
    signal_active = False


def monitor_sensors():
    _, _, _, alert = None, None, None, None
    with open("sensor_data.csv", "r") as f:
        _, _, _, alert = f.readlines()[-1].strip().split(",")
    try:
        if alert == 1:
            trigger_signal()
    except Exception as e:
        print("Serial error:", e)

    root.after(500, monitor_sensors)  # check again in 500 ms


# ---- Control Buttons ----
frame_buttons = tk.Frame(root, bg="#f0f0f0")
frame_buttons.pack(pady=10)

btn_start = tk.Button(
    frame_buttons, text="▶ Start", fg="green", width=12, command=start_voice
)
btn_start.grid(row=0, column=0, padx=20)

btn_stop = tk.Button(
    frame_buttons, text="⏹ Stop", fg="red", width=12, command=stop_voice
)
btn_stop.grid(row=0, column=1, padx=20)

# ---- Run ----
# Start background thread to read sensors continuously
thread = threading.Thread(target=read_sensors, daemon=True)
thread.start()
# -------------------------------
update_sensors()  # initialize sensor bars
monitor_sensors()  # start monitoring alerts
root.mainloop()
