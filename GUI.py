# modern_dashboard_numbers_scroll.py
import tkinter as tk
from tkinter import ttk
import threading
import time
from collections import deque
import serial
import numpy as np
import sounddevice as sd
import librosa
import joblib

# -------------------- Serial & Model Setup --------------------
ser = None
arduino_connected = False
arduino_port = "COM11"  # change if needed
baud_rate = 9600


def try_connect():
    global ser, arduino_connected
    if not arduino_connected:  # only try if not already connected
        try:
            ser = serial.Serial(arduino_port, baud_rate, timeout=1)
            arduino_connected = True
            print("✅ Arduino connected.")
            log.append("✅ Arduino connected.")
        except Exception:
            text = "❌ Arduino not connected. Retrying..."
            if text not in log:
                log.append(text)


def check_connection():
    global arduino_connected, ser
    if arduino_connected and ser:
        if not ser.is_open:  # if lost connection
            arduino_connected = False
    else:
        try_connect()  # try to reconnect

    root.after(2000, check_connection)  # check again every 2s


ser = serial.Serial("COM11", 9600, timeout=1)  # Update COM port
time.sleep(2)
pause_reading = threading.Event()

model = joblib.load("on_off_rf_model.pkl")
encoder = joblib.load("label_encoder.pkl")

history_log = "sensor_data.csv"
log = []

# -------------------- Tkinter Setup --------------------
root = tk.Tk()
root.title("Voice + Sensors Dashboard")
root.geometry("800x650")
root.configure(bg="#2e2e2e")


# -------------------- Logs --------------------
def history():
    tp = tk.Toplevel()
    tp.title("Log")
    tp.geometry("400x300")
    tp.configure(bg="#1c1c1c")
    txt = tk.Text(tp, wrap="word", bg="#1c1c1c", fg="white", font=("Helvetica", 10))
    txt.pack(expand=True, fill="both")
    for item in log:
        txt.insert("end", item + "\n")


def show_all_data():
    try:
        tp = tk.Toplevel()
        tp.title("All Sensor Data")
        tp.geometry("550x400")
        tp.configure(bg="#1c1c1c")
        frame = tk.Frame(tp, bg="#1c1c1c")
        frame.pack(expand=True, fill="both")
        txt = tk.Text(
            frame, wrap="none", bg="#1c1c1c", fg="white", font=("Helvetica", 10)
        )
        txt.pack(side="left", expand=True, fill="both")
        scroll_y = tk.Scrollbar(frame, orient="vertical", command=txt.yview)
        scroll_y.pack(side="right", fill="y")
        txt.configure(yscrollcommand=scroll_y.set)
        with open(history_log, "r") as f:
            for line in f:
                if line == "Timestamp,Temperature,Light,alert\n" or line.strip() == "":
                    continue
                timestamp, temp, light, alert = line.strip().split(",")

                alert_text = "Yes" if alert == "1" else "No"
                txt.insert(
                    "end",
                    f"Time: {timestamp} | Temperature: {temp} °C | Light: {light} Unit | Alert: {alert_text}\n",
                )
    except FileNotFoundError:
        log.append("No CSV file found.")


# -------------------- Voice Waveform --------------------
WIDTH, HEIGHT = 600, 100
buffer_size = 1024
waveform = deque([0] * WIDTH, maxlen=WIDTH)
stream = None

frame_voice = tk.Frame(root, bg="#3a3a3a", bd=2, relief="sunken")
frame_voice.pack(pady=10, padx=20, fill="x")

lbl_voice = tk.Label(
    frame_voice,
    text="🎤 Voice (5 seconds)",
    font=("Helvetica", 14, "bold"),
    bg="#3a3a3a",
    fg="white",
)
lbl_voice.pack()

canvas = tk.Canvas(frame_voice, width=WIDTH, height=HEIGHT, bg="#1c1c1c")
canvas.pack(padx=10, pady=5)


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
    predict_from_mic()


def stop_voice():
    global stream
    if stream:
        stream.stop()
        stream.close()
        lbl_voice.config(text="⏹ Stopped")
        stream = None


# -------------------- Sensor Display --------------------
frame_sensors = tk.Frame(root, bg="#3a3a3a")
frame_sensors.pack(pady=10, padx=20, fill="x")

lbl_temp_text = tk.Label(
    frame_sensors,
    text="🌡️ Temperature:",
    font=("Helvetica", 12, "bold"),
    bg="#3a3a3a",
    fg="white",
)
lbl_temp_text.grid(row=0, column=0, padx=20, pady=5, sticky="w")
lbl_temp_val = tk.Label(
    frame_sensors, text="0 °C", font=("Helvetica", 12, "bold"), bg="#3a3a3a", fg="white"
)
lbl_temp_val.grid(row=0, column=1, padx=10, pady=5, sticky="w")

lbl_light_text = tk.Label(
    frame_sensors,
    text="💡                     Light:",
    font=("Helvetica", 12, "bold"),
    bg="#3a3a3a",
    fg="white",
)
lbl_light_text.grid(row=1, column=0, padx=20, pady=5, sticky="w")
lbl_light_val = tk.Label(
    frame_sensors,
    text="0 Unit",
    font=("Helvetica", 12, "bold"),
    bg="#3a3a3a",
    fg="white",
)
lbl_light_val.grid(row=1, column=1, padx=10, pady=5, sticky="w")


def update_sensors(temp_val=0, light_val=0):
    lbl_temp_val.config(text=f"{temp_val} °C")
    lbl_light_val.config(text=f"{light_val} Unit")


# -------------------- Read Sensor Data --------------------
def read_sensor_data():
    if not pause_reading.is_set():
        try:
            line = ser.readline().decode().strip()
            parts = line.split(",")
            if len(parts) == 3:
                temp_val = float(parts[0])
                light_val = int(parts[1])
                alert_val = int(parts[2])
                return temp_val, light_val, alert_val
        except:
            print("Serial read error")
    return None, None, None


def read_sensors():
    while True:
        line = read_sensor_data()
        if line and all(v is not None for v in line):
            try:
                temp, light, alert = line
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(history_log, "a") as f:
                    f.write(f"{timestamp},{temp},{light},{alert}\n")
            except Exception as e:
                print("Error parsing:", line, e)


# -------------------- Alert Popup --------------------
signal_active = False
popup = None


def trigger_signal():
    global signal_active, popup
    if not signal_active:
        signal_active = True
        log.append("⚠ Object Detected!")

        popup = tk.Toplevel(root)
        popup.title("🚨 Alert")
        popup.geometry("300x150")
        popup.configure(bg="#ff4c4c")  # softer red
        popup.resizable(False, False)

        # Rounded frame effect
        frame = tk.Frame(popup, bg="#ff6666", bd=2, relief="ridge", padx=15, pady=15)
        frame.pack(expand=True, fill="both", padx=10, pady=10)

        lbl = tk.Label(
            frame,
            text="⚠ Object Detected!",
            font=("Helvetica", 16, "bold"),
            fg="white",
            bg="#ff6666",
        )
        lbl.pack(pady=(0, 20))

        # Modern flat button
        btn_close = tk.Button(
            frame,
            text="Dismiss",
            font=("Helvetica", 12, "bold"),
            fg="white",
            bg="#222222",
            activebackground="#555555",
            relief="flat",
            padx=20,
            pady=5,
            command=clear_signal,
        )
        btn_close.pack()


def clear_signal():
    global signal_active, popup
    if popup:
        popup.destroy()
        popup = None
    signal_active = False


def monitor_sensors():
    try:
        with open(history_log, "r") as f:
            last_line = f.readlines()[-1].strip()
            _, _, _, alert = last_line.split(",")
            if int(alert) == 1:
                trigger_signal()
    except:
        pass
    root.after(500, monitor_sensors)


# -------------------- Send ON/OFF --------------------
def send_data(command, pause_time=1):
    pause_reading.set()
    ser.write(f"{command}\n".encode())
    time.sleep(pause_time)
    pause_reading.clear()


# -------------------- Voice Prediction --------------------
def predict_from_mic(seconds=2, n_mfcc=13, max_len=88, rms_threshold=0.01):
    def worker():
        audio = sd.rec(
            int(seconds * 16000), samplerate=16000, channels=1, dtype="float32"
        )
        sd.wait()
        audio = audio.flatten()
        audio, _ = librosa.effects.trim(audio, top_db=20)
        if len(audio) > seconds * 16000:
            audio = audio[: int(seconds * 16000)]
        else:
            audio = np.pad(audio, (0, int(seconds * 16000) - len(audio)), "constant")
        rms = np.sqrt(np.mean(audio**2))
        if rms < rms_threshold:
            label = "silence"
        else:
            mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=n_mfcc)
            if mfcc.shape[1] < max_len:
                pad_width = max_len - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
            else:
                mfcc = mfcc[:, :max_len]
            mfcc = mfcc.flatten().reshape(1, -1)
            prediction = model.predict(mfcc)[0]
            label = encoder.inverse_transform([prediction])[0]
        if label == "on":
            send_data("1", pause_time=1)
            log.append("Command: ON → Sent '1'")
        elif label == "off":
            send_data("0", pause_time=1)
            log.append("Command: OFF → Sent '0'")
        root.after(50, lambda: threading.Thread(target=worker).start())

    threading.Thread(target=worker, daemon=True).start()


# -------------------- Request Latest Data --------------------
def request_latest_data():
    try:
        with open(history_log, "r") as f:
            last_line = f.readlines()[-1].strip()
            timestamp, temp, light, alert = last_line.split(",")
            update_sensors(temp_val=temp, light_val=light)
            log.append(f"🌡️ Temperature requested: {temp} °C")
            log.append(f"💡 Light requested: {light} Unit")
    except:
        log.append("⚠️ No readings received yet.")


# -------------------- Control Buttons --------------------
frame_buttons = tk.Frame(root, bg="#2e2e2e")
frame_buttons.pack(pady=10)

btn_start = tk.Button(
    frame_buttons,
    text="▶ Start Listening",
    bg="#4CAF50",
    fg="white",
    font=("Helvetica", 12, "bold"),
    width=20,
    height=2,
    relief="flat",
    command=start_voice,
)
btn_start.grid(row=0, column=0, padx=10)

btn_stop = tk.Button(
    frame_buttons,
    text="⏹ Stop Listening",
    bg="#F44336",
    fg="white",
    font=("Helvetica", 12, "bold"),
    width=20,
    height=2,
    relief="flat",
    command=stop_voice,
)
btn_stop.grid(row=0, column=1, padx=10)

btn_req = tk.Button(
    frame_buttons,
    text="Request Latest Data",
    bg="#2196F3",
    fg="white",
    font=("Helvetica", 12, "bold"),
    width=20,
    height=2,
    relief="flat",
    command=request_latest_data,
)
btn_req.grid(row=1, column=0, padx=10, pady=10)

btn_show_all = tk.Button(
    frame_buttons,
    text="Show All Data",
    bg="#FF9800",
    fg="white",
    font=("Helvetica", 12, "bold"),
    width=20,
    height=2,
    relief="flat",
    command=show_all_data,
)
btn_show_all.grid(row=1, column=1, padx=10, pady=10)

btn_log = tk.Button(
    frame_buttons,
    text="Show Log",
    bg="#9C27B0",
    fg="white",
    font=("Helvetica", 12, "bold"),
    width=42,
    height=2,
    relief="flat",
    command=history,
)
btn_log.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# -------------------- Start Threads --------------------
thread = threading.Thread(target=read_sensors, daemon=True)
thread.start()
monitor_sensors()
update_sensors()
root.mainloop()
