import sounddevice as sd
import soundfile as sf


def record_command(filename, duration=2, samplerate=16000):
    print("🎤 Recording... Speak now!")
    audio = sd.rec(
        int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32"
    )
    sd.wait()  # Wait until recording is finished
    sf.write(filename, audio, samplerate)
    print(f"✅ Saved: {filename}")


for i in range(18, 39):
    # Record "on"
    record_command(f"dataset/on/on_{i}.wav", duration=2)

    # Record "off"
    record_command(f"dataset/off/off_{i}.wav", duration=2)

    # # Record "unknown"
    # record_command(f"dataset/unknown/unknown_{i}.wav", duration=2)
