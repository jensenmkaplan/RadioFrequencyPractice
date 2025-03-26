import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd

# Audio parameters
fs = 44100  # Standard audio sampling rate
duration = 2  # seconds
t = np.linspace(0, duration, int(fs * duration))

# 1. Generate a simple audio message (a tone that changes frequency)
message = np.sin(2 * np.pi * 440 * t)  # 440 Hz (A4 note)
message = message + .5 * np.sin(2 * np.pi * 880 * t)  # 880 Hz (A5 note)

# 2. Carrier Waves (testing both 1kHz and 4kHz)
fc1 = 1000  # 1 kHz carrier
fc2 = 4000  # 4 kHz carrier
carrier1 = np.sin(2 * np.pi * fc1 * t)
carrier2 = np.sin(2 * np.pi * fc2 * t)

# 3. Amplitude Modulation (AM) for both carriers
modulation_index = 0.8
am_signal1 = (1 + modulation_index * message) * carrier1
am_signal2 = (1 + modulation_index * message) * carrier2

# 4. Frequency Modulation (FM) for both carriers
frequency_deviation = 100  # Hz deviation
beta1 = frequency_deviation / 440  # modulation index for FM at 1kHz
beta2 = frequency_deviation / 440  # modulation index for FM at 4kHz
fm_signal1 = np.sin(2 * np.pi * fc1 * t + beta1 * message)
fm_signal2 = np.sin(2 * np.pi * fc2 * t + beta2 * message)

# Normalize signals for audio playback
message = message / np.max(np.abs(message))
am_signal1 = am_signal1 / np.max(np.abs(am_signal1))
am_signal2 = am_signal2 / np.max(np.abs(am_signal2))
fm_signal1 = fm_signal1 / np.max(np.abs(fm_signal1))
fm_signal2 = fm_signal2 / np.max(np.abs(fm_signal2))

# Plotting
plt.figure(figsize=(15, 15))

# Plot message signal
plt.subplot(5, 1, 1)
plt.plot(t[:2000], message[:2000])
plt.title('Message Signal (Audio)')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot carrier waves
plt.subplot(5, 1, 2)
plt.plot(t[:2000], carrier1[:2000], label='1 kHz Carrier')
plt.plot(t[:2000], carrier2[:2000], '--', label='4 kHz Carrier')
plt.title('Carrier Signals')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot AM signals with envelopes
plt.subplot(5, 1, 3)
plt.plot(t[:2000], am_signal1[:2000], label='AM Signal (1 kHz)')
envelope = 1 + modulation_index * message
plt.plot(t[:2000], envelope[:2000], 'r--', label='Envelope')
plt.plot(t[:2000], -envelope[:2000], 'r--')
plt.title('AM Signal with 1 kHz Carrier')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 4)
plt.plot(t[:2000], am_signal2[:2000], label='AM Signal (4 kHz)')
plt.plot(t[:2000], envelope[:2000], 'r--', label='Envelope')
plt.plot(t[:2000], -envelope[:2000], 'r--')
plt.title('AM Signal with 4 kHz Carrier')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot FM signals
plt.subplot(5, 1, 5)
plt.plot(t[:2000], fm_signal1[:2000], label='FM Signal (1 kHz)')
plt.plot(t[:2000], fm_signal2[:2000], '--', label='FM Signal (4 kHz)')
plt.title('FM Signals Comparison')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Function to play audio
def play_audio(signal, fs):
    sd.play(signal, fs)
    sd.wait()

# Print parameters
print(f"\nAudio Signal Parameters:")
print(f"Message Frequency: 440 Hz (A4) + 880 Hz (A5)")
print(f"AM Modulation Index: {modulation_index:.2f}")
print(f"FM Deviation: {frequency_deviation:.1f} Hz")
print(f"FM Modulation Index (β): {beta1:.2f}")
print(f"Sampling Rate: {fs/1000:.1f} kHz")
print(f"Duration: {duration:.1f} seconds")

# Play the signals in sequence for comparison
print("\nPlaying signals for comparison...")
print("1. Original message (A4 + A5 notes)")
play_audio(message, fs)

print("2. AM modulated signal with 1 kHz carrier")
play_audio(am_signal1, fs)

print("3. AM modulated signal with 4 kHz carrier")
play_audio(am_signal2, fs)

print("4. FM modulated signal with 1 kHz carrier")
play_audio(fm_signal1, fs)

print("5. FM modulated signal with 4 kHz carrier")
play_audio(fm_signal2, fs) 