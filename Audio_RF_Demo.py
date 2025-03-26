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

# 2. Carrier Wave (using a higher frequency that's still audible for demonstration)
fc = 2000  # 2 kHz carrier, original suggested carrier
# uncomment one of the following to change the carrier frequency    
# fc = 1000  # 1 kHz carrier
# fc = 4000  # 4 kHz carrier
carrier = np.sin(2 * np.pi * fc * t)

# 3. Amplitude Modulation (AM)
modulation_index = 0.8
am_signal = (1 + modulation_index * message) * carrier

# 4. Frequency Modulation (FM)
frequency_deviation = 100  # Hz deviation
beta = frequency_deviation / 440  # modulation index for FM
fm_signal = np.sin(2 * np.pi * fc * t + beta * message)

# Normalize signals for audio playback
message = message / np.max(np.abs(message))
am_signal = am_signal / np.max(np.abs(am_signal))
fm_signal = fm_signal / np.max(np.abs(fm_signal))

# Plotting
plt.figure(figsize=(15, 12))

# Plot message signal
plt.subplot(4, 1, 1)
plt.plot(t[:2000], message[:2000])
plt.title('Message Signal (Audio)')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot carrier wave
plt.subplot(4, 1, 2)
plt.plot(t[:2000], carrier[:2000])
plt.title(f'Carrier Signal ({fc/1000:.1f} kHz)')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot AM signal with envelope
plt.subplot(4, 1, 3)
plt.plot(t[:2000], am_signal[:2000], label='AM Signal')
envelope = 1 + modulation_index * message
plt.plot(t[:2000], envelope[:2000], 'r--', label='Envelope')
plt.plot(t[:2000], -envelope[:2000], 'r--')
plt.title('Amplitude Modulated Signal (with envelope)')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot FM signal with original carrier
plt.subplot(4, 1, 4)
plt.plot(t[:2000], fm_signal[:2000], label='FM Signal')
plt.plot(t[:2000], carrier[:2000], 'r--', label='Original Carrier')
plt.title('Frequency Modulated Signal (with original carrier)')
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

# Print some interesting parameters
print(f"\nAudio Signal Parameters:")
print(f"Carrier Frequency: {fc/1000:.1f} kHz")
print(f"Message Frequency: 440 Hz (A4) + 880 Hz (A5)")
print(f"AM Modulation Index: {modulation_index:.2f}")
print(f"FM Deviation: {frequency_deviation:.1f} Hz")
print(f"FM Modulation Index (β): {beta:.2f}")
print(f"Sampling Rate: {fs/1000:.1f} kHz")
print(f"Duration: {duration:.1f} seconds")

# Play the signals
print("\nPlaying signals...")
print("1. Original message (A4 + A5 notes)")
play_audio(message, fs)

print("2. AM modulated signal")
play_audio(am_signal, fs)

print("3. FM modulated signal")
play_audio(fm_signal, fs) 