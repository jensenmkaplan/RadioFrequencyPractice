import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal

# Audio parameters
fs = 44100  # Standard audio sampling rate
duration = 2  # seconds
t = np.linspace(0, duration, int(fs * duration))

# 1. Generate a simple audio message (a tone that changes frequency)
message = np.sin(2 * np.pi * 440 * t)  # 440 Hz (A4 note)
message = message + 0.5 * np.sin(2 * np.pi * 880 * t)  # 880 Hz (A5 note)

# 2. Carrier Wave
fc = 2000  # 2 kHz carrier
carrier = np.sin(2 * np.pi * fc * t)

# 3. Amplitude Modulation (AM)
modulation_index = 0.8
am_signal = (1 + modulation_index * message) * carrier

# 4. Frequency Modulation (FM)
frequency_deviation = 100  # Hz deviation
beta = frequency_deviation / 440  # modulation index for FM
fm_signal = np.sin(2 * np.pi * fc * t + beta * message)

# 5. Demodulation

# AM Demodulation (Envelope Detection using Hilbert Transform)
def am_demodulate(signal, fs):
    # Get the analytic signal using Hilbert transform
    analytic_signal = scipy.signal.hilbert(signal)
    
    # Get the envelope
    envelope = np.abs(analytic_signal)
    
    # Remove DC offset and normalize
    envelope = envelope - np.mean(envelope)
    envelope = envelope / np.max(np.abs(envelope))
    
    # Low-pass filter to remove carrier frequency components
    cutoff = 1000  # Hz
    b, a = scipy.signal.butter(4, cutoff/(fs/2), btype='low')
    envelope = scipy.signal.filtfilt(b, a, envelope)
    
    return envelope

# FM Demodulation (Frequency Discrimination)
def fm_demodulate(signal, fs):
    # Differentiate the signal to get frequency
    diff_signal = np.diff(signal)
    
    # Add zero to maintain same length
    diff_signal = np.append(diff_signal, 0)
    
    # Remove DC offset and normalize
    diff_signal = diff_signal - np.mean(diff_signal)
    diff_signal = diff_signal / np.max(np.abs(diff_signal))
    
    # Low-pass filter to remove carrier frequency components
    cutoff = 1000  # Hz
    b, a = scipy.signal.butter(4, cutoff/(fs/2), btype='low')
    diff_signal = scipy.signal.filtfilt(b, a, diff_signal)
    
    return diff_signal

# Demodulate signals
am_demod = am_demodulate(am_signal, fs)
fm_demod = fm_demodulate(fm_signal, fs)

# Normalize all signals for audio playback
message = message / np.max(np.abs(message))
am_demod = am_demod / np.max(np.abs(am_demod))
fm_demod = fm_demod / np.max(np.abs(fm_demod))

# Plotting
plt.figure(figsize=(15, 15))

# Plot original message
plt.subplot(5, 1, 1)
plt.plot(t[:2000], message[:2000])
plt.title('Original Message Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot AM signal
plt.subplot(5, 1, 2)
plt.plot(t[:2000], am_signal[:2000])
plt.title('AM Modulated Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot AM demodulated
plt.subplot(5, 1, 3)
plt.plot(t[:2000], am_demod[:2000])
plt.title('AM Demodulated Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot FM signal
plt.subplot(5, 1, 4)
plt.plot(t[:2000], fm_signal[:2000])
plt.title('FM Modulated Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot FM demodulated
plt.subplot(5, 1, 5)
plt.plot(t[:2000], fm_demod[:2000])
plt.title('FM Demodulated Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()

# Function to play audio
def play_audio(signal, fs):
    sd.play(signal, fs)
    sd.wait()

# Print parameters
print(f"\nSignal Parameters:")
print(f"Carrier Frequency: {fc/1000:.1f} kHz")
print(f"Message Frequency: 440 Hz (A4) + 880 Hz (A5)")
print(f"AM Modulation Index: {modulation_index:.2f}")
print(f"FM Deviation: {frequency_deviation:.1f} Hz")
print(f"FM Modulation Index (β): {beta:.2f}")

# Play the signals
print("\nPlaying signals...")
print("1. Original message (A4 + A5 notes)")
play_audio(message, fs)

print("2. AM modulated signal")
play_audio(am_signal, fs)

print("3. AM demodulated signal")
play_audio(am_demod, fs)

print("4. FM modulated signal")
play_audio(fm_signal, fs)

print("5. FM demodulated signal")
play_audio(fm_demod, fs) 