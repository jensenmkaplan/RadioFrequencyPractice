import numpy as np
import matplotlib.pyplot as plt

# Time and sampling parameters
duration = 1e-6  # 1 microsecond duration
fs = 500e6      # 500 MHz sampling rate (must be > 2 * highest frequency component)
t = np.linspace(0, duration, int(fs * duration))

# 1. Carrier Wave Generation (FM radio frequency)
fc = 100e6  # 100 MHz carrier (typical FM radio frequency)
carrier = np.sin(2 * np.pi * fc * t)

# 2. Message Signal (audio frequency)
fm = 15000  # 15 kHz message frequency (typical audio frequency)
message = np.sin(2 * np.pi * fm * t)

# 3. Amplitude Modulation (AM)
modulation_index = 0.5
am_signal = (1 + modulation_index * message) * carrier

# 4. Frequency Modulation (FM)
# FM deviation for commercial FM radio is typically 75 kHz
frequency_deviation = 75e3  # 75 kHz
beta = frequency_deviation / fm  # modulation index for FM
fm_signal = np.sin(2 * np.pi * fc * t + beta * message)

# Plotting
plt.figure(figsize=(12, 10))

# Plot carrier wave
plt.subplot(4, 1, 1)
plt.plot(t[:500] * 1e6, carrier[:500])  # Convert time to microseconds
plt.title(f'Carrier Signal ({fc/1e6:.1f} MHz)')
plt.xlabel('Time (μs)')
plt.ylabel('Amplitude')

# Plot message signal
plt.subplot(4, 1, 2)
plt.plot(t[:500] * 1e6, message[:500])
plt.title(f'Message Signal ({fm/1e3:.1f} kHz)')
plt.xlabel('Time (μs)')
plt.ylabel('Amplitude')

# Plot AM signal
plt.subplot(4, 1, 3)
plt.plot(t[:500] * 1e6, am_signal[:500])
plt.title('Amplitude Modulated Signal')
plt.xlabel('Time (μs)')
plt.ylabel('Amplitude')

# Plot FM signal
plt.subplot(4, 1, 4)
plt.plot(t[:500] * 1e6, fm_signal[:500])
plt.title('Frequency Modulated Signal')
plt.xlabel('Time (μs)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Print some interesting RF parameters
print(f"\nRF Signal Parameters:")
print(f"Carrier Frequency: {fc/1e6:.1f} MHz")
print(f"Message Frequency: {fm/1e3:.1f} kHz")
print(f"FM Deviation: {frequency_deviation/1e3:.1f} kHz")
print(f"FM Modulation Index (β): {beta:.2f}")
print(f"Sampling Rate: {fs/1e6:.1f} MHz")
print(f"Sample Duration: {duration*1e6:.2f} μs")
