import numpy as np
import matplotlib.pyplot as plt

# Simple RF generation with AM and FM. showing the effect of the envelope and the original carrier. 

# Time and sampling parameters
duration = 0.001  # 1 millisecond duration (longer to see more cycles)
fs = 1e6         # 1 MHz sampling rate
t = np.linspace(0, duration, int(fs * duration))

# 1. Carrier Wave Generation (lower frequency for visibility)
fc = 10000  # 10 kHz carrier (much lower for demonstration)
carrier = np.sin(2 * np.pi * fc * t)

# 2. Message Signal (lower frequency for visibility)
fm = 1000  # 1 kHz message frequency
message = np.sin(2 * np.pi * fm * t)

# 3. Amplitude Modulation (AM)
modulation_index = 0.8  # Increased for more visible effect
am_signal = (1 + modulation_index * message) * carrier

# 4. Frequency Modulation (FM)
frequency_deviation = 2000  # 2 kHz deviation
beta = frequency_deviation / fm  # modulation index for FM
fm_signal = np.sin(2 * np.pi * fc * t + beta * message)

# Plotting
plt.figure(figsize=(15, 12))

# Plot carrier wave
plt.subplot(4, 1, 1)
plt.plot(t[:2000] * 1000, carrier[:2000])  # Convert time to milliseconds
plt.title(f'Carrier Signal ({fc/1000:.1f} kHz)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot message signal
plt.subplot(4, 1, 2)
plt.plot(t[:2000] * 1000, message[:2000])
plt.title(f'Message Signal ({fm/1000:.1f} kHz)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot AM signal with envelope
plt.subplot(4, 1, 3)
plt.plot(t[:2000] * 1000, am_signal[:2000], label='AM Signal')
# Plot the envelope
envelope = 1 + modulation_index * message
plt.plot(t[:2000] * 1000, envelope[:2000], 'r--', label='Envelope')
plt.plot(t[:2000] * 1000, -envelope[:2000], 'r--')
plt.title('Amplitude Modulated Signal (with envelope)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot FM signal with original carrier
plt.subplot(4, 1, 4)
plt.plot(t[:2000] * 1000, fm_signal[:2000], label='FM Signal')
plt.plot(t[:2000] * 1000, carrier[:2000], 'r--', label='Original Carrier')
plt.title('Frequency Modulated Signal (with original carrier)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print some interesting RF parameters
print(f"\nRF Signal Parameters:")
print(f"Carrier Frequency: {fc/1000:.1f} kHz")
print(f"Message Frequency: {fm/1000:.1f} kHz")
print(f"AM Modulation Index: {modulation_index:.2f}")
print(f"FM Deviation: {frequency_deviation/1000:.1f} kHz")
print(f"FM Modulation Index (β): {beta:.2f}")
print(f"Sampling Rate: {fs/1000:.1f} kHz")
print(f"Sample Duration: {duration*1000:.2f} ms")
