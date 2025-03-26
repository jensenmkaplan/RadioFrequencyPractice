import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Time and sampling parameters
duration = 0.001  # 1 millisecond duration (longer to see more cycles)
fs = 1e6         # 1 MHz sampling rate
t = np.linspace(0, duration, int(fs * duration))

# 1. Carrier Wave Generation (lower frequency for visibility)
fc = 10000  # 10 kHz carrier (much lower for demonstration)
carrier = np.sin(2 * np.pi * fc * t)

# 2. Different Message Signals
fm = 1000  # 1 kHz message frequency

# Create different message signals
sine_message = np.sin(2 * np.pi * fm * t)
square_message = signal.square(2 * np.pi * fm * t)
triangle_message = signal.sawtooth(2 * np.pi * fm * t, width=0.5)  # width=0.5 makes it triangular
sawtooth_message = signal.sawtooth(2 * np.pi * fm * t)

# 3. Amplitude Modulation (AM) for different messages
modulation_index = 0.8  # Increased for more visible effect
am_sine = (1 + modulation_index * sine_message) * carrier
am_square = (1 + modulation_index * square_message) * carrier
am_triangle = (1 + modulation_index * triangle_message) * carrier
am_sawtooth = (1 + modulation_index * sawtooth_message) * carrier

# 4. Frequency Modulation (FM) for different messages
frequency_deviation = 2000  # 2 kHz deviation
beta = frequency_deviation / fm  # modulation index for FM
fm_sine = np.sin(2 * np.pi * fc * t + beta * sine_message)
fm_square = np.sin(2 * np.pi * fc * t + beta * square_message)
fm_triangle = np.sin(2 * np.pi * fc * t + beta * triangle_message)
fm_sawtooth = np.sin(2 * np.pi * fc * t + beta * sawtooth_message)

# Plotting
plt.figure(figsize=(15, 15))

# Plot carrier wave
plt.subplot(5, 1, 1)
plt.plot(t[:2000] * 1000, carrier[:2000])
plt.title(f'Carrier Signal ({fc/1000:.1f} kHz)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot different message signals
plt.subplot(5, 1, 2)
plt.plot(t[:2000] * 1000, sine_message[:2000], label='Sine')
plt.plot(t[:2000] * 1000, square_message[:2000], label='Square')
plt.plot(t[:2000] * 1000, triangle_message[:2000], label='Triangle')
plt.plot(t[:2000] * 1000, sawtooth_message[:2000], label='Sawtooth')
plt.title('Different Message Signals')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot AM signals with envelope
plt.subplot(5, 1, 3)
plt.plot(t[:2000] * 1000, am_sine[:2000], label='AM (Sine)')
plt.plot(t[:2000] * 1000, am_square[:2000], label='AM (Square)')
plt.plot(t[:2000] * 1000, am_triangle[:2000], label='AM (Triangle)')
plt.plot(t[:2000] * 1000, am_sawtooth[:2000], label='AM (Sawtooth)')
plt.title('Amplitude Modulated Signals')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot FM signals
plt.subplot(5, 1, 4)
plt.plot(t[:2000] * 1000, fm_sine[:2000], label='FM (Sine)')
plt.plot(t[:2000] * 1000, fm_square[:2000], label='FM (Square)')
plt.plot(t[:2000] * 1000, fm_triangle[:2000], label='FM (Triangle)')
plt.plot(t[:2000] * 1000, fm_sawtooth[:2000], label='FM (Sawtooth)')
plt.plot(t[:2000] * 1000, carrier[:2000], 'k--', label='Original Carrier')
plt.title('Frequency Modulated Signals')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot zoomed view of FM signals
plt.subplot(5, 1, 5)
plt.plot(t[:500] * 1000, fm_sine[:500], label='FM (Sine)')
plt.plot(t[:500] * 1000, fm_square[:500], label='FM (Square)')
plt.plot(t[:500] * 1000, fm_triangle[:500], label='FM (Triangle)')
plt.plot(t[:500] * 1000, fm_sawtooth[:500], label='FM (Sawtooth)')
plt.plot(t[:500] * 1000, carrier[:500], 'k--', label='Original Carrier')
plt.title('Zoomed View of FM Signals')
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
