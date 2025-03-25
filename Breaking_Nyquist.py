import numpy as np
import matplotlib.pyplot as plt

def generate_signal(fs, duration, fc):
    """Generate signals with different sampling rates"""
    t = np.linspace(0, duration, int(fs * duration))
    carrier = np.sin(2 * np.pi * fc * t)
    return t, carrier

# Signal parameters
duration = 1e-6  # 1 microsecond
fc = 100e6      # 100 MHz carrier frequency

# Generate signals with different sampling rates
fs_correct = 500e6    # Satisfies Nyquist (500 MHz > 2*100 MHz)
fs_minimum = 200e6    # Just meets Nyquist (200 MHz = 2*100 MHz)
fs_violation = 150e6  # Violates Nyquist (150 MHz < 2*100 MHz)

# Generate three versions of the signal
t1, signal1 = generate_signal(fs_correct, duration, fc)
t2, signal2 = generate_signal(fs_minimum, duration, fc)
t3, signal3 = generate_signal(fs_violation, duration, fc)

# Plotting
plt.figure(figsize=(12, 10))

# Plot with correct sampling
plt.subplot(3, 1, 1)
plt.plot(t1[:500] * 1e6, signal1[:500], 'b-', label='Signal')
plt.plot(t1[:500] * 1e6, signal1[:500], 'r.', label='Samples')
plt.title(f'Good Sampling Rate (fs = {fs_correct/1e6} MHz > 2*fc)')
plt.xlabel('Time (μs)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot with minimum sampling
plt.subplot(3, 1, 2)
plt.plot(t2[:200] * 1e6, signal2[:200], 'b-', label='Signal')
plt.plot(t2[:200] * 1e6, signal2[:200], 'r.', label='Samples')
plt.title(f'Minimum Sampling Rate (fs = {fs_minimum/1e6} MHz = 2*fc)')
plt.xlabel('Time (μs)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot with Nyquist violation
plt.subplot(3, 1, 3)
plt.plot(t3[:150] * 1e6, signal3[:150], 'b-', label='Signal')
plt.plot(t3[:150] * 1e6, signal3[:150], 'r.', label='Samples')
plt.title(f'Undersampling (fs = {fs_violation/1e6} MHz < 2*fc) - Aliasing occurs!')
plt.xlabel('Time (μs)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print sampling information
print("\nNyquist Theorem Demonstration:")
print(f"Signal Frequency (fc): {fc/1e6} MHz")
print(f"Nyquist Rate (2*fc): {2*fc/1e6} MHz")
print("\nSampling Rates Used:")
print(f"1. Good Sampling: {fs_correct/1e6} MHz (> 2*fc)")
print(f"2. Minimum Sampling: {fs_minimum/1e6} MHz (= 2*fc)")
print(f"3. Undersampling: {fs_violation/1e6} MHz (< 2*fc)")
print("\nNote: When sampling rate is below Nyquist rate (2*fc),")
print("aliasing occurs, causing the signal to appear at a false lower frequency!")
