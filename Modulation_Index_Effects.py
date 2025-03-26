import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Audio parameters
fs = 44100  # Standard audio sampling rate
duration = 2  # seconds
t = np.linspace(0, duration, int(fs * duration))

# 1. Generate a simple audio message (a tone that changes frequency)
message = np.sin(2 * np.pi * 440 * t)  # 440 Hz (A4 note)
message = message + .5 * np.sin(2 * np.pi * 880 * t)  # 880 Hz (A5 note)

# 2. Carrier Wave
fc = 2000  # 2 kHz carrier
carrier = np.sin(2 * np.pi * fc * t)

# 3. Amplitude Modulation (AM) with different modulation indices
# AM modulation index (m) bounds: 0 < m ≤ 1
# m = 0: No modulation
# m = 1: 100% modulation (maximum without distortion)
# m > 1: Over-modulation (causes envelope distortion)
modulation_indices = [0.0, 0.5, 1.5]  # No modulation, Under-modulated, Over-modulated
am_signals = []
for m in modulation_indices:
    am_signal = (1 + m * message) * carrier
    am_signals.append(am_signal)

# 4. Frequency Modulation (FM) with different modulation indices
frequency_deviation = 100  # Hz deviation
# FM modulation index (β) bounds: β ≥ 0
# β = 0: No modulation
# β < 1: Narrowband FM
# β > 1: Wideband FM
betas = [0.0, 0.5, 2.0]  # No modulation, Narrowband, Wideband
fm_signals = []
for beta in betas:
    fm_signal = np.sin(2 * np.pi * fc * t + beta * message)
    fm_signals.append(fm_signal)

# Normalize signals for audio playback
message = message / np.max(np.abs(message))
for i in range(len(am_signals)):
    am_signals[i] = am_signals[i] / np.max(np.abs(am_signals[i]))
for i in range(len(fm_signals)):
    fm_signals[i] = fm_signals[i] / np.max(np.abs(fm_signals[i]))

# Function to play audio
def play_audio(signal, fs):
    sd.play(signal, fs)
    sd.wait()

# Function to play all sounds
def play_all_sounds():
    print("\nPlaying signals for comparison...")
    print("1. Original message (A4 + A5 notes)")
    play_audio(message, fs)

    print("\n2. AM signals with different modulation indices:")
    for i, m in enumerate(modulation_indices):
        print(f"   AM signal with m={m:.1f}")
        play_audio(am_signals[i], fs)

    print("\n3. FM signals with different modulation indices:")
    for i, beta in enumerate(betas):
        print(f"   FM signal with β={beta:.1f}")
        play_audio(fm_signals[i], fs)

# Print parameters
print(f"\nAudio Signal Parameters:")
print(f"Message Frequency: 440 Hz (A4) + 880 Hz (A5)")
print(f"Carrier Frequency: {fc/1000:.1f} kHz")
print(f"AM Modulation Indices: {modulation_indices}")
print(f"FM Modulation Indices (β): {betas}")
print(f"FM Frequency Deviation: {frequency_deviation:.1f} Hz")
print(f"Sampling Rate: {fs/1000:.1f} kHz")
print(f"Duration: {duration:.1f} seconds")

# Create the main window and scrollable frame
root = tk.Tk()
root.title("Modulation Index Effects")
root.geometry("1200x800")  # Set initial window size

# Set up protocol handler to play sounds when window is closed
root.protocol("WM_DELETE_WINDOW", lambda: [root.destroy(), play_all_sounds()])

# Create a canvas with scrollbar
canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Create figure with more space between subplots
fig = Figure(figsize=(12, 27), dpi=100)  # Made figure taller for extra subplot
fig.subplots_adjust(hspace=0.4)  # Add more vertical space between subplots

# Plot message signal
ax1 = fig.add_subplot(9, 1, 1)
ax1.plot(t[:2000], message[:2000])
ax1.set_title('Message Signal (Audio)')
ax1.set_xlabel('Time (samples)')
ax1.set_ylabel('Amplitude')
ax1.grid(True)

# Plot carrier wave
ax2 = fig.add_subplot(9, 1, 2)
ax2.plot(t[:2000], carrier[:2000])
ax2.set_title(f'Carrier Signal ({fc/1000:.1f} kHz)')
ax2.set_xlabel('Time (samples)')
ax2.set_ylabel('Amplitude')
ax2.grid(True)

# Plot AM signals with different modulation indices
for i, (am_signal, m) in enumerate(zip(am_signals, modulation_indices)):
    ax = fig.add_subplot(9, 1, 3+i)
    ax.plot(t[:2000], am_signal[:2000], label=f'AM Signal (m={m:.1f})')
    envelope = 1 + m * message
    ax.plot(t[:2000], envelope[:2000], 'r--', label='Envelope')
    ax.plot(t[:2000], -envelope[:2000], 'r--')
    ax.set_title(f'AM Signal with Modulation Index m={m:.1f}')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)

# Plot FM signals with different modulation indices
for i, (fm_signal, beta) in enumerate(zip(fm_signals, betas)):
    ax = fig.add_subplot(9, 1, 6+i)
    ax.plot(t[:2000], fm_signal[:2000], label=f'FM Signal (β={beta:.1f})')
    ax.set_title(f'FM Signal with Modulation Index β={beta:.1f}')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)

# Plot all FM signals overlaid for comparison
ax9 = fig.add_subplot(9, 1, 9)
for fm_signal, beta in zip(fm_signals, betas):
    ax9.plot(t[:2000], fm_signal[:2000], label=f'FM Signal (β={beta:.1f})')
ax9.set_title('All FM Signals Overlaid for Comparison')
ax9.set_xlabel('Time (samples)')
ax9.set_ylabel('Amplitude')
ax9.legend()
ax9.grid(True)

# Create the canvas and add it to the scrollable frame
canvas_widget = FigureCanvasTkAgg(fig, master=scrollable_frame)
canvas_widget.draw()
canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Pack the scrollbar and canvas
scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

# Start the Tkinter event loop
root.mainloop() 