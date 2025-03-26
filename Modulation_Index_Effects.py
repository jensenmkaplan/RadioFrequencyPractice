import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.fft import fft, fftfreq

# User-defined functions
########################################################
def play_audio(signal, fs):
    """Play an audio signal using sounddevice."""
    sd.play(signal, fs)
    sd.wait()

def play_all_sounds():
    """Play all generated signals in sequence."""
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

def on_closing():
    """Handle window closing event."""
    root.withdraw()  # Hide the window immediately
    root.after(100, play_and_destroy)  # Schedule the sound playback and destruction

def play_and_destroy():
    """Play sounds and destroy the window."""
    play_all_sounds()  # Play sounds
    root.destroy()  # Then destroy the window

def enhance_spectrum_visualization(ax, signal_type, params):
    """Add annotations to show modulation concepts in spectra.
    
    Args:
        ax: Matplotlib axis to annotate
        signal_type: 'am' or 'fm'
        params: Dictionary containing relevant parameters
    """
    if signal_type == 'am':
        # Add carrier frequency line
        ax.axvline(x=fc, color='r', linestyle='--', alpha=0.5)
        ax.text(fc, ax.get_ylim()[1]*0.9, f'Carrier\n{fc} Hz', 
                horizontalalignment='center', color='r')
        
        # Add sidebands for each message frequency
        for f in [440, 880]:  # A4 and A5 frequencies
            # Upper sideband
            ax.axvline(x=fc + f, color='g', linestyle='--', alpha=0.5)
            ax.text(fc + f, ax.get_ylim()[1]*0.8, f'USB\n+{f} Hz', 
                    horizontalalignment='center', color='g')
            # Lower sideband
            ax.axvline(x=fc - f, color='g', linestyle='--', alpha=0.5)
            ax.text(fc - f, ax.get_ylim()[1]*0.8, f'LSB\n-{f} Hz', 
                    horizontalalignment='center', color='g')
        
        # Add bandwidth annotation
        bandwidth = 2 * 880  # 2 × highest message frequency
        ax.axvspan(fc - bandwidth/2, fc + bandwidth/2, 
                  color='yellow', alpha=0.2)
        ax.text(fc, ax.get_ylim()[0]*1.1, 
                f'Bandwidth = {bandwidth} Hz', 
                horizontalalignment='center')
    
    elif signal_type == 'fm':
        # Add carrier frequency line
        ax.axvline(x=fc, color='r', linestyle='--', alpha=0.5)
        ax.text(fc, ax.get_ylim()[1]*0.9, f'Carrier\n{fc} Hz', 
                horizontalalignment='center', color='r')
        
        # Calculate and show bandwidth
        beta = params['beta']
        freq_dev = params['freq_dev']
        msg_freq = 880  # Highest message frequency
        bandwidth = 2 * (freq_dev + msg_freq) * (beta + 1)
        
        # Add bandwidth annotation
        ax.axvspan(fc - bandwidth/2, fc + bandwidth/2, 
                  color='yellow', alpha=0.2)
        ax.text(fc, ax.get_ylim()[0]*1.1, 
                f'Bandwidth ≈ {int(bandwidth)} Hz\nβ={beta:.1f}, Δf={freq_dev} Hz', 
                horizontalalignment='center')

def show_spectrum_window():
    """Create a new window showing frequency spectra of all signals."""
    spectrum_window = tk.Toplevel(root)
    spectrum_window.title("Frequency Spectra")
    spectrum_window.geometry("1200x800")

    # Create scrollable frame for the new window
    canvas = tk.Canvas(spectrum_window)
    scrollbar = tk.Scrollbar(spectrum_window, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Calculate FFT parameters
    n_fft = len(t)  # Use full signal length for FFT
    freqs = fftfreq(n_fft, 1/fs)
    pos_freqs = freqs[:n_fft//2]  # Only positive frequencies

    # Create figure for spectra
    total_plots = 2 + len(modulation_indices) + len(betas) + 1  # Same as main window
    fig = Figure(figsize=(12, 3*total_plots), dpi=100)
    fig.subplots_adjust(hspace=0.4)

    # Plot message spectrum
    ax1 = fig.add_subplot(total_plots, 1, 1)
    message_fft = np.abs(fft(message))[:n_fft//2]
    ax1.plot(pos_freqs, message_fft)
    ax1.set_title('Message Signal Spectrum')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude')
    ax1.grid(True)
    ax1.set_xlim(0, 2000)  # Focus on relevant frequency range

    # Plot carrier spectrum
    ax2 = fig.add_subplot(total_plots, 1, 2)
    carrier_fft = np.abs(fft(carrier))[:n_fft//2]
    ax2.plot(pos_freqs, carrier_fft)
    ax2.set_title('Carrier Signal Spectrum')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.grid(True)
    ax2.set_xlim(0, 2500)  # Focus on carrier frequency range

    # Plot AM signal spectra
    for i, (am_signal, m) in enumerate(zip(am_signals, modulation_indices)):
        ax = fig.add_subplot(total_plots, 1, 3+i)
        am_fft = np.abs(fft(am_signal))[:n_fft//2]
        ax.plot(pos_freqs, am_fft)
        ax.set_title(f'AM Signal Spectrum (m={m:.1f})')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.grid(True)
        ax.set_xlim(0, 2500)  # Focus on relevant frequency range
        
        # Add AM visualization (comment out to disable)
        enhance_spectrum_visualization(ax, 'am', {'m': m})

    # Plot FM signal spectra
    for i, (fm_signal, beta) in enumerate(zip(fm_signals, betas)):
        ax = fig.add_subplot(total_plots, 1, 3+len(modulation_indices)+i)
        fm_fft = np.abs(fft(fm_signal))[:n_fft//2]
        ax.plot(pos_freqs, fm_fft)
        ax.set_title(f'FM Signal Spectrum (β={beta:.1f})')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.grid(True)
        ax.set_xlim(0, 2500)  # Focus on relevant frequency range
        
        # Add FM visualization (comment out to disable)
        enhance_spectrum_visualization(ax, 'fm', {'beta': beta, 'freq_dev': frequency_deviation})

    # Plot all FM spectra overlaid for comparison
    ax_final = fig.add_subplot(total_plots, 1, total_plots)
    for fm_signal, beta in zip(fm_signals, betas):
        fm_fft = np.abs(fft(fm_signal))[:n_fft//2]
        ax_final.plot(pos_freqs, fm_fft, label=f'FM Signal (β={beta:.1f})')
    ax_final.set_title('All FM Signal Spectra Overlaid')
    ax_final.set_xlabel('Frequency (Hz)')
    ax_final.set_ylabel('Magnitude')
    ax_final.legend()
    ax_final.grid(True)
    ax_final.set_xlim(0, 2500)  # Focus on relevant frequency range

    # Create the canvas and add it to the scrollable frame
    canvas_widget = FigureCanvasTkAgg(fig, master=scrollable_frame)
    canvas_widget.draw()
    canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Pack the scrollbar and canvas
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

########################################################    




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
modulation_indices = [0.0, 0.5, 1.5, 5.0]  # No modulation, Under-modulated, Over-modulated
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
betas = [0.0, 0.5, 2.0, 3.0]  # No modulation, Narrowband, Wideband
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

# Create a button frame at the bottom
button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

# Create buttons
play_button = tk.Button(button_frame, text="Play Sounds and Close", command=play_and_destroy)
play_button.pack(side=tk.RIGHT)

spectrum_button = tk.Button(button_frame, text="Show Frequency Spectra", command=show_spectrum_window)
spectrum_button.pack(side=tk.RIGHT, padx=5)

# Set up protocol handler to play sounds when window is closed
root.protocol("WM_DELETE_WINDOW", on_closing)

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
# Calculate total number of subplots needed
total_subplots = 2 + len(modulation_indices) + len(betas) + 1  # message + carrier + AM signals + FM signals + FM overlay
fig = Figure(figsize=(12, 3*total_subplots), dpi=100)  # Adjust height based on number of subplots
fig.subplots_adjust(hspace=0.4)  # Add more vertical space between subplots

# Plot message signal
ax1 = fig.add_subplot(total_subplots, 1, 1)
ax1.plot(t[:2000], message[:2000])
ax1.set_title('Message Signal (Audio)')
ax1.set_xlabel('Time (samples)')
ax1.set_ylabel('Amplitude')
ax1.grid(True)

# Plot carrier wave
ax2 = fig.add_subplot(total_subplots, 1, 2)
ax2.plot(t[:2000], carrier[:2000])
ax2.set_title(f'Carrier Signal ({fc/1000:.1f} kHz)')
ax2.set_xlabel('Time (samples)')
ax2.set_ylabel('Amplitude')
ax2.grid(True)

# Plot AM signals with different modulation indices
for i, (am_signal, m) in enumerate(zip(am_signals, modulation_indices)):
    ax = fig.add_subplot(total_subplots, 1, 3+i)
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
    ax = fig.add_subplot(total_subplots, 1, 3+len(modulation_indices)+i)
    ax.plot(t[:2000], fm_signal[:2000], label=f'FM Signal (β={beta:.1f})')
    ax.set_title(f'FM Signal with Modulation Index β={beta:.1f}')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)

# Plot all FM signals overlaid for comparison
ax_final = fig.add_subplot(total_subplots, 1, total_subplots)
for fm_signal, beta in zip(fm_signals, betas):
    # Show only 500 samples instead of 2000 for better visibility
    ax_final.plot(t[:500], fm_signal[:500], label=f'FM Signal (β={beta:.1f})')
ax_final.set_title('All FM Signals Overlaid for Comparison (First 500 samples)')
ax_final.set_xlabel('Time (samples)')
ax_final.set_ylabel('Amplitude')
ax_final.legend()
ax_final.grid(True)

# Create the canvas and add it to the scrollable frame
canvas_widget = FigureCanvasTkAgg(fig, master=scrollable_frame)
canvas_widget.draw()
canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Pack the scrollbar and canvas
scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

# Start the Tkinter event loop
root.mainloop() 