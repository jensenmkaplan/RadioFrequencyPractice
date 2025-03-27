import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Title and Introduction
intro_md = """# AM Radio Modulation and Demodulation Visualization

This notebook demonstrates amplitude modulation (AM) and demodulation of an audio signal, with interactive visualizations of both time and frequency domains.

## Features:
- Load and process audio files
- Generate carrier wave at 100 kHz
- Apply AM modulation with adjustable modulation index
- Demodulate using envelope detection (Hilbert transform)
- Interactive time domain visualizations
- Frequency domain analysis
- Audio playback of original and modulated signals"""

nb.cells.append(nbf.v4.new_markdown_cell(intro_md))

# Import Libraries
import_code = """import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
from pydub import AudioSegment
from ipywidgets import interact, FloatSlider

# Enable inline plotting
%matplotlib inline"""

nb.cells.append(nbf.v4.new_code_cell(import_code))

# Helper Functions
helper_functions = """def load_audio_file(file_path, duration_seconds=5):
    \"\"\"Load and normalize a WAV audio file.\"\"\"
    # Load WAV file using scipy
    fs, samples = wavfile.read(file_path)
    
    # Convert to mono if stereo
    if len(samples.shape) > 1:
        samples = samples.mean(axis=1)
    
    # Calculate number of samples for desired duration
    samples_to_keep = int(duration_seconds * fs)
    
    # Take only the first duration_seconds worth of samples
    samples = samples[:samples_to_keep]
    
    # Convert to float32 and normalize
    samples = samples.astype(np.float32)
    samples = samples / np.max(np.abs(samples))
    
    return fs, samples

def create_carrier(fs, duration, fc, message_length):
    \"\"\"Create a carrier wave with exact same length as message signal.\"\"\"
    t = np.linspace(0, duration, message_length, endpoint=False)
    carrier = np.sin(2 * np.pi * fc * t)
    return t, carrier

def amplitude_modulate(message, carrier, m):
    \"\"\"Apply amplitude modulation to the message signal with improved quality.\"\"\"
    # Create the modulated signal
    modulated = (1 + m * message) * carrier
    
    # Simulate demodulation (envelope detection)
    # Use Hilbert transform to get the envelope
    analytic_signal = np.abs(hilbert(modulated))
    
    # Normalize to prevent clipping
    return analytic_signal / np.max(np.abs(analytic_signal))

def play_audio(signal, fs):
    \"\"\"Play an audio signal using sounddevice.\"\"\"
    sd.play(signal, fs)
    sd.wait()"""

nb.cells.append(nbf.v4.new_code_cell(helper_functions))

# Setup Parameters and Load Audio
setup_code = """# Audio parameters
fs = 44100  # Standard audio sampling rate
fc = 100000  # 100 kHz carrier (simulating radio-like behavior)

# Modulation parameters
modulation_indices = [1.0, 3.0]  # Standard modulation index for AM radio, m = 1, and then an example of overmodulation m=3

# Load audio file (first 5 seconds only)
file_path = "Gabenyeh's Lullaby.wav"  # Replace with your WAV file path
fs, message = load_audio_file(file_path, duration_seconds=5)

# Create carrier wave with exact same length as message
duration = len(message) / fs
t, carrier = create_carrier(fs, duration, fc, len(message))

# Generate AM signals
am_signals = []
for m in modulation_indices:
    am_signal = amplitude_modulate(message, carrier, m)
    am_signals.append(am_signal)

# Normalize signals for audio playback
message = message / np.max(np.abs(message)) * 0.99
for i in range(len(am_signals)):
    am_signals[i] = am_signals[i] / np.max(np.abs(am_signals[i])) * 0.99"""

nb.cells.append(nbf.v4.new_code_cell(setup_code))

# Time Domain Visualization Functions
time_domain_code = """def plot_time_domain(start_time=0, duration=0.5):
    \"\"\"Plot time domain signals for the specified time window.\"\"\"
    start_idx = int(start_time * fs)
    end_idx = int((start_time + duration) * fs)
    
    # Create figure for main plots
    total_subplots = 2 + len(modulation_indices) + 3  # Message, Carrier, AM signals, Hilbert components, Envelope, Demodulated
    fig, axes = plt.subplots(total_subplots, 1, figsize=(12, 3*total_subplots))
    fig.subplots_adjust(hspace=0.4)
    
    # Plot message signal
    axes[0].plot(t[start_idx:end_idx], message[start_idx:end_idx])
    axes[0].set_title(f'Message Signal (Audio) - {start_time:.1f}s to {start_time+duration:.1f}s')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True)
    
    # Plot carrier wave
    axes[1].plot(t[start_idx:end_idx], carrier[start_idx:end_idx])
    axes[1].set_title(f'Carrier Signal ({fc/1000:.1f} kHz) - {start_time:.1f}s to {start_time+duration:.1f}s')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True)
    
    # Plot AM signals with different modulation indices
    for i, (am_signal, m) in enumerate(zip(am_signals, modulation_indices)):
        axes[2+i].plot(t[start_idx:end_idx], am_signal[start_idx:end_idx], label=f'AM Signal (m={m:.1f})')
        envelope = 1 + m * message
        axes[2+i].plot(t[start_idx:end_idx], envelope[start_idx:end_idx], 'r--', label='Envelope')
        axes[2+i].plot(t[start_idx:end_idx], -envelope[start_idx:end_idx], 'r--')
        axes[2+i].set_title(f'AM Signal with Modulation Index m={m:.1f} - {start_time:.1f}s to {start_time+duration:.1f}s')
        axes[2+i].set_xlabel('Time (s)')
        axes[2+i].set_ylabel('Amplitude')
        axes[2+i].legend()
        axes[2+i].grid(True)
    
    # Plot Hilbert transform components
    m = modulation_indices[0]  # Use first modulation index
    modulated = (1 + m * message) * carrier
    analytic_signal = hilbert(modulated)
    
    # Calculate smaller time window for Hilbert components (1/100th of original)
    hilbert_samples = int(duration * fs) // 100
    hilbert_end_idx = min(start_idx + hilbert_samples, end_idx)
    hilbert_duration = (hilbert_end_idx - start_idx) / fs
    
    axes[-3].plot(t[start_idx:hilbert_end_idx], np.real(analytic_signal[start_idx:hilbert_end_idx]), 'b-', label='Real Part')
    axes[-3].plot(t[start_idx:hilbert_end_idx], np.imag(analytic_signal[start_idx:hilbert_end_idx]), 'g-', label='Imaginary Part')
    axes[-3].set_title(f'Hilbert Transform Components - {start_time:.3f}s to {start_time+hilbert_duration:.3f}s')
    axes[-3].set_xlabel('Time (s)')
    axes[-3].set_ylabel('Amplitude')
    axes[-3].legend()
    axes[-3].grid(True)
    
    # Plot magnitude of analytic signal vs true envelope
    demodulated = np.abs(analytic_signal)
    envelope = 1 + m * message
    axes[-2].plot(t[start_idx:end_idx], demodulated[start_idx:end_idx], 'g-', label='|Analytic Signal|')
    axes[-2].plot(t[start_idx:end_idx], envelope[start_idx:end_idx], 'r--', label='True Envelope')
    axes[-2].set_title('Magnitude of Analytic Signal vs True Envelope')
    axes[-2].set_xlabel('Time (s)')
    axes[-2].set_ylabel('Amplitude')
    axes[-2].legend()
    axes[-2].grid(True)
    
    # Plot demodulated vs original
    axes[-1].plot(t[start_idx:end_idx], demodulated[start_idx:end_idx], 'g-', label='Demodulated')
    axes[-1].plot(t[start_idx:end_idx], message[start_idx:end_idx], 'r--', label='Original Message')
    axes[-1].set_title('Demodulated Signal vs Original Message')
    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_ylabel('Amplitude')
    axes[-1].legend()
    axes[-1].grid(True)
    
    plt.tight_layout()
    return fig"""

nb.cells.append(nbf.v4.new_code_cell(time_domain_code))

# Interactive Time Domain Plot
interactive_time_code = """def interactive_time_plot(start_time=0.0, duration=0.5):
    fig = plot_time_domain(start_time, duration)
    plt.show()

interact(interactive_time_plot,
         start_time=FloatSlider(min=0, max=4.5, step=0.1, value=0),
         duration=FloatSlider(min=0.1, max=1.0, step=0.1, value=0.5))"""

nb.cells.append(nbf.v4.new_code_cell(interactive_time_code))

# Frequency Domain Visualization Functions
freq_domain_code = """def plot_frequency_domain(start_time=0, duration=0.5):
    \"\"\"Plot frequency domain signals for the specified time window.\"\"\"
    start_idx = int(start_time * fs)
    end_idx = int((start_time + duration) * fs)
    
    # Calculate FFT parameters
    n_fft = 2**16  # Use a fixed, large FFT size for better frequency resolution
    freqs = fftfreq(n_fft, 1/fs)
    pos_freqs = freqs[:n_fft//2]
    
    # Create figure for spectra
    total_plots = 4  # Message, Carrier, Modulated, Demodulated
    fig, axes = plt.subplots(total_plots, 1, figsize=(12, 3*total_plots))
    fig.subplots_adjust(hspace=0.4)
    
    # Calculate FFTs for message and demodulated (using original fs)
    message_fft = np.abs(fft(message[start_idx:end_idx], n=n_fft))[:n_fft//2]
    
    # Generate modulated signal at original sampling rate for demodulation
    m = modulation_indices[0]
    modulated = (1 + m * message[start_idx:end_idx]) * carrier[start_idx:end_idx]
    analytic_signal = hilbert(modulated)
    demodulated = np.abs(analytic_signal)
    demodulated_fft = np.abs(fft(demodulated, n=n_fft))[:n_fft//2]
    
    # For carrier and modulated spectra, use higher sampling rate
    fs_high = 240000  # 240 kHz sampling rate for carrier visualization (2 × 120 kHz)
    t_high = np.linspace(start_time, start_time + duration, int(duration * fs_high))
    
    # Generate carrier and modulated signals at higher sampling rate
    carrier_high = np.sin(2 * np.pi * fc * t_high)
    message_high = np.interp(t_high, np.linspace(start_time, start_time + duration, len(message[start_idx:end_idx])), message[start_idx:end_idx])
    modulated_high = (1 + m * message_high) * carrier_high
    
    # Calculate FFTs for carrier and modulated (using higher fs)
    n_fft_high = 2**16
    freqs_high = fftfreq(n_fft_high, 1/fs_high)
    pos_freqs_high = freqs_high[:n_fft_high//2]
    
    carrier_fft = np.abs(fft(carrier_high, n=n_fft_high))[:n_fft_high//2]
    modulated_fft = np.abs(fft(modulated_high, n=n_fft_high))[:n_fft_high//2]
    
    # Scale all spectra relative to message spectrum maximum
    max_magnitude = np.max(message_fft)
    message_fft = message_fft / max_magnitude
    carrier_fft = carrier_fft / max_magnitude
    modulated_fft = modulated_fft / max_magnitude
    demodulated_fft = demodulated_fft / max_magnitude
    
    # Plot message spectrum
    axes[0].semilogx(pos_freqs, message_fft, label='Message Spectrum')
    axes[0].set_title('Message Spectrum')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Relative Magnitude')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xlim(10, fs/2)  # Show from 10 Hz to Nyquist frequency
    axes[0].set_ylim(0, 1.2)
    
    # Plot carrier spectrum
    axes[1].plot(pos_freqs_high, carrier_fft, label='Carrier Spectrum')
    axes[1].set_title(f'Carrier Spectrum ({fc/1000:.1f} kHz)')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Relative Magnitude')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xlim(fc-20000, fc+20000)  # Show ±20 kHz around carrier
    axes[1].set_ylim(0, 1.2)
    
    # Plot modulated spectrum
    axes[2].plot(pos_freqs_high, modulated_fft, label='Modulated Spectrum')
    axes[2].set_title('Modulated Spectrum')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Relative Magnitude')
    axes[2].legend()
    axes[2].grid(True)
    axes[2].set_xlim(fc-20000, fc+20000)  # Show ±20 kHz around carrier
    axes[2].set_ylim(0, 1.2)
    
    # Plot demodulated spectrum
    axes[3].semilogx(pos_freqs, demodulated_fft, label='Demodulated Spectrum')
    axes[3].set_title('Demodulated Spectrum')
    axes[3].set_xlabel('Frequency (Hz)')
    axes[3].set_ylabel('Relative Magnitude')
    axes[3].legend()
    axes[3].grid(True)
    axes[3].set_xlim(10, fs/2)  # Show from 10 Hz to Nyquist frequency
    axes[3].set_ylim(0, 1.2)
    
    plt.tight_layout()
    return fig"""

nb.cells.append(nbf.v4.new_code_cell(freq_domain_code))

# Interactive Frequency Domain Plot
interactive_freq_code = """def interactive_freq_plot(start_time=0.0, duration=0.5):
    fig = plot_frequency_domain(start_time, duration)
    plt.show()

interact(interactive_freq_plot,
         start_time=FloatSlider(min=0, max=4.5, step=0.1, value=0),
         duration=FloatSlider(min=0.1, max=1.0, step=0.1, value=0.5))"""

nb.cells.append(nbf.v4.new_code_cell(interactive_freq_code))

# Audio Playback
audio_playback_code = """print(\"Playing original message...\")
play_audio(message, fs)

print(\"\\nPlaying AM signals with different modulation indices:\")
for i, m in enumerate(modulation_indices):
    print(f\"   AM signal with m={m:.1f}\")
    play_audio(am_signals[i], fs)"""

nb.cells.append(nbf.v4.new_code_cell(audio_playback_code))

# Write the notebook to a file
with open('Audio_File_Modulation.ipynb', 'w') as f:
    nbf.write(nb, f) 