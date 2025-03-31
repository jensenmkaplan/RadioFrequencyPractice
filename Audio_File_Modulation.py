import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, butter, filtfilt
from pydub import AudioSegment

def load_audio_file(file_path, duration_seconds=5):
    """Load and normalize a WAV audio file.
    
    Args:
        file_path: Path to the WAV file
        duration_seconds: Number of seconds to load (default: 5)
    """
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
    """Create a carrier wave with exact same length as message signal."""
    t = np.linspace(0, duration, message_length, endpoint=False)
    carrier = np.sin(2 * np.pi * fc * t)
    return t, carrier

def lowpass_filter(signal, cutoff_freq, fs, order=4):
    """Apply a low-pass filter to the signal."""
    nyquist = fs / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff, btype='low')
    return filtfilt(b, a, signal)

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Apply a band-pass filter to the signal."""
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def amplitude_modulate(message, carrier, m):
    """Apply amplitude modulation to the message signal with improved quality."""
    # Create the modulated signal
    modulated = (1 + m * message) * carrier
    
    # Simulate demodulation (envelope detection)
    # Use Hilbert transform to get the envelope
    analytic_signal = np.abs(hilbert(modulated))
    
    # Normalize to prevent clipping
    return analytic_signal / np.max(np.abs(analytic_signal))

def frequency_modulate(message, t, fc, beta):
    """Apply frequency modulation to the message signal."""
    return np.sin(2 * np.pi * fc * t + beta * message)

def play_audio(signal, fs):
    """Play an audio signal using sounddevice."""
    sd.play(signal, fs)
    sd.wait()

def enhance_spectrum_visualization(ax, signal_type, params):
    """Add annotations to show modulation concepts in spectra."""
    if signal_type == 'am':
        # Add carrier frequency line
        ax.axvline(x=fc, color='r', linestyle='--', alpha=0.5)
        ax.text(fc, ax.get_ylim()[1]*0.9, f'Carrier\n{fc} Hz', 
                horizontalalignment='center', color='r')
        
        # Add bandwidth annotation
        bandwidth = 2 * params['message_freq']  # 2 × highest message frequency
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
        msg_freq = params['message_freq']
        bandwidth = 2 * (freq_dev + msg_freq) * (beta + 1)
        
        # Add bandwidth annotation
        ax.axvspan(fc - bandwidth/2, fc + bandwidth/2, 
                  color='yellow', alpha=0.2)
        ax.text(fc, ax.get_ylim()[0]*1.1, 
                f'Bandwidth ≈ {int(bandwidth)} Hz\nβ={beta:.1f}, Δf={freq_dev} Hz', 
                horizontalalignment='center')

def create_time_window_frame(parent):
    """Create a frame with time window input fields."""
    time_frame = tk.Frame(parent)
    time_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    
    # Start time input
    tk.Label(time_frame, text="Start Time (s):").pack(side=tk.LEFT, padx=5)
    start_time = tk.Entry(time_frame, width=10)
    start_time.insert(0, "0.0")
    start_time.pack(side=tk.LEFT, padx=5)
    
    # Duration input
    tk.Label(time_frame, text="Duration (s):").pack(side=tk.LEFT, padx=5)
    duration = tk.Entry(time_frame, width=10)
    duration.insert(0, "0.5")
    duration.pack(side=tk.LEFT, padx=5)
    
    return start_time, duration

def get_time_window(start_time_entry, duration_entry):
    """Get the time window from input fields."""
    try:
        start_time = float(start_time_entry.get())
        duration = float(duration_entry.get())
        if start_time < 0 or duration <= 0 or start_time + duration > len(t)/fs:
            raise ValueError("Invalid time window")
        return start_time, duration
    except ValueError:
        tk.messagebox.showerror("Error", "Please enter valid numbers for time window")
        return None, None

def show_zoomed_time_plots():
    """Create a new window showing zoomed-in time domain views."""
    zoom_window = tk.Toplevel(root)
    zoom_window.title("Zoomed Time Domain Views")
    zoom_window.geometry("1200x800")

    # Create scrollable frame
    canvas = tk.Canvas(zoom_window)
    scrollbar = tk.Scrollbar(zoom_window, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create figure for zoomed plots
    total_plots = 7  # Message, Carrier, AM signals (all indices), Hilbert components, Envelope, Demodulated
    fig = Figure(figsize=(12, 3*total_plots), dpi=100)
    fig.subplots_adjust(hspace=0.4)

    # Define zoom window
    zoom_start = int(0.1 * fs)  # Start at 0.1 seconds
    zoom_samples = 500  # Show more samples for message and envelope
    zoom_t = t[zoom_start:zoom_start+zoom_samples]

    # Plot message signal
    ax1 = fig.add_subplot(total_plots, 1, 1)
    ax1.plot(zoom_t, message[zoom_start:zoom_start+zoom_samples])
    ax1.set_title('Message Signal (Audio) - Zoomed to 0.1 seconds')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)

    # Plot carrier wave
    ax2 = fig.add_subplot(total_plots, 1, 2)
    # Use fewer samples for carrier to see oscillations
    carrier_zoom = 100
    ax2.plot(t[zoom_start:zoom_start+carrier_zoom], carrier[zoom_start:zoom_start+carrier_zoom])
    ax2.set_title(f'Carrier Signal ({fc/1000:.1f} kHz) - Zoomed to 0.1 seconds')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)

    # Plot AM signals with different modulation indices
    for i, (am_signal, m) in enumerate(zip(am_signals, modulation_indices)):
        ax = fig.add_subplot(total_plots, 1, 3+i)
        ax.plot(zoom_t, am_signal[zoom_start:zoom_start+zoom_samples], label=f'AM Signal (m={m:.1f})')
        envelope = 1 + m * message
        ax.plot(zoom_t, envelope[zoom_start:zoom_start+zoom_samples], 'r--', label='True Envelope')
        ax.plot(zoom_t, -envelope[zoom_start:zoom_start+zoom_samples], 'r--')
        ax.set_title(f'AM Modulated Signal (m={m:.1f}) - Zoomed to 0.1 seconds')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True)

    # Plot Hilbert transform components
    m = modulation_indices[0]  # Use first modulation index
    modulated = (1 + m * message) * carrier
    analytic_signal = hilbert(modulated)
    # Use carrier_zoom samples for Hilbert components to see phase shift clearly
    zoom_t_hilbert = t[zoom_start:zoom_start+carrier_zoom]
    zoom_real = np.real(analytic_signal[zoom_start:zoom_start+carrier_zoom])
    zoom_imag = np.imag(analytic_signal[zoom_start:zoom_start+carrier_zoom])
    
    ax_hilbert = fig.add_subplot(total_plots, 1, 5)
    ax_hilbert.plot(zoom_t_hilbert, zoom_real, 'b-', label='Real Part')
    ax_hilbert.plot(zoom_t_hilbert, zoom_imag, 'g-', label='Imaginary Part')
    ax_hilbert.set_title('Hilbert Transform Components - Zoomed to 0.1 seconds')
    ax_hilbert.set_xlabel('Time (s)')
    ax_hilbert.set_ylabel('Amplitude')
    ax_hilbert.legend()
    ax_hilbert.grid(True)

    # Plot magnitude of analytic signal vs true envelope
    ax_envelope = fig.add_subplot(total_plots, 1, 6)
    demodulated = np.abs(analytic_signal)
    envelope = 1 + m * message
    ax_envelope.plot(zoom_t, demodulated[zoom_start:zoom_start+zoom_samples], 'g-', label='|Analytic Signal|')
    ax_envelope.plot(zoom_t, envelope[zoom_start:zoom_start+zoom_samples], 'r--', label='True Envelope')
    ax_envelope.set_title('Magnitude of Analytic Signal vs True Envelope - Zoomed to 0.1 seconds')
    ax_envelope.set_xlabel('Time (s)')
    ax_envelope.set_ylabel('Amplitude')
    ax_envelope.legend()
    ax_envelope.grid(True)

    # Plot demodulated vs original
    ax_demod = fig.add_subplot(total_plots, 1, 7)
    ax_demod.plot(zoom_t, demodulated[zoom_start:zoom_start+zoom_samples], 'g-', label='Demodulated')
    ax_demod.plot(zoom_t, message[zoom_start:zoom_start+zoom_samples], 'r--', label='Original Message')
    ax_demod.set_title('Demodulated Signal vs Original Message - Zoomed to 0.1 seconds')
    ax_demod.set_xlabel('Time (s)')
    ax_demod.set_ylabel('Amplitude')
    ax_demod.legend()
    ax_demod.grid(True)

    # Create the canvas and add it to the scrollable frame
    canvas_widget = FigureCanvasTkAgg(fig, master=scrollable_frame)
    canvas_widget.draw()
    canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Pack the scrollbar and canvas
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

def on_closing():
    """Handle window closing event."""
    root.withdraw()  # Hide the window immediately
    root.after(100, play_and_destroy)  # Schedule the sound playback and destruction

def play_and_destroy():
    """Play sounds and destroy the window."""
    play_all_sounds()  # Play sounds
    root.destroy()  # Then destroy the window

def play_all_sounds():
    """Play all generated signals in sequence."""
    print("\nPlaying signals for comparison...")
    print("1. Original message")
    play_audio(message, fs)

    #note that the system can't actually produce the modulated signal since 1MHz is much too high.
    #we hear an aliased version of the carrier which is probably 14.7kHz
    print("\n2. Modulated signals with different modulation indices:")
    for i, m in enumerate(modulation_indices):
        print(f"   Modulated signal with m={m:.1f}")
        # Generate modulated signal
        modulated = (1 + m * message) * carrier
        # Normalize for playback
        modulated = modulated / np.max(np.abs(modulated)) * 0.99
        play_audio(modulated, fs)

    print("\n3. Demodulated signals:")
    for i, m in enumerate(modulation_indices):
        print(f"   Demodulated signal with m={m:.1f}")
        # Generate demodulated signal
        am_signal = amplitude_modulate(message, carrier, m)
        play_audio(am_signal, fs)

def update_main_plots(start_time_entry, duration_entry):
    """Update the main window plots with the current time window."""
    start_time, duration = get_time_window(start_time_entry, duration_entry)
    if start_time is None or duration is None:
        return

    # Calculate sample indices for the time window
    start_idx = int(start_time * fs)
    end_idx = int((start_time + duration) * fs)
    samples_to_plot = end_idx - start_idx

    # Create figure for main plots
    total_subplots = 2 + len(modulation_indices) + 3  # Message, Carrier, AM signals, Hilbert components, Envelope, Demodulated
    fig = Figure(figsize=(12, 3*total_subplots), dpi=100)
    fig.subplots_adjust(hspace=0.4)

    # Plot message signal
    ax1 = fig.add_subplot(total_subplots, 1, 1)
    ax1.plot(t[start_idx:end_idx], message[start_idx:end_idx])
    ax1.set_title(f'Message Signal (Audio) - {start_time:.1f}s to {start_time+duration:.1f}s')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)

    # Plot carrier wave
    ax2 = fig.add_subplot(total_subplots, 1, 2)
    ax2.plot(t[start_idx:end_idx], carrier[start_idx:end_idx])
    ax2.set_title(f'Carrier Signal ({fc/1000:.1f} kHz) - {start_time:.1f}s to {start_time+duration:.1f}s')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)

    # Plot AM signals with different modulation indices
    for i, (am_signal, m) in enumerate(zip(am_signals, modulation_indices)):
        ax = fig.add_subplot(total_subplots, 1, 3+i)
        ax.plot(t[start_idx:end_idx], am_signal[start_idx:end_idx], label=f'AM Signal (m={m:.1f})')
        envelope = 1 + m * message
        ax.plot(t[start_idx:end_idx], envelope[start_idx:end_idx], 'r--', label='Envelope')
        ax.plot(t[start_idx:end_idx], -envelope[start_idx:end_idx], 'r--')
        ax.set_title(f'AM Signal with Modulation Index m={m:.1f} - {start_time:.1f}s to {start_time+duration:.1f}s')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True)

    # Plot Hilbert transform components for the first AM signal
    m = modulation_indices[0]  # Use first modulation index
    modulated = (1 + m * message) * carrier
    analytic_signal = hilbert(modulated)
    
    # Calculate smaller time window for Hilbert components (1/100th of original)
    hilbert_samples = samples_to_plot // 100
    hilbert_end_idx = min(start_idx + hilbert_samples, end_idx)
    hilbert_duration = (hilbert_end_idx - start_idx) / fs
    
    # Plot Hilbert components with smaller time window
    ax_hilbert = fig.add_subplot(total_subplots, 1, 3+len(modulation_indices))
    ax_hilbert.plot(t[start_idx:hilbert_end_idx], np.real(analytic_signal[start_idx:hilbert_end_idx]), 'b-', label='Real Part')
    ax_hilbert.plot(t[start_idx:hilbert_end_idx], np.imag(analytic_signal[start_idx:hilbert_end_idx]), 'g-', label='Imaginary Part')
    ax_hilbert.set_title(f'Hilbert Transform Components - {start_time:.3f}s to {start_time+hilbert_duration:.3f}s')
    ax_hilbert.set_xlabel('Time (s)')
    ax_hilbert.set_ylabel('Amplitude')
    ax_hilbert.legend()
    ax_hilbert.grid(True)

    # Plot magnitude of analytic signal vs true envelope
    ax_envelope = fig.add_subplot(total_subplots, 1, 4+len(modulation_indices))
    demodulated = np.abs(analytic_signal)
    envelope = 1 + m * message
    ax_envelope.plot(t[start_idx:end_idx], demodulated[start_idx:end_idx], 'g-', label='|Analytic Signal|')
    ax_envelope.plot(t[start_idx:end_idx], envelope[start_idx:end_idx], 'r--', label='True Envelope')
    ax_envelope.set_title('Magnitude of Analytic Signal vs True Envelope')
    ax_envelope.set_xlabel('Time (s)')
    ax_envelope.set_ylabel('Amplitude')
    ax_envelope.legend()
    ax_envelope.grid(True)

    # Plot demodulated vs original
    ax_demod = fig.add_subplot(total_subplots, 1, 5+len(modulation_indices))
    ax_demod.plot(t[start_idx:end_idx], demodulated[start_idx:end_idx], 'g-', label='Demodulated')
    ax_demod.plot(t[start_idx:end_idx], message[start_idx:end_idx], 'r--', label='Original Message')
    ax_demod.set_title('Demodulated Signal vs Original Message')
    ax_demod.set_xlabel('Time (s)')
    ax_demod.set_ylabel('Amplitude')
    ax_demod.legend()
    ax_demod.grid(True)

    # Update the canvas
    if hasattr(update_main_plots, 'canvas_widget'):
        update_main_plots.canvas_widget.get_tk_widget().destroy()
    update_main_plots.canvas_widget = FigureCanvasTkAgg(fig, master=scrollable_frame)
    update_main_plots.canvas_widget.draw()
    update_main_plots.canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def show_spectrum_window():
    """Create a new window showing frequency spectra of all signals."""
    spectrum_window = tk.Toplevel(root)
    spectrum_window.title("Frequency Spectra")
    spectrum_window.geometry("1200x800")

    # Create scrollable frame
    canvas = tk.Canvas(spectrum_window)
    scrollbar = tk.Scrollbar(spectrum_window, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Add bandwidth control
    control_frame = tk.Frame(scrollable_frame)
    control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    # Add bandwidth control
    bw_frame = tk.Frame(control_frame)
    bw_frame.pack(side=tk.LEFT, padx=5)
    tk.Label(bw_frame, text="Bandwidth (kHz):").pack(side=tk.LEFT)
    bw_scale = tk.Scale(bw_frame, from_=1, to=20, orient=tk.HORIZONTAL, length=200)
    bw_scale.set(5)  # Default to 5 kHz
    bw_scale.pack(side=tk.LEFT, padx=5)
    
    # Add bandwidth text box
    bw_entry = tk.Entry(bw_frame, width=10)
    bw_entry.insert(0, "5.0")  # Default to 5 kHz
    bw_entry.pack(side=tk.LEFT, padx=5)
    
    # Function to sync slider and text box
    def update_bw_scale(*args):
        try:
            value = float(bw_entry.get())
            if 1 <= value <= 20:
                bw_scale.set(value)
                update_spectrum_plots(start_time, duration, scrollable_frame, value)
        except ValueError:
            pass
    
    def update_bw_entry(*args):
        bw_entry.delete(0, tk.END)
        bw_entry.insert(0, f"{bw_scale.get():.1f}")
    
    # Bind events to sync slider and text box
    bw_entry.bind('<Return>', update_bw_scale)
    bw_scale.config(command=update_bw_entry)

    # Add time window input fields
    start_time, duration = create_time_window_frame(scrollable_frame)

    # Create a button frame at the bottom of the spectrum window
    button_frame = tk.Frame(spectrum_window)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    # Add update button
    update_button = tk.Button(button_frame, text="Update Plots", 
                            command=lambda: update_spectrum_plots(start_time, duration, scrollable_frame, float(bw_entry.get())))
    update_button.pack(side=tk.RIGHT, padx=5)

    # Initial plot
    update_spectrum_plots(start_time, duration, scrollable_frame, float(bw_entry.get()))

    # Pack the scrollbar and canvas
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

def update_spectrum_plots(start_time_entry, duration_entry, spectrum_frame, bandwidth_khz):
    """Update the spectrum plots with the current time window."""
    start_time, duration = get_time_window(start_time_entry, duration_entry)
    if start_time is None or duration is None:
        return

    # Calculate sample indices for the time window
    start_idx = int(start_time * fs)
    end_idx = int((start_time + duration) * fs)
    samples_to_plot = end_idx - start_idx

    # Calculate FFT parameters
    n_fft = 2**16  # Use a fixed, large FFT size for better frequency resolution
    freqs = fftfreq(n_fft, 1/fs)
    pos_freqs = freqs[:n_fft//2]

    # Create figure for spectra
    total_plots = 4  # Message, Carrier, Modulated, Demodulated
    fig = Figure(figsize=(12, 3*total_plots), dpi=100)
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
    fs_high = 4500000  # 4.5 MHz sampling rate for carrier visualization
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
    
    # Calculate maximum value of message spectrum for scaling
    max_message_magnitude = np.max(message_fft)
    
    # Normalize spectra
    message_fft = message_fft / max_message_magnitude
    carrier_fft = carrier_fft / np.max(carrier_fft)
    modulated_fft = modulated_fft / np.max(modulated_fft)
    demodulated_fft = demodulated_fft / max_message_magnitude  # Scale relative to message spectrum

    # Plot message spectrum
    ax1 = fig.add_subplot(total_plots, 1, 1)
    ax1.semilogx(pos_freqs, message_fft, label='Message Spectrum')
    ax1.set_title('Message Spectrum')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Normalized Magnitude')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(10, fs/2)  # Show from 10 Hz to Nyquist frequency
    ax1.set_ylim(0, 1.2)

    # Plot carrier spectrum
    ax2 = fig.add_subplot(total_plots, 1, 2)
    ax2.plot(pos_freqs_high, carrier_fft, label='Carrier Spectrum')
    ax2.set_title(f'Carrier Spectrum ({fc/1000:.1f} kHz)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Normalized Magnitude')
    ax2.legend()
    ax2.grid(True)
    bandwidth_hz = bandwidth_khz * 1000
    ax2.set_xlim(fc-bandwidth_hz, fc+bandwidth_hz)  # Show ±bandwidth around carrier
    ax2.set_ylim(0, 1.2)

    # Plot modulated spectrum
    ax3 = fig.add_subplot(total_plots, 1, 3)
    ax3.plot(pos_freqs_high, modulated_fft, label='Modulated Spectrum')
    ax3.set_title('Modulated Spectrum')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Normalized Magnitude')
    ax3.legend()
    ax3.grid(True)
    ax3.set_xlim(fc-bandwidth_hz, fc+bandwidth_hz)  # Show ±bandwidth around carrier
    ax3.set_ylim(0, 1.2)

    # Plot demodulated spectrum
    ax4 = fig.add_subplot(total_plots, 1, 4)
    ax4.semilogx(pos_freqs, demodulated_fft, label='Demodulated Spectrum')
    ax4.set_title('Demodulated Spectrum')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude (relative to message)')
    ax4.legend()
    ax4.grid(True)
    ax4.set_xlim(10, fs/2)  # Show from 10 Hz to Nyquist frequency
    ax4.set_ylim(0, 1.2)  # Use same scale as message spectrum

    # Update the canvas
    if hasattr(update_spectrum_plots, 'canvas_widget'):
        update_spectrum_plots.canvas_widget.get_tk_widget().destroy()
    update_spectrum_plots.canvas_widget = FigureCanvasTkAgg(fig, master=spectrum_frame)
    update_spectrum_plots.canvas_widget.draw()
    update_spectrum_plots.canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    # Audio parameters
    fs = 44100  # Standard audio sampling rate
    fc = 1000000  # 1000 kHz (1 MHz) carrier frequency - typical AM radio frequency
    
    # Modulation parameters
    modulation_indices = [1.0, 3.0]  # Standard modulation index for AM radio, m = 1, and then an example of overmodulation m=3 where distortion occurs

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
        am_signals[i] = am_signals[i] / np.max(np.abs(am_signals[i])) * 0.99

    # Create the main window
    root = tk.Tk()
    root.title("AM Modulation Visualization")
    root.geometry("1200x800")

    # Create scrollable frame for the main window
    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Add time window input fields to main window
    start_time, duration = create_time_window_frame(scrollable_frame)

    # Create a button frame at the bottom
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    # Create buttons
    play_button = tk.Button(button_frame, text="Play Sounds and Close", command=play_and_destroy)
    play_button.pack(side=tk.RIGHT)

    # Add spectrum button
    spectrum_button = tk.Button(button_frame, text="Show Frequency Spectra", command=show_spectrum_window)
    spectrum_button.pack(side=tk.RIGHT, padx=5)

    # Add update button
    update_button = tk.Button(button_frame, text="Update Plots", 
                            command=lambda: update_main_plots(start_time, duration))
    update_button.pack(side=tk.RIGHT, padx=5)

    # Initial plot
    update_main_plots(start_time, duration)

    # Set up protocol handler to play sounds when window is closed
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Pack the scrollbar and canvas
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    # Start the Tkinter event loop
    root.mainloop() 