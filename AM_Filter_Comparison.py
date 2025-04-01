import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, butter, cheby1, cheby2, ellip, filtfilt, freqz
from pydub import AudioSegment

def load_audio_file(file_path, duration_seconds=5):
    """Load and normalize a WAV audio file."""
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

def add_noise(signal, noise_level=0.1):
    """Add noise to a signal.
    
    Args:
        signal: Input signal
        noise_level: Noise level relative to signal power (0-1)
    Returns:
        tuple: (noisy_signal, noise_components) where noise_components is a dict containing individual noise types
    """
    # Use signal power for reference if signal is not zero
    if np.any(signal):
        signal_power = np.mean(signal**2)
        noise_power = signal_power * noise_level
    else:
        # For zero signal, use a reference power
        noise_power = noise_level
    
    # Initialize noise array
    noise = np.zeros_like(signal)
    noise_components = {}
    
    # Add real-world noise types
    # 1. Impulse noise (more visible spikes)
    impulse_noise = np.zeros_like(signal)
    num_impulses = int(len(signal) * noise_level * 0.1)  # Increased number of impulses
    impulse_indices = np.random.choice(len(signal), num_impulses, replace=False)
    for idx in impulse_indices:
        burst_length = 10  # Longer burst
        burst_indices = np.arange(idx, min(idx + burst_length, len(signal)))
        taper = np.exp(-np.arange(burst_length) / (burst_length/2))
        impulse_noise[burst_indices] = np.random.choice([-1, 1]) * np.sqrt(noise_power * 100) * taper
    noise += impulse_noise
    noise_components['impulse'] = impulse_noise
    
    # 2. Fading (more pronounced variations)
    t = np.arange(len(signal)) / len(signal)
    fade_rate = 0.2  # Increased fade rate
    fade = 1 + 0.8 * np.sin(2 * np.pi * fade_rate * t)  # Increased amplitude
    fade += 0.4 * np.random.randn(len(signal))  # More random variation
    fade = np.maximum(fade, 0.3)  # Allow deeper fades
    fading_noise = signal * (fade - 1)
    noise += fading_noise
    noise_components['fading'] = fading_noise
    
    # 3. Adjacent channel (stronger interference)
    offset_freq = 0.15  # Increased offset
    carrier = np.sin(2 * np.pi * offset_freq * t)
    modulation = np.random.randn(len(signal))
    modulation = np.convolve(modulation, np.ones(200)/200, mode='same')  # Smoother modulation
    adjacent_noise = carrier * modulation * np.sqrt(noise_power * 2)  # Doubled power
    noise += adjacent_noise
    noise_components['adjacent'] = adjacent_noise
    
    # 4. Multipath (more visible echo)
    delay = int(len(signal) * 0.02)  # Longer delay
    attenuated = 0.5  # Increased amplitude
    multipath = np.zeros_like(signal)
    multipath[delay:] = attenuated * signal[:-delay]
    phase = np.random.uniform(0, 2*np.pi)
    multipath_noise = multipath * np.cos(phase)
    noise += multipath_noise
    noise_components['multipath'] = multipath_noise
    
    print("Adding noise components:", list(noise_components.keys()))
    return signal + noise, noise_components

def butter_bandpass(signal, lowcut, highcut, fs, order=4):
    """Apply a Butterworth bandpass filter."""
    # print("\nButterworth Filter Debug:")
    # print(f"Input signal shape: {signal.shape}")
    # print(f"Input signal range: [{np.min(signal)}, {np.max(signal)}]")
    # print(f"Input signal mean: {np.mean(signal)}")
    # print(f"Input signal std: {np.std(signal)}")
    
    # Check for invalid input
    if not np.all(np.isfinite(signal)):
        print("Warning: Input signal contains infs or NaNs")
        return signal
    
    # Remove DC offset first
    signal = signal - np.mean(signal)
    
    # Normalize the signal
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    else:
        print("Warning: Signal has zero magnitude")
        return signal
    
    # print(f"After normalization - range: [{np.min(signal)}, {np.max(signal)}]")
    
    # Apply bandpass filter
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # print(f"Cutoff frequencies: low={low:.4f}, high={high:.4f}")
    
    # Ensure cutoff frequencies are within valid range
    if low <= 0 or high >= 1:
        print("Warning: Invalid cutoff frequencies")
        return signal
    
    try:
        b, a = butter(order, [low, high], btype='band')
        # print("Filter coefficients created successfully")
        
        filtered = filtfilt(b, a, signal, method='gust')
        # print(f"Filtered signal range: [{np.min(filtered)}, {np.max(filtered)}]")
        
        # Check for invalid output
        if not np.all(np.isfinite(filtered)):
            print("Warning: Filtered signal contains infs or NaNs")
            return signal
        
        # Normalize the filtered signal
        max_val = np.max(np.abs(filtered))
        if max_val > 0:
            filtered = filtered / max_val
        else:
            print("Warning: Filtered signal has zero magnitude")
            return signal
        
        # print(f"Final filtered signal range: [{np.min(filtered)}, {np.max(filtered)}]")
        return filtered
    except Exception as e:
        print(f"Error in filtering: {e}")
        return signal

def cheby1_bandpass(signal, lowcut, highcut, fs, order=4, ripple=1):
    """Apply a Chebyshev Type I bandpass filter."""
    # print("\nChebyshev Type I Filter Debug:")
    # print(f"Input signal shape: {signal.shape}")
    # print(f"Input signal range: [{np.min(signal)}, {np.max(signal)}]")
    # print(f"Input signal mean: {np.mean(signal)}")
    # print(f"Input signal std: {np.std(signal)}")
    
    # Check for invalid input
    if not np.all(np.isfinite(signal)):
        print("Warning: Input signal contains infs or NaNs")
        return signal
    
    # Remove DC offset first
    signal = signal - np.mean(signal)
    
    # Normalize the signal
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    else:
        print("Warning: Signal has zero magnitude")
        return signal
    
    # print(f"After normalization - range: [{np.min(signal)}, {np.max(signal)}]")
    
    # Apply bandpass filter
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # print(f"Cutoff frequencies: low={low:.4f}, high={high:.4f}")
    
    # Ensure cutoff frequencies are within valid range
    if low <= 0 or high >= 1:
        print("Warning: Invalid cutoff frequencies")
        return signal
    
    try:
        b, a = cheby1(order, ripple, [low, high], btype='band')
        # print("Filter coefficients created successfully")
        
        filtered = filtfilt(b, a, signal, method='gust')
        # print(f"Filtered signal range: [{np.min(filtered)}, {np.max(filtered)}]")
        
        # Check for invalid output
        if not np.all(np.isfinite(filtered)):
            print("Warning: Filtered signal contains infs or NaNs")
            return signal
        
        # Normalize the filtered signal
        max_val = np.max(np.abs(filtered))
        if max_val > 0:
            filtered = filtered / max_val
        else:
            print("Warning: Filtered signal has zero magnitude")
            return signal
        
        # print(f"Final filtered signal range: [{np.min(filtered)}, {np.max(filtered)}]")
        return filtered
    except Exception as e:
        print(f"Error in filtering: {e}")
        return signal

def cheby2_bandpass(signal, lowcut, highcut, fs, order=4, stopband=40):
    """Apply a Chebyshev Type II bandpass filter."""
    # Check for invalid input
    if not np.all(np.isfinite(signal)):
        print("Warning: Input signal contains infs or NaNs")
        return signal
    
    # Remove DC offset first
    signal = signal - np.mean(signal)
    
    # Normalize the signal
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    else:
        print("Warning: Signal has zero magnitude")
        return signal
    
    # Apply bandpass filter
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Ensure cutoff frequencies are within valid range
    if low <= 0 or high >= 1:
        print("Warning: Invalid cutoff frequencies")
        return signal
    
    try:
        b, a = cheby2(order, stopband, [low, high], btype='band')
        filtered = filtfilt(b, a, signal, method='gust')
        
        # Check for invalid output
        if not np.all(np.isfinite(filtered)):
            print("Warning: Filtered signal contains infs or NaNs")
            return signal
        
        # Normalize the filtered signal
        max_val = np.max(np.abs(filtered))
        if max_val > 0:
            filtered = filtered / max_val
        else:
            print("Warning: Filtered signal has zero magnitude")
            return signal
        
        return filtered
    except Exception as e:
        print(f"Error in filtering: {e}")
        return signal

def ellip_bandpass(signal, lowcut, highcut, fs, order=4, ripple=1, stopband=20):
    """Apply an Elliptic bandpass filter."""
    # Check for invalid input
    if not np.all(np.isfinite(signal)):
        print("Warning: Input signal contains infs or NaNs")
        return signal
    
    # Remove DC offset first
    signal = signal - np.mean(signal)
    
    # Normalize the signal
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    else:
        print("Warning: Signal has zero magnitude")
        return signal
    
    # Apply bandpass filter
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Ensure cutoff frequencies are within valid range
    if low <= 0 or high >= 1:
        print("Warning: Invalid cutoff frequencies")
        return signal
    
    try:
        b, a = ellip(order, ripple, stopband, [low, high], btype='band')
        filtered = filtfilt(b, a, signal, method='gust')
        
        # Check for invalid output
        if not np.all(np.isfinite(filtered)):
            print("Warning: Filtered signal contains infs or NaNs")
            return signal
        
        # Normalize the filtered signal
        max_val = np.max(np.abs(filtered))
        if max_val > 0:
            filtered = filtered / max_val
        else:
            print("Warning: Filtered signal has zero magnitude")
            return signal
        
        return filtered
    except Exception as e:
        print(f"Error in filtering: {e}")
        return signal

def amplitude_modulate(message, carrier, m, noise_type=None, noise_level=0):
    """Apply amplitude modulation to the message signal with optional noise."""
    modulated = (1 + m * message) * carrier
    
    if noise_level > 0:  # Only check noise_level
        print(f"Adding noise in amplitude_modulate with level {noise_level}")
        modulated, noise_components = add_noise(modulated, noise_level=noise_level)
        print("Noise components added in amplitude_modulate:", list(noise_components.keys()))
    else:
        noise_components = {}
    
    modulated = modulated / np.max(np.abs(modulated))
    demodulated = np.abs(hilbert(modulated))
    demodulated = demodulated / np.max(np.abs(demodulated))
    
    return demodulated, noise_components

def play_audio(signal, fs):
    """Play an audio signal using sounddevice."""
    sd.play(signal, fs)
    sd.wait()

def update_plots(start_time_entry, duration_entry, spectrum_frame, noise_level):
    """Update the plots with the current time window and noise settings."""
    start_time, duration = get_time_window(start_time_entry, duration_entry)
    if start_time is None or duration is None:
        return

    # Calculate sample indices for the time window
    start_idx = int(start_time * fs)
    end_idx = int((start_time + duration) * fs)
    samples_to_plot = end_idx - start_idx

    # Create figure for spectra
    total_plots = 9  # Message, Carrier, Noise, Modulated, Demodulated, Butterworth, Cheby1, Cheby2, Elliptic
    fig = Figure(figsize=(12, 3*total_plots), dpi=100)
    fig.subplots_adjust(hspace=0.4)

    # Use higher sampling rate for spectrum visualization
    fs_plot = 4500000  # 4.5 MHz sampling rate for proper carrier visualization
    t_plot = np.linspace(start_time, start_time + duration, int(duration * fs_plot))
    
    # Generate signals at higher sampling rate for plotting
    carrier_plot = np.sin(2 * np.pi * fc * t_plot)
    message_plot = np.interp(t_plot, t[start_idx:end_idx], message[start_idx:end_idx])
    modulated_plot = (1 + m * message_plot) * carrier_plot
    
    # Add noise if specified
    if noise_level > 0:
        modulated_plot, noise_components = add_noise(modulated_plot, noise_level=noise_level)
    
    # Calculate FFTs
    n_fft = 2**16
    freqs = fftfreq(n_fft, 1/fs)
    freqs_plot = fftfreq(n_fft, 1/fs_plot)
    pos_freqs = freqs[:n_fft//2]
    pos_freqs_plot = freqs_plot[:n_fft//2]

    # Calculate FFTs at appropriate sampling rates
    message_fft = np.abs(fft(message[start_idx:end_idx], n=n_fft))[:n_fft//2]
    carrier_fft = np.abs(fft(carrier_plot, n=n_fft))[:n_fft//2]
    modulated_fft = np.abs(fft(modulated_plot, n=n_fft))[:n_fft//2]
    
    # Calculate noise spectrum separately
    if noise_level > 0:
        # Calculate FFTs for each noise component
        noise_ffts = {}
        for noise_type, noise_component in noise_components.items():
            noise_ffts[noise_type] = np.abs(fft(noise_component, n=n_fft))[:n_fft//2]
        
        # Calculate combined noise spectrum
        noise_signal = sum(noise_components.values())
        noise_fft = np.abs(fft(noise_signal, n=n_fft))[:n_fft//2]
        
        # Normalize all components
        max_val = np.max([np.max(fft) for fft in noise_ffts.values()])
        for name in noise_ffts:
            noise_ffts[name] = noise_ffts[name] / max_val
        noise_fft = noise_fft / max_val
    else:
        noise_fft = np.zeros_like(carrier_fft)
        noise_ffts = {
            'impulse': np.zeros_like(carrier_fft),
            'fading': np.zeros_like(carrier_fft),
            'adjacent': np.zeros_like(carrier_fft),
            'multipath': np.zeros_like(carrier_fft)
        }

    # Generate demodulated signal
    modulated = (1 + m * message[start_idx:end_idx]) * carrier[start_idx:end_idx]
    if noise_level > 0:
        modulated, _ = add_noise(modulated, noise_level=noise_level)
    demodulated = np.abs(hilbert(modulated))
    demodulated_fft = np.abs(fft(demodulated, n=n_fft))[:n_fft//2]
    
    # Apply different filters
    butter_filtered = butter_bandpass(demodulated, 100, 2000, fs, order=4)
    butter_fft = np.abs(fft(butter_filtered, n=n_fft))[:n_fft//2]
    
    # Use more conservative parameters for Chebyshev Type I
    cheby1_filtered = cheby1_bandpass(demodulated, 100, 2000, fs, order=4, ripple=0.5)
    cheby1_fft = np.abs(fft(cheby1_filtered, n=n_fft))[:n_fft//2]
    
    # Use more conservative parameters for Chebyshev Type II
    cheby2_filtered = cheby2_bandpass(demodulated, 100, 2000, fs, order=4, stopband=40)
    cheby2_fft = np.abs(fft(cheby2_filtered, n=n_fft))[:n_fft//2]
    
    # Use more conservative parameters for Elliptic filter
    ellip_filtered = ellip_bandpass(demodulated, 100, 2000, fs, order=4, ripple=0.5, stopband=20)
    ellip_fft = np.abs(fft(ellip_filtered, n=n_fft))[:n_fft//2]
    
    # Normalize spectra
    max_message_magnitude = np.max(message_fft)
    message_fft = message_fft / max_message_magnitude
    carrier_fft = carrier_fft / np.max(carrier_fft)
    modulated_fft = modulated_fft / np.max(modulated_fft)
    demodulated_fft = demodulated_fft / max_message_magnitude
    butter_fft = butter_fft / max_message_magnitude
    cheby1_fft = cheby1_fft / max_message_magnitude
    cheby2_fft = cheby2_fft / max_message_magnitude
    ellip_fft = ellip_fft / max_message_magnitude

    # Plot spectra
    ax1 = fig.add_subplot(total_plots, 1, 1)
    ax1.semilogx(pos_freqs, message_fft, label='Message Spectrum')
    ax1.set_title('Message Spectrum')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Normalized Magnitude')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(10, fs/2)
    ax1.set_ylim(0, 1.2)

    ax2 = fig.add_subplot(total_plots, 1, 2)
    ax2.plot(pos_freqs_plot, carrier_fft, label='Carrier Spectrum')
    ax2.set_title(f'Carrier Spectrum ({fc/1000:.1f} kHz)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Normalized Magnitude')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(fc-100000, fc+100000)
    ax2.set_ylim(0, 1.2)

    ax3 = fig.add_subplot(total_plots, 1, 3)
    if noise_level > 0:
        # Plot each noise component with different colors and transparency
        ax3.plot(pos_freqs_plot, noise_fft, label='Combined Noise', color='black', linewidth=2)
        ax3.plot(pos_freqs_plot, noise_ffts['impulse'], label='Impulse (Lightning)', alpha=0.7, color='red')
        ax3.plot(pos_freqs_plot, noise_ffts['fading'], label='Fading (Signal Strength)', alpha=0.7, color='green')
        ax3.plot(pos_freqs_plot, noise_ffts['adjacent'], label='Adjacent Channel', alpha=0.7, color='purple')
        ax3.plot(pos_freqs_plot, noise_ffts['multipath'], label='Multipath (Echo)', alpha=0.7, color='orange')
    else:
        ax3.plot(pos_freqs_plot, noise_fft, label='No Noise', color='black')
    ax3.set_title('Combined Noise Spectrum Components')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Normalized Magnitude')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True)
    ax3.set_xlim(fc-25000, fc+25000)  # Show ±25 kHz around carrier
    ax3.set_ylim(0, 0.22)  # Set y-axis limit to 0.22

    ax4 = fig.add_subplot(total_plots, 1, 4)
    ax4.plot(pos_freqs_plot, modulated_fft, label='Modulated Spectrum')
    title = 'Modulated Spectrum'
    if noise_level > 0:
        title += f' (SNR: {20*np.log10(1/noise_level):.1f} dB)'
    ax4.set_title(title)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Normalized Magnitude')
    ax4.legend()
    ax4.grid(True)
    ax4.set_xlim(fc-100000, fc+100000)
    ax4.set_ylim(0, 0.1)

    ax5 = fig.add_subplot(total_plots, 1, 5)
    ax5.semilogx(pos_freqs, demodulated_fft, label='Demodulated Spectrum')
    ax5.set_title('Demodulated Spectrum')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Magnitude (relative to message)')
    ax5.legend()
    ax5.grid(True)
    ax5.set_xlim(10, fs/2)
    ax5.set_ylim(0, 1.2)

    ax6 = fig.add_subplot(total_plots, 1, 6)
    ax6.semilogx(pos_freqs, butter_fft, label='Butterworth Filtered')
    ax6.set_title('Butterworth Filtered Spectrum')
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Magnitude (relative to message)')
    ax6.legend()
    ax6.grid(True)
    ax6.set_xlim(10, fs/2)
    ax6.set_ylim(0, 1.2)

    ax7 = fig.add_subplot(total_plots, 1, 7)
    ax7.semilogx(pos_freqs, cheby1_fft, label='Chebyshev Type I Filtered')
    ax7.set_title('Chebyshev Type I Filtered Spectrum (Ripple in Passband)')
    ax7.set_xlabel('Frequency (Hz)')
    ax7.set_ylabel('Magnitude (relative to message)')
    ax7.legend()
    ax7.grid(True)
    ax7.set_xlim(10, fs/2)
    ax7.set_ylim(0, 1.2)

    ax8 = fig.add_subplot(total_plots, 1, 8)
    ax8.semilogx(pos_freqs, cheby2_fft, label='Chebyshev Type II Filtered')
    ax8.set_title('Chebyshev Type II Filtered Spectrum (Ripple in Stopband)')
    ax8.set_xlabel('Frequency (Hz)')
    ax8.set_ylabel('Magnitude (relative to message)')
    ax8.legend()
    ax8.grid(True)
    ax8.set_xlim(10, fs/2)
    ax8.set_ylim(0, 1.2)

    ax9 = fig.add_subplot(total_plots, 1, 9)
    ax9.semilogx(pos_freqs, ellip_fft, label='Elliptic Filtered')
    ax9.set_title('Elliptic Filtered Spectrum (Ripple in Both Bands)')
    ax9.set_xlabel('Frequency (Hz)')
    ax9.set_ylabel('Magnitude (relative to message)')
    ax9.legend()
    ax9.grid(True)
    ax9.set_xlim(10, fs/2)
    ax9.set_ylim(0, 1.2)

    # Update the canvas
    if hasattr(update_plots, 'canvas_widget'):
        update_plots.canvas_widget.get_tk_widget().destroy()
    update_plots.canvas_widget = FigureCanvasTkAgg(fig, master=spectrum_frame)
    update_plots.canvas_widget.draw()
    update_plots.canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def play_all_sounds():
    """Play all generated signals in sequence."""
    print("\nPlaying signals for comparison...")
    print("1. Original message")
    play_audio(message, fs)

    print("\n2. Modulated signal with noise:")
    # Generate modulated signal with noise
    m = 1.0
    modulated = (1 + m * message) * carrier
    noise_level = float(noise_level_entry.get())
    if noise_level > 0:
        print(f"Adding all noise types with level {noise_level}")
        # Add noise to the modulated signal
        modulated, noise_components = add_noise(modulated, noise_level=noise_level)
        print("Noise components added:", list(noise_components.keys()))
        # Store noise components for later use
        play_all_sounds.noise_components = noise_components
    modulated = modulated / np.max(np.abs(modulated)) * 0.99
    play_audio(modulated, fs)

    print("\n3. Demodulated signal:")
    # Generate demodulated signal with noise
    demodulated, _ = amplitude_modulate(message, carrier, m, noise_level=noise_level)
    play_audio(demodulated, fs)

    print("\n4. Butterworth filtered signal:")
    # Apply Butterworth bandpass filter
    butter_filtered = butter_bandpass(demodulated, 100, 2000, fs, order=4)
    butter_filtered = butter_filtered / np.max(np.abs(butter_filtered)) * 0.99
    play_audio(butter_filtered, fs)

    print("\n5. Chebyshev Type I filtered signal:")
    # Apply Chebyshev Type I filter
    cheby1_filtered = cheby1_bandpass(demodulated, 100, 2000, fs, order=4, ripple=0.5)
    cheby1_filtered = cheby1_filtered / np.max(np.abs(cheby1_filtered)) * 0.99
    play_audio(cheby1_filtered, fs)

    print("\n6. Chebyshev Type II filtered signal:")
    # Apply Chebyshev Type II filter
    cheby2_filtered = cheby2_bandpass(demodulated, 100, 2000, fs, order=4, stopband=40)
    cheby2_filtered = cheby2_filtered / np.max(np.abs(cheby2_filtered)) * 0.99
    play_audio(cheby2_filtered, fs)

    print("\n7. Elliptic filtered signal:")
    # Apply Elliptic filter
    ellip_filtered = ellip_bandpass(demodulated, 100, 2000, fs, order=4, ripple=0.5, stopband=20)
    ellip_filtered = ellip_filtered / np.max(np.abs(ellip_filtered)) * 0.99
    play_audio(ellip_filtered, fs)

def on_closing():
    """Handle window closing event."""
    root.withdraw()
    root.after(100, play_and_destroy)

def play_and_destroy():
    """Play sounds and destroy the window."""
    play_all_sounds()
    root.destroy()

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

if __name__ == "__main__":
    # Audio parameters
    fs = 44100  # Standard audio sampling rate
    fc = 1000000  # 1 MHz carrier frequency
    
    # Modulation parameters
    m = 1.0  # Standard modulation index

    # Load audio file
    file_path = "Gabenyeh's Lullaby.wav"
    fs, message = load_audio_file(file_path, duration_seconds=5)
    
    # Create carrier wave
    duration = len(message) / fs
    t, carrier = create_carrier(fs, duration, fc, len(message))
    
    # Create the main window
    root = tk.Tk()
    root.title("AM Filter Comparison")
    root.geometry("1200x800")

    # Create scrollable frame
    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Add noise controls
    noise_frame = tk.Frame(scrollable_frame)
    noise_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    
    # Noise level input
    tk.Label(noise_frame, text="Noise Level (0-1):").pack(side=tk.LEFT, padx=5)
    noise_level_entry = tk.Entry(noise_frame, width=10)
    noise_level_entry.insert(0, "0.1")
    noise_level_entry.pack(side=tk.LEFT, padx=5)

    # Add time window input fields
    start_time, duration = create_time_window_frame(scrollable_frame)

    # Create a button frame at the bottom
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    # Create buttons
    play_button = tk.Button(button_frame, text="Play Sounds and Close", command=play_and_destroy)
    play_button.pack(side=tk.RIGHT)

    # Add update button
    update_button = tk.Button(button_frame, text="Update Plots", 
                            command=lambda: update_plots(start_time, duration, scrollable_frame, 
                                                       float(noise_level_entry.get())))
    update_button.pack(side=tk.RIGHT, padx=5)

    # Initial plot
    update_plots(start_time, duration, scrollable_frame, float(noise_level_entry.get()))

    # Set up protocol handler
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Pack the scrollbar and canvas
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    # Start the Tkinter event loop
    root.mainloop() 