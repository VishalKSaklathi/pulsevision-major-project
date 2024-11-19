#Fast Fourier Transform
import cv2
import numpy as np
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt, medfilt
import matplotlib.pyplot as plt

def extract_signal(video_path):
    """
    Extracts the average green channel intensity from the forehead ROI.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) 
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    signal = []

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Extract ROI (forehead area)
        roi = frame[int(height * 0.2):int(height * 0.35), int(width * 0.3):int(width * 0.7)] 
        
        green_channel = roi[:, :, 1]  # Extract green channel
        signal.append(np.mean(green_channel))  # Average green intensity

    cap.release()
    cv2.destroyAllWindows()
    return np.array(signal), fps


def bandpass_filter(data, lowcut, highcut, fps):
    """
    Apply a bandpass filter to the signal.
    """
    nyquist = 0.5 * fps
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype='band')
    return filtfilt(b, a, data)


def analyze_heart_rate(signal, fps):
    """
    Analyze the heart rate from the signal using FFT.
    """
    smoothed_signal = medfilt(signal, kernel_size=5)

    filtered_signal = bandpass_filter(smoothed_signal, lowcut=0.8, highcut=3.5, fps=fps)

    # Perform FFT on the filtered signal
    n = len(filtered_signal)
    freqs = np.fft.fftfreq(n, d=1 / fps)
    fft_values = np.abs(fft(filtered_signal - np.mean(filtered_signal)))

    mask = (freqs > 0.8) & (freqs < 3.5)
    filtered_freqs = freqs[mask]
    filtered_fft_values = fft_values[mask]

    dominant_freq = filtered_freqs[np.argmax(filtered_fft_values)]
    heart_rate = dominant_freq * 60  # Convert Hz to BPM

    return heart_rate, filtered_freqs, filtered_fft_values, filtered_signal


def plot_results(freqs, fft_values, heart_rate, signal, filtered_signal):
    """
    Plot the frequency spectrum and the raw/filtered signal.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal, label='Raw Signal')
    plt.plot(filtered_signal, label='Filtered Signal')
    plt.legend()
    plt.title("Signal Before and After Filtering(Band Pass Filter)")
    plt.xlabel('Time (Frames)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(freqs * 60, fft_values, label='FFT Spectrum')
    plt.axvline(x=heart_rate, color='r', linestyle='--', label=f'Heart Rate: {heart_rate:.2f} BPM')
    plt.xlabel('Frequency (BPM)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# Main 
video_path = r'C:\\Users\\visha\\OneDrive\\Pictures\\Camera Roll\\pra3.mp4' 
signal, fps = extract_signal(video_path)
heart_rate, freqs, fft_values, filtered_signal = analyze_heart_rate(signal, fps)
plot_results(freqs, fft_values, heart_rate, signal, filtered_signal)
print(f"Estimated Heart Rate: {heart_rate:.2f} BPM")
