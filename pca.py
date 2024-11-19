
# FFT + Principal Compenent Analysis
import cv2
import numpy as np
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt, medfilt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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


def apply_pca(signal):
    """
    Apply PCA to the signal to extract the most dominant component.
    """
    pca = PCA(n_components=1)
    signal_reshaped = signal.reshape(-1, 1)  
    pca_signal = pca.fit_transform(signal_reshaped)
    return pca_signal.flatten()  


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
    pca_signal = apply_pca(signal)

    # Directly apply bandpass filter to PCA output
    filtered_signal = bandpass_filter(pca_signal, lowcut=0.8, highcut=3.5, fps=fps)
    smoothed_signal = medfilt(filtered_signal, kernel_size=5)

    # Perform FFT on the filtered signal
    n = len(smoothed_signal)
    freqs = np.fft.fftfreq(n, d=1 / fps)
    fft_values = np.abs(fft(smoothed_signal - np.mean(smoothed_signal)))

    mask = (freqs > 0.8) & (freqs < 3.5)
    filtered_freqs = freqs[mask]
    filtered_fft_values = fft_values[mask]

    dominant_freq = filtered_freqs[np.argmax(filtered_fft_values)]
    heart_rate = dominant_freq * 60  # Convert Hz to BPM

    return heart_rate, filtered_freqs, filtered_fft_values, filtered_signal, signal, pca_signal


def plot_results(freqs, fft_values, heart_rate, signal, filtered_signal, pca_signal, signal_before_pca):
    """
    Plot the frequency spectrum and the raw/filtered signal, including the PCA result.
    """
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(signal_before_pca, label='Original Signal (Before PCA)')
    plt.plot(pca_signal, label='Signal After PCA')
    plt.legend()
    plt.title("Signal Before and After PCA")
    plt.xlabel('Time (Frames)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.plot(pca_signal, label='PCA Signal')
    plt.plot(filtered_signal, label='Filtered Signal')
    plt.legend()
    plt.title("Signal Before and After Filtering (Band Pass Filter)")
    plt.xlabel('Time (Frames)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
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
heart_rate, freqs, fft_values, filtered_signal, signal_before_pca, pca_signal = analyze_heart_rate(signal, fps)
plot_results(freqs, fft_values, heart_rate, signal, filtered_signal, pca_signal, signal_before_pca)
print(f"Estimated Heart Rate: {heart_rate:.2f} BPM")
