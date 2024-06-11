import numpy as np
import matplotlib.pyplot as plt

def impulse_response(t):
    return np.heaviside(t, 1) * np.exp(-t)

def calculate_output(input_signal, impulse_response):
    output_signal = np.convolve(input_signal, impulse_response, mode='full')
    return output_signal[:len(input_signal)]

def calculate_fourier_series(input_signal, t, T, num_harmonics):
    coefficients = []
    for n in range(-num_harmonics, num_harmonics + 1):
        coefficient = np.sum(input_signal * np.exp(-1j * 2 * np.pi * n * t / T)) / T
        coefficients.append(coefficient)
    return np.array(coefficients)

def calculate_fourier_transform(input_signal, t):
    frequency = np.fft.fftfreq(len(t), t[1] - t[0])
    transform = np.fft.fft(input_signal)
    return frequency, transform

def log_data(filename, data_dict):
    with open(filename, "w") as log_file:
        for key, value in data_dict.items():
            log_file.write(f"{key}:\n{value}\n\n")

def plot_results(t, input_signal, output_signal, fourier_series, frequency, fourier_transform):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(t, input_signal)
    plt.title('Input Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.subplot(2, 2, 2)
    plt.plot(t, output_signal)
    plt.title('Output Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.subplot(2, 2, 3)
    plt.stem(range(-len(fourier_series)//2, len(fourier_series)//2 + 1), np.abs(fourier_series), basefmt=" ")
    plt.title('Fourier Series Coefficients')
    plt.xlabel('Harmonics')
    plt.ylabel('Magnitude')

    plt.subplot(2, 2, 4)
    plt.plot(frequency, np.abs(fourier_transform))
    plt.title('Fourier Transform')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.show()

def main():
    t = np.linspace(0, 5, 500)  # Time vector
    input_signal = np.sin(2 * np.pi * t)  # Example input signal

    # Calculate output signal and transforms
    impulse_resp = impulse_response(t)
    output_signal = calculate_output(input_signal, impulse_resp)
    fourier_series = calculate_fourier_series(input_signal, t, t[-1], 10)
    frequency, fourier_transform = calculate_fourier_transform(input_signal, t)

    # Log the results
    data_to_log = {
        "Input Signal": input_signal,
        "Output Signal": output_signal,
        "Fourier Series Coefficients": fourier_series,
        "Frequency": frequency,
        "Fourier Transform": fourier_transform
    }
    log_data("log.txt", data_to_log)

    # Plot the results
    plot_results(t, input_signal, output_signal, fourier_series, frequency, fourier_transform)

if __name__ == "__main__":
    main()
