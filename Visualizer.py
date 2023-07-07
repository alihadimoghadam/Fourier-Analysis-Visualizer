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
    return coefficients

def calculate_fourier_transform(input_signal, t):
    frequency = np.fft.fftfreq(len(t), t[1] - t[0])
    transform = np.fft.fft(input_signal)
    return frequency, transform

# Example usage
t = np.linspace(0, 5, 500)  # Time vector
input_signal = np.sin(2 * np.pi * t)  # Example input signal

output_signal = calculate_output(input_signal, impulse_response(t))
fourier_series = calculate_fourier_series(input_signal, t, t[-1], 10)
frequency, fourier_transform = calculate_fourier_transform(input_signal, t)

# Print signals and numbers
print("Input Signal:")
print(input_signal)
print("\nOutput Signal:")
print(output_signal)
print("\nFourier Series Coefficients:")
print(fourier_series)
print("\nFrequency:")
print(frequency)
print("\nFourier Transform:")
print(fourier_transform)

# Save the logs to a file
with open("log.txt", "w") as log_file:
    log_file.write("Input Signal:\n")
    log_file.write(str(input_signal))
    log_file.write("\n\nOutput Signal:\n")
    log_file.write(str(output_signal))
    log_file.write("\n\nFourier Series Coefficients:\n")
    log_file.write(str(fourier_series))
    log_file.write("\n\nFrequency:\n")
    log_file.write(str(frequency))
    log_file.write("\n\nFourier Transform:\n")
    log_file.write(str(fourier_transform))

# Plotting the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t, input_signal)
plt.title('Input Signal')

plt.subplot(2, 2, 2)
plt.plot(t, output_signal)
plt.title('Output Signal')

plt.subplot(2, 2, 3)
plt.stem(range(-10, 11), np.abs(fourier_series))
plt.title('Fourier Series')

plt.subplot(2, 2, 4)
plt.plot(frequency, np.abs(fourier_transform))
plt.title('Fourier Transform')

plt.tight_layout()
plt.show()
