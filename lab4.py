import numpy as np
import random as rnd
import matplotlib.pyplot as plt


def create_harmon():
    harmon = [0 for _ in range(vidlik_number)]
    for i in range(number_of_harmonics):
        A = rnd.uniform(min_number, max_number)
        Fi = rnd.uniform(min_number, max_number)
        for t in range(vidlik_number):
            harmon[t] += A*np.sin(frequency/number_of_harmonics*t*i + Fi)
    return harmon


def fft(x: list):
    N = len(x)
    fftt = [[0] * 2 for i in range(N)]
    for i in range(N // 2):
        array1 = [0] * 2
        array2 = [0] * 2
        for j in range(N // 2):
            cos = np.cos(4 * np.pi * i * j / N)
            sin = np.sin(4 * np.pi * i * j / N)
            array1[0] += x[2 * j + 1] * cos  # real
            array1[1] += x[2 * j + 1] * sin  # imagine
            array2[0] += x[2 * j] * cos  # real
            array2[1] += x[2 * j] * sin  # imagine
        cos = np.cos(2 * np.pi * i / N)
        sin = np.sin(2 * np.pi * i / N)
        fftt[i][0] = array2[0] + array1[0] * cos - array1[1] * sin  # real
        fftt[i][1] = array2[1] + array1[0] * sin + array1[1] * cos  # imagine
        fftt[i + N // 2][0] = array2[0] - (array1[0] * cos - array1[1] * sin)  # real
        fftt[i + N // 2][1] = array2[1] - (array1[0] * sin + array1[1] * cos)  # imagine
    return fftt



number_of_harmonics = 10
vidlik_number = 256
frequency = 1500
min_number = 0
max_number = 1

x = create_harmon()
fft = fft(x)
data_fft = [np.sqrt(fft[i][0] ** 2 + fft[i][1] ** 2) for i in range(vidlik_number)]

# fft в з бібліотеки numpy
np_fft = np.fft.fft(x)
data_np_fft = [np.sqrt(np_fft.real[i] ** 2 + np_fft.imag[i] ** 2) for i in range(vidlik_number)]
# Відхилення мого fft від numpy fft
vidhil = [data_fft[i] - data_np_fft[i] for i in range(vidlik_number)]

fig = plt.figure()
ax_1 = fig.add_subplot(2, 1, 1)
ax_2 = fig.add_subplot(2, 1, 2)
ax_1.plot(range(vidlik_number), data_fft)
ax_2.plot(range(vidlik_number), data_np_fft)
ax_1.set(title='myFFT')
ax_2.set(title='npFFT')
plt.show()


plt.plot(range(vidlik_number), vidhil, label="Відхилення")
plt.legend()
plt.show()
