from __future__ import division
import os
import numpy as np
from scipy.fftpack import fft
import operator
import librosa
import peakutils
from scipy import signal



class AudioTool:

    def __init__(self, sys='fft', n_fft=4096, win_len=1024, hop=1024, path='samples', extension='wav', s_rate=None, s_list=None, fc=None):
        self.n_fft = n_fft
        self.sys = sys
        self.win_len = win_len
        self.hop = hop
        self.path = path
        self.extension = extension
        self.s_rate = s_rate
        self.sample_list = s_list
        self.filter = fc
        self.fir = None
        if fc is not None:
            cutf = self.filter / self.s_rate
            self.fir = signal.firwin(numtaps=100, cutoff=cutf)

    def audio_analisys(self):
        if self.sample_list is not None:
            samples_list = self.sample_list
        else:
            samples_list = AudioTool.scan_folder(self)

        notes = []
        for i in range(0, len(samples_list)):
            audio, sample_rate = librosa.core.load(samples_list[i], sr=self.s_rate, dtype=np.float32)  # stereo sound to mono
            if self.fir is not None:
                audio = signal.lfilter(self.fir, 1, audio)
            #normalize TODO check
            #audio_max = max(abs(audio))
            #audio = audio / audio_max

            # if self.sys == 'fft':
            #     fft_signal = np.array(fft(audio, len(audio)))
            #     magnitude = abs(fft_signal)
            #     index = max(enumerate(magnitude), key=operator.itemgetter(1))
            #     fft_frequency = (index[0]*sample_rate)/len(audio)

            if self.sys == 'acor':
                acorr = signal.fftconvolve(audio, audio[::-1])
                acorr = acorr[0:len(acorr)//2]
                peakind = peakutils.indexes(acorr, min_dist=4)
                peakind = peakind[::-1]
                lag = len(acorr) - peakind[0]
                f_0 = sample_rate / lag

                if f_0 != 0:
                    nota = np.round(librosa.hz_to_midi(f_0))
                    notes.append(int(nota))
                if f_0 == 0:
                    notes.append(0)


            if self.sys == 'stft':
                stft_audio = librosa.stft(
                    audio,
                    n_fft=self.n_fft,
                    win_length=self.win_len,
                    hop_length=self.hop,
                    window=signal.hamming(self.win_len, sym=False)
                )
                magnitude = librosa.core.amplitude_to_db(stft_audio)
                frequency = np.zeros((1, len(stft_audio[0, :])))
                for j in range(0, len(magnitude[0, :])):
                    index = np.argmax(magnitude[:, j])
                    # index = max(enumerate(magnitude[:, i]), key=operator.itemgetter(1))
                    # frequency[0, j] = float(index[0] * sample_rate) / self.n_fft
                    frequency[0, j] = float(index * sample_rate) / self.n_fft
                f_0 = max(enumerate(frequency[0]), key=operator.itemgetter(1))
                if f_0[1] != 0:
                    nota = librosa.hz_to_midi(f_0[1])
                    notes.append(np.round(nota))
                if f_0[1] == 0:
                    notes.append(0)

            if self.sys == 'pip':
                p, m = librosa.core.piptrack(
                    y=audio,
                    sr=sample_rate,
                    S=None,
                    n_fft=self.n_fft,
                    hop_length=self.hop,
                    fmin=38,
                    fmax=3700,
                    threshold=0.9)

                k, j = np.nonzero(p)
                f_0 = np.median(p[k, j])
                if np.isnan(f_0):
                    f_0 = 0
                if f_0 != 0:
                    nota = np.round(librosa.hz_to_midi(f_0))
                    notes.append(int(nota))
                if f_0 == 0:
                    notes.append(0)
        return notes

    def scan_folder(self):
        samples = []
        if not os.path.isdir('./' + self.path):
            os.makedirs('./' + self.path)
            self.scan_folder()
        else:
            for i in range(0, len(os.listdir('./' + self.path))):
                if os.listdir('./' + self.path)[i].find('.' + self.extension) != -1:
                    samples.append('./' + self.path + '/' + os.listdir('./' + self.path)[i])

            samples.sort(key=lambda x: os.path.getmtime(x))

        return samples

    def __str__(self):
        values = [
            "Settings",
            "FFT Point: " + str(self.n_fft),
            "sys: "+str(self.sys),
            "win_len: " + str(self.win_len),
            "hop: " + str(self.hop),
            "path: " + str(self.path),
            "extension: " + str(self.extension)
        ]
        return values


