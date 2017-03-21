import os
import soundfile as sf
import numpy as np
import librosa



class AudioFeatures:

    def __init__(self, feature='stft', n_fft=4096, win_len=1024, hop=1024, path='samples', extension='wav', channels='20'):
        self.n_fft = n_fft
        self.feature = feature
        self.win_len = win_len
        self.hop = hop
        self.path = path
        self.channels = channels
        self.extension = extension

    def __str__(self):
        values = [
            "Settings",
            "Feature: " + str(self.feature),
            "FFT Points: " + str(self.n_fft),
            "win_len: " + str(self.win_len),
            "hop: " + str(self.hop),
            "path: " + str(self.path),
            "extension: " + str(self.extension)
        ]
        return values

    def scan_folder(self):
        samples = []
        if not os.path.isdir('./' + self.path):
            print("This is not a directory!")
        else:
            for i in range(0, len(os.listdir('./' + self.path))):
                if os.listdir('./' + self.path)[i].find('.' + self.extension) != -1:
                    samples.append('./' + self.path + '/' + os.listdir('./' + self.path)[i])

        if len(samples) == 0:
            print ("NO FILES FOUND!")
        else:
            print ("I've found " +str(len(samples)) + " files")
        return samples



    def feat_extract(self):
        samples_list = AudioFeatures.scan_folder(self)
        feat_fold_name = os.path.join(self.path, self.feature)

        if not feat_fold_name:
            os.makedirs(feat_fold_name)

        for i in range(0, len(samples_list)):
            filename = samples_list[i]
            audio, sample_rate = sf.read(filename, dtype=np.float32)

            if self.feature == 'stft':
                y = librosa.core.stft(y=audio, n_fft=self.n_fft, hop_length=self.hop)
                np.save(path.join(feat_fold_name, file[0:-4]), y)

            if self.feature == 'mfcc':
                y = np.abs(librosa.feature.mfcc(y=audio, sr=sample_rate, hop_length=self.hop, n_fft=self.n_fft, n_mfcc=self.channels))
                np.save(path.join(feat_fold_name, file[0:-4]), y)




