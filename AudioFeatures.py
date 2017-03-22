import os
import soundfile as sf
import numpy as np
import librosa

class AudioFeatures:

    def __init__(self, feature='stft', n_fft=2048, win_len=1024, hop=1024, path='samples', extension='wav', channels=20):
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

        if not os.path.isdir(feat_fold_name):
            os.makedirs(feat_fold_name)

        for filename in samples_list:
            feat_name = os.path.basename(filename)[0:-4] + ".npy"
            if os.path.isfile(os.path.join(feat_fold_name,feat_name)):
                print ("This file exists. Skipping!")
            else:
                #audio, sample_rate = sf.read(filename, dtype=np.float32)
                audio, sample_rate = librosa.core.load(filename, dtype=np.float32)

                if self.feature == 'stft':
                    y = librosa.core.stft(y=audio, n_fft=self.n_fft, hop_length=self.hop)
                    np.save(os.path.join(feat_fold_name, feat_name), y)

                if self.feature == 'mfcc':
                    y = np.abs(librosa.feature.mfcc(y=audio, sr=sample_rate, hop_length=self.hop, n_fft=self.n_fft, n_mfcc=self.channels))
                    np.save(os.path.join(feat_fold_name, feat_name), y)







