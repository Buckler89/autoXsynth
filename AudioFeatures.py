import os
import numpy as np
import librosa
from tqdm import tqdm

class AudioFeatures:
    """
    Compute stft of audio file inside the directory ( <path> ) passed as parameter (recursive).
    Put the stfts in in a sub folder inside <path> named <featType>-<hop>: all file in the same folder!!!
    """

    def __init__(self, feature='stft', n_fft=2048, win_len=1024, hop=1024, path='samples', extension='wav', channels=20, s_rate=None, free_disk=False):
        self.n_fft = n_fft
        self.feature = feature
        self.win_len = win_len
        self.hop = hop
        self.path = path
        self.channels = channels
        self.extension = extension
        self.s_rate = s_rate
        self.free_disk = free_disk

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
            for root, dirnames, filenames in os.walk('./' + self.path):
                for file in filenames:
                    ext = os.path.splitext(file)[-1].lower()
                    if ext == self.extension:
                        samples.append(os.path.join(root, file))
        if len(samples) == 0:
            print ("NO FILES FOUND!")
        else:
            print ("I've found " +str(len(samples)) + " files")
        return samples



    def feat_extract(self):
        samples_list = AudioFeatures.scan_folder(self)

        feat_fold_name = os.path.join(self.path, self.feature + "-" + str(self.hop))

        if not os.path.isdir(feat_fold_name):
            os.makedirs(feat_fold_name)

        for filename in tqdm(samples_list):
            feat_name = os.path.basename(filename)[0:-4] + ".npy"
            print("Extracting feature " + self.feature + " from file: " + filename)
            if os.path.isfile(os.path.join(feat_fold_name, feat_name)):
                print ("This file exists. Skipping!")

            else:
                audio, sample_rate = librosa.core.load(filename, sr=self.s_rate, dtype=np.float32) #stereo sound to mono

                if self.feature == 'stft':
                    y = librosa.core.stft(y=audio, n_fft=self.n_fft, hop_length=self.hop)
                    np.save(os.path.join(feat_fold_name, feat_name), y)

                if self.feature == 'mfcc':
                    y = np.abs(librosa.feature.mfcc(y=audio, sr=sample_rate, hop_length=self.hop, n_fft=self.n_fft, n_mfcc=self.channels))
                    np.save(os.path.join(feat_fold_name, feat_name), y)
            if self.free_disk:
                try:
                    os.remove(filename)
                    print('file {0} removed'.format(filename))
                except Exception as e:
                    print('failed to remove file {0}'.format(filename))
                    print(e)
                    pass

    def feat_extract_from_npy(self):
        feat_fold_name = os.path.join(os.path.splitext(self.path)[0], self.feature + "-" + str(self.hop))
        if not os.path.isdir(feat_fold_name):
            os.makedirs(feat_fold_name)
        data = np.load(self.path)
        feat_name = 000

        print ("\nThis file contains " + str(len(data)) + " sequences. Extracting!")
        for note in tqdm(data):
            audio = note[0]
            if self.feature == 'stft':
                y = librosa.core.stft(y=audio, n_fft=self.n_fft, hop_length=self.hop)
                np.save(os.path.join(feat_fold_name, str(feat_name)), y)
            feat_name += 1









