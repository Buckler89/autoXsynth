from AudioFeatures import *

audio_files_path = '/media/buckler/DataSSD/Phd/autoXsynthImproved/autoXsynth/wavs/source'

test = AudioFeatures(feature='stft', n_fft=2048, path=audio_files_path, extension='wav')
A = test.scan_folder()

B = test.feat_extract()
