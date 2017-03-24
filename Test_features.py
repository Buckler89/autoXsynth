from AudioFeatures import *

audio_files_path = 'dataset/train'
HOP = 1024
FTBINS = 4096

test = AudioFeatures(feature='stft', n_fft=FTBINS, hop=HOP, path=audio_files_path, extension='.wav')

test.feat_extract()
