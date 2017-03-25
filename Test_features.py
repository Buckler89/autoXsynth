from AudioFeatures import *

audio_files_path = 'dataset/train'
HOP = 1024
FTBINS = 4096
SR = 22050

test = AudioFeatures(feature='stft', n_fft=FTBINS, hop=HOP, path=audio_files_path, s_rate=SR, extension='.wav')

test.feat_extract()
