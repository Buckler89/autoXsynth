from AudioFeatures import *

audio_files_path = 'wavs/source/'
HOP = 2048
FTBINS = 4096
SR = 22050

test = AudioFeatures(feature='stft', n_fft=FTBINS, hop=HOP, path=audio_files_path, s_rate=SR, extension='.ogg')

test.feat_extract()
