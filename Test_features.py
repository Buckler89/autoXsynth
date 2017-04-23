from AudioFeatures import *

audio_files_path = 'wavs/vox'
HOP = 2048
FTBINS = 4096
SR = 16000

test = AudioFeatures(feature='stft', n_fft=FTBINS, hop=HOP, path=audio_files_path, s_rate=SR, extension='.wav')

test.feat_extract()
