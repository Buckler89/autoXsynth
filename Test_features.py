from AudioFeatures import *

audio_files_path = 'dataset/vox'
HOP = 1024
FTBINS = 4096

test = AudioFeatures(feature='stft', n_fft=FTBINS, hop=HOP, path=audio_files_path, extension='.wav')

B = test.feat_extract()
