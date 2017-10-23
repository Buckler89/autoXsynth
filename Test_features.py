from AudioFeatures import *
audio_files_path = 'dataset/source/wav/app/'
HOP = 2048
FTBINS = 4096
SR = 16000

test = AudioFeatures(feature='stft', n_fft=FTBINS, hop=HOP, path=audio_files_path, s_rate=SR, extension='.mp3', free_disk=False)

test.feat_extract()
