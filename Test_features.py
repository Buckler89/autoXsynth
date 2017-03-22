from AudioFeatures import *

audio_files_path = 'wavs/source'

test = AudioFeatures(feature='stft', n_fft=2048, path=audio_files_path)
A = test.scan_folder()

B = test.feat_extract()
