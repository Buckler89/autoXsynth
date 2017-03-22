from AudioFeatures import *

audio_files_path = 'Beethoven-wav/BeethovenPianoSonata13'

test = AudioFeatures(feature='stft', n_fft=2048, path=audio_files_path, extension='ogg')
A = test.scan_folder()

B = test.feat_extract()
