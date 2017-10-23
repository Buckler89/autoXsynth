from AudioFeatures import *
import time


audio_files_path = 'dataset/test'
HOP = 2048
FTBINS = 4096
SR = 22050

ts0 = time.time()
test = AudioFeatures(feature='stft', n_fft=FTBINS, hop=HOP, path=audio_files_path, s_rate=SR, extension='.wav', free_disk=False)
test.feat_extract()


ts1 = time.time()
tot_time = (ts1-ts0)
print (str(tot_time))


