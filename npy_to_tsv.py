import numpy as np
import os

file_path = '/media/fabio/DATA/Work/crossSynth/autoXsynth/dataset/train_guns/stft-2048/Sweet Child O\' Mine Guitar.npy'
out_file = os.path.splitext(file_path)[0]+'.tsv'
max_frame = 500

spect = np.load(file_path)
spect = spect.T

spect_real = spect.real
spect_imag = spect.imag

spect = np.concatenate((spect_real,spect_imag), axis=1)

l = len(spect)
print ("This file has " + str(l) + " frames!")

f = open(out_file, 'w')
c = 0
for frame in spect:
    for bin in frame:
        f.write(str("{0:.3f}".format(bin))+'\t')
    f.write('\n')
    c += 1
    if c > max_frame:
        print ("Maximum number of printable frames reached!")
        break
f.close()