# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:25:10 2016

@author: cella
"""
import os
import theano
import matplotlib.pyplot as plt
import numpy as np
import librosa
import joblib
import os
import fnmatch
import sys
from keras.layers import Input, Dense, LSTM
from keras.models import Model

#os.environ["PATH"]= os.environ["PATH"] + ":/usr/local/cuda/bin"
#os.environ["THEANO_FLAGS"]="device=gpu"

print(theano.config.device)
print (os.environ["PATH"])

DATA_DIR = "wavs/source/"
HOP = 1024
FTBINS = 4096
CQBINS = 80
BSIZE = 256
EPOCHS = 300
SOURCE_FILE = "wavs/vox/Vox.wav"
SR = 44100


def get_features(file, hop, ftbins, cqbins, sr=SR):
    yt, sr = librosa.core.load(file, sr=sr, mono=True)

    if len(yt) == 0:
        print('*** warning: empty file -> ' + file + '! ***')
        return 0

    F = librosa.core.stft(y=yt, n_fft=ftbins, hop_length=hop)
    CQ = np.log1p(1000 * np.abs(librosa.core.cqt(y=yt, sr=sr, hop_length=hop, n_bins=cqbins, real=False)))

    return F, CQ


cachedir = os.path.expanduser('./autoXsynth_joblib')
memory = joblib.Memory(cachedir=cachedir, verbose=1)
cached_get_features = memory.cache(get_features)


def compute_features(root_path, hop=512, ftbins=FTBINS, cqbins=CQBINS):
    X_list = []
    for root, dir, files in os.walk(root_path):
        waves = fnmatch.filter(files, "*.wav")
        if len(waves) != 0:
            X_list = joblib.Parallel(n_jobs=1)(
                joblib.delayed(cached_get_features)(
                    os.path.join(root, item), hop, ftbins, cqbins)
                for item in waves
            )

    Fs, CQs = list(map(np.hstack, zip(*X_list)))
    return Fs, CQs


def build_model(bins=CQBINS, activ='tanh', batch_size=2, timesteps=None):
    # this is the size of our encoded representations
    # encoding_dim = 80

    # this is our input placeholder
    input_img = Input(shape=(timesteps, bins))

    # "encoded" is the encoded representation of the input
    encoded1 = Dense(1024, activation=activ)(input_img)
    encoded2 = Dense(1024, activation=activ)(encoded1)
    encoded3 = Dense(1024, activation=activ)(encoded2)
    #encoded4 = Dense(1024, activation=activ)(encoded3)
    #encoded5 = Dense(1024, activation=activ)(encoded4)

    #ENCODED REPRESENTATION
    bottleneck = LSTM(80, activation=activ, return_sequences=True)(encoded3)

    # "decoded" is the lossy reconstruction of the input
    decoded1 = Dense(1024, activation=activ)(bottleneck)
    decoded2 = Dense(1024, activation=activ)(decoded1)
    decoded3 = Dense(1024, activation=activ)(decoded2)
    #decoded4 = Dense(1024, activation=activ)(decoded3)
    #decoded5 = Dense(1024, activation=activ)(decoded4)
    output_AE = Dense(bins, activation='linear')(decoded3)

    # this model maps an input to its reconstruction
    autoencoder = Model(input=input_img, output=output_AE)

    middle_layer_model = Model(input=input_img, output=bottleneck)

    autoencoder.compile(optimizer='adadelta', loss='mse')
    middle_layer_model.compile(optimizer='adadelta', loss='mse')

    autoencoder.summary()

    return autoencoder, middle_layer_model

if __name__ == "__main__":

    print ("Cross-synthesis with autoencoders");
    print ("")

    print ("computing features...")
    sys.stdout.flush()
    X_data_fft, X_data_cqt = compute_features(DATA_DIR, HOP, FTBINS, CQBINS)

    X_data_fft_real = X_data_fft.T.view().T
    X_data_fft_real.dtype = 'float32'

    # reshape input to be [samples, time steps, features]
    X_data_fft_real = np.reshape(X_data_fft_real, (X_data_fft_real.shape[1], 1, X_data_fft_real.shape[0]))

    print ("fitting model...")
    sys.stdout.flush()
    model, middle_layer = build_model(bins=X_data_fft_real.shape[2], timesteps=X_data_fft_real.shape[0], batch_size=1)
    model.fit(X_data_fft_real.T, X_data_fft_real.T, batch_size=BSIZE, nb_epoch=EPOCHS)

    sys.stdout.flush()

    F, C = get_features(SOURCE_FILE, HOP, FTBINS, CQBINS)
    model_output = np.zeros_like(C)

    F_real = F.T.view().T
    F_real.dtype = "float32"

    F_real.shape

    p = np.asarray(model.predict(F_real.T[0:10000]), order="C")
    pcomplex = p.T.view()
    pcomplex.dtype = "complex64"
    p.shape, pcomplex.shape

    synthesised_direct_fft = librosa.core.istft(pcomplex, hop_length=HOP, win_length=FTBINS)
    librosa.output.write_wav("./voice_check.wav", synthesised_direct_fft, SR)