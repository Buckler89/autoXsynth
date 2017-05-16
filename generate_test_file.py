#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:00:00 2017

@author: buckler
"""
import numpy as np
import os
import keras
import dataset_manupulation as dm
import librosa
import utility as u
import autoencoder


frame_context = 2
module_len = 2049
hops = 2048
nfft = 4096
sr = 16000

# set mix reconstruction params
aS = 0
bS = 0.5
aP = 1 - aS
bP = 1 - bS
aM = 0.001

root_dir = '/media/buckler/DataSSD/Phd/autoXsynthImproved/autoXsynth'
dataset_path = 'dataset/source/stft-2048/test_set_paper/'
modelBasePath = os.path.join(root_dir,'result','model')
destBasePath = os.path.join(root_dir,'result','reconstructedWav')

for root, dir, modelNames in os.walk(modelBasePath):
    for modelName in modelNames:
    #modelName = 'model_998.hd5'
        modelPath = os.path.join(modelBasePath, modelName)
        strID = modelName.replace('model_','').replace('.hd5','')
        model = autoencoder.autoencoder_fall_detection(strID)
        model.define_sequential_arch(params=None, path=modelPath)
        destPath = os.path.join(destBasePath, strID)
        u.makedir(destPath)

        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                #sourceStftPath = [os.path.join(dataset_path, file)]
                source_stft = dm.load_DATASET(dataset_path, [file])

                source = dm.reshape_set(source_stft, net_type='dense')
                source_sig = source[0].T.view().T

                source_sig_module = np.absolute(source_sig)
                source_sig_phase = np.angle(source_sig)
                cos_source_sig = np.cos(source_sig_phase)
                sin_source_sig = np.sin(source_sig_phase)

                source_sig_input = np.concatenate([source_sig_module, cos_source_sig, sin_source_sig], axis=1)
                #source_sig_input = np.hstack([source_sig_module, source_sig_phase])

                source_sig_input, _ = dm.create_context(source_sig_input, look_back=frame_context)
                source_sig_module = source_sig_module[: - frame_context - 1, :]
                source_sig_phase = source_sig_phase[: - frame_context - 1, :]

                prediction = np.asarray(model.reconstruct_spectrogram(source_sig_input), order="C")

                prediction_module = prediction[:, 0:module_len]
                prediction_cos = prediction[:, module_len:(module_len*2)]
                prediction_sin = prediction[:, (module_len*2):(module_len*3)]
                prediction_phase = prediction_cos + 1j * prediction_sin
                #prediction_phase = prediction[:, module_len:]


                Mx = aS * source_sig_module + aP * prediction_module + aM * np.sqrt( source_sig_module * prediction_module)
                Phix = bS * source_sig_phase + bP * prediction_phase

                prediction_complex = Mx * Phix

                S = librosa.core.istft(prediction_complex.T, hop_length=hops, win_length=nfft)
                out_filename = "reconstruction_" + file.replace('.npy','.wav')
                librosa.output.write_wav(os.path.join(destPath,out_filename), S, sr)

            print('recostruction of {0} done'.format(file))

print("END.")