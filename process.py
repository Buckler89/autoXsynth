#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:11:09 2017

@author: buckler
"""
import numpy as np

np.random.seed(888)  # for experiment repetibility: this goes here, before importing keras (inside autoencoder modele) It works?
import autoencoder
import dataset_manupulation as dm

from os import path
import argparse
import os
import errno
import json
import fcntl
import time
import datetime
import utility as u
import librosa
import copy
###################################################PARSER ARGUMENT SECTION########################################
parser = argparse.ArgumentParser(description="AutoXSynthesis Autoencoder")


class eval_action(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(eval_action, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        values = eval(values)
        setattr(namespace, self.dest, values)


# Global params
parser.add_argument("-id", "--exp-index", dest="id", default=0, type=int)
parser.add_argument("-root", "--root-dir",dest="root_dir", default=".", type=str)
parser.add_argument("-log", "--logging", dest="log", default=False, action="store_true")

parser.add_argument("-cf", "--config-file", dest="config_filename", default=None)
parser.add_argument("-sp", "--score-path", dest="scorePath", default="score")
parser.add_argument("-tl", "--trainset-list", dest="trainNameLists", action=eval_action, default=["trainset.lst"])
parser.add_argument("-c", "--case", dest="case", default="case6")
parser.add_argument("-tln", "--test-list-names", dest="testNamesLists", action=eval_action,
                    default=["testset_1.lst", "testset_2.lst", "testset_3.lst", "testset_4.lst"])
parser.add_argument("-dl", "--dev-list-names", dest="devNamesLists", action=eval_action,
                    default=["devset_1.lst", "devset_2.lst", "devset_3.lst", "devset_4.lst"])
parser.add_argument("-it", "--input-type", dest="input_type", default="spectrograms")
parser.add_argument("-tt", "--target-type", dest="target_type", default="mfcc")

sr = 22050
hops = 1024
nfft = 4096

# CNN params
# parser.add_argument("-cln", "--conv-layers-numb", dest="conv_layer_numb", default=3, type=int)
# parser.add_argument("-is", "--cnn-input-shape", dest="cnn_input_shape", action=eval_action, default=[1, 129, 197])
# parser.add_argument("-kn", "--kernels-number", dest="kernel_number", action=eval_action, default=[16, 8, 8])
# parser.add_argument("-ks", "--kernel-shape", dest="kernel_shape", action=eval_action, default=[[3, 3], [3, 3], [3, 3]])
# parser.add_argument("-mp", "--max-pool-shape", dest="m_pool", action=eval_action, default=[[2, 2], [2, 2], [2, 2]])
# parser.add_argument("-s", "--strides", dest="strides", action=eval_action, default=[[1, 1], [1, 1], [1, 1]])
# parser.add_argument("-cwr", "--cnn-w-reg", dest="cnn_w_reg",
#                     default="None")  # in autoencoder va usato con eval("funz(parametri)")
# parser.add_argument("-cbr", "--cnn-b-reg", dest="cnn_b_reg", default="None")
# parser.add_argument("-car", "--cnn-act-reg", dest="cnn_a_reg", default="None")
# parser.add_argument("-cwc", "--cnn-w-constr", dest="cnn_w_constr", default="None")
# parser.add_argument("-cbc", "--cnn-b-constr", dest="cnn_b_constr", default="None")
# parser.add_argument("-ac", "--cnn-conv-activation", dest="cnn_conv_activation", default="tanh", choices=["tanh"])
#dense
parser.add_argument("-is", "--dense-input-shape", dest="dense_input_shape", default=20, type=int)
parser.add_argument("-dln", "--dense-layers-numb", dest="dense_layer_numb", default=1, type=int)
parser.add_argument("-ds", "--dense-shapes", dest="dense_shapes", action=eval_action, default=[64])
parser.add_argument("-i", "--init", dest="init", default="glorot_uniform", choices=["glorot_uniform"])
parser.add_argument("-ad", "--dense-activation", dest="dense_activation", default="tanh", choices=["tanh"])
parser.add_argument("-bm", "--border-mode", dest="border_mode", default="same", choices=["valid", "same"])
parser.add_argument("-dwr", "--d-w-reg", dest="d_w_reg",
                    default="None")  # in autoencoder va usato con eval("funz(parametri)")
parser.add_argument("-dbr", "--d-b-reg", dest="d_b_reg", default="None")
parser.add_argument("-dar", "--d-act-reg", dest="d_a_reg", default="None")
parser.add_argument("-dwc", "--d-w-constr", dest="d_w_constr", default="None")
parser.add_argument("-dbc", "--d-b-constr", dest="d_b_constr", default="None")
parser.add_argument("-drp", "--dropout", dest="dropout", default=False, action="store_true")
parser.add_argument("-drpr", "--drop-rate", dest="drop_rate", default=0.5, type=float)

parser.add_argument("-nb", "--no-bias", dest="bias", default=True, action="store_false")
parser.add_argument("-p", "--pool-type", dest="pool_type", default="all", choices=["all", "only_end"])

# fit params
parser.add_argument("-e", "--epoch", dest="epoch", default=50, type=int)
parser.add_argument("-ns", "--no-shuffle", dest="shuffle", default=True, action="store_false")
parser.add_argument("-bs", "--batch-size", dest="batch_size", default=128, type=int)
parser.add_argument("-f", "--fit-net", dest="fit_net", default=False, action="store_true")
parser.add_argument("-o", "--optimizer", dest="optimizer", default="adadelta", choices=["adadelta", "adam", "sgd"])
parser.add_argument("-l", "--loss", dest="loss", default="mse", choices=["mse", "msle"])
parser.add_argument("-pt", "--patiance", dest="patiance", default=20, type=int)
parser.add_argument("-ami", "--aucMinImp", dest="aucMinImprovment", default=0.01, type=float)
parser.add_argument("-lr", "--learning-rate", dest="learning_rate", default=1.0, type=float)
parser.add_argument("-vl", "--validation-split", dest="val_split", default=0.0, type=float) #TODO add validation_split at the generator script

args = parser.parse_args()

if args.config_filename is not None:
    with open(args.config_filename, "r") as f:
        lines = f.readlines()
    arguments = []
    for line in lines:
        arguments.extend(line.split("#")[0].split())
    # First parse the arguments specified in the config file
    args, unknown = parser.parse_known_args(args=arguments)
    # Then append the command line arguments
    # Command line arguments have the priority: an argument is specified both
    # in the config file and in the command line, the latter is used
    args = parser.parse_args(namespace=args)

###################################################END PARSER ARGUMENT SECTION########################################



###################################################INIT LOG########################################
# redirect all the stream of both standard.out and standard.err to the same logger
strID = str(args.id)

print("init log")
root_dir = path.realpath(args.root_dir)
baseResultPath = os.path.join(root_dir,'result')
logFolder = os.path.join(baseResultPath, 'logs')
csvFolder = os.path.join(baseResultPath, 'csv')
wavDestPath = os.path.join(baseResultPath, 'reconstructedWav')
argsFolder = os.path.join(baseResultPath, 'args')

u.makedir(logFolder)
u.makedir(csvFolder)
u.makedir(wavDestPath)
u.makedir(argsFolder)

nameFileLog = os.path.join(logFolder, 'process_' + strID + '.log')
nameFileLogCsv = os.path.join(csvFolder, 'process_' + strID + '.csv')  # log in csv file the losses for further analysis
reconstructedFile = os.path.join(wavDestPath, 'process_' + strID + '.wav')
jsonargsFileName = os.path.join(argsFolder, 'process_' + strID + '.json')
jsonargs = json.dumps(args.__dict__)


with open(os.path.join(jsonargsFileName), 'w') as file:
    file.write(json.dumps(jsonargs, indent=4))

if args.log:
    import logging
    import sys

    u.makedir(logFolder)  # crea la fold solo se non esiste

    stdout_logger = logging.getLogger(strID)
    sl = u.StreamToLogger(stdout_logger, nameFileLog, logging.INFO)
    sys.stdout = sl  # ovverride funcion

    stderr_logger = logging.getLogger(strID)
    sl = u.StreamToLogger(stderr_logger, nameFileLog, logging.ERROR)
    sys.stderr = sl  # ovverride funcion
###################################################END INIT LOG########################################

print("LOG OF PROCESS ID = " + strID)
ts0 = time.time()
st0 = datetime.datetime.fromtimestamp(ts0).strftime('%Y-%m-%d %H:%M:%S')
print("experiment start in date: " + st0)

trainStftPath = os.path.join(root_dir, 'dataset', 'train', args.input_type)


# LOAD DATASET
X_data = dm.load_DATASET(trainStftPath)
#todo reshape dataset: la funzione che ce in autoencoder lo reshapa per darlo ad una rete cnn! noi abbiamo un semplice dense per il momento
X_data_reshaped = dm.reshape_set(X_data, net_type='dense')
X_data_reshaped = X_data_reshaped[0].astype("complex64")
X_data_reshaped = X_data_reshaped.T.view().T
X_data_reshaped.dtype = 'float32'
args.dense_input_shape = X_data_reshaped.shape[1]
#model definition
model = autoencoder.autoencoder_fall_detection(strID)
model.define_sequential_arch(args)
#model copile
model.model_compile(optimizer=args.optimizer, loss=args.loss, learning_rate=args.learning_rate)

#model fit
m = model.model_fit(X_data_reshaped, X_data_reshaped, validation_split=args.val_split, nb_epoch=args.epoch,
                  batch_size=args.batch_size, shuffle=args.shuffle,
                  fit_net=args.fit_net, patiance=args.patiance,
                  nameFileLogCsv=nameFileLogCsv)


sourceStftPath = os.path.join(root_dir, 'dataset', 'source', args.input_type)

source_stft = dm.load_DATASET(sourceStftPath)
#todo reshape source_stft
source = dm.reshape_set(source_stft, net_type='dense')
source_real = source[0].astype("complex64")
source_real = source_real.T.view().T
source_real_CAST = copy.deepcopy(source_real)
source_real_CAST.dtype = 'float32'
prediction = np.asarray(model.reconstruct_spectrogram(source_real_CAST), order="C")

prediction = prediction.view()
prediction_CAST = copy.deepcopy(prediction)
prediction_CAST.dtype = "complex64"

S = librosa.core.istft(prediction_CAST.T, hop_length=hops, win_length=nfft)
out_filename = "reconstruction_" + strID + ".wav"
librosa.output.write_wav(os.path.join(wavDestPath,out_filename), S, sr)

ts1 = time.time()
tot_time = (ts1-ts0)/60

print "Experiment emplased " +str(tot_time) + " minutes."
print "END."

