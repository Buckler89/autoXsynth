#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:11:09 2017

@author: buckler
"""
import numpy as np

np.random.seed(888)  # for experiment repeatability: this goes here, before importing keras (inside autoencoder modele) It works?
import autoencoder
import dataset_manupulation as dm

from os import path
import argparse
import os
import json
import time
import datetime
import utility as u
import librosa
import copy
import logging
import sys
from tensorflow.python.keras.models import load_model
# from tensorflow.python.keras.callbacks import Callback
from CustomCallback import GenerateWavCallback
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
parser.add_argument("-cf", "--config-file", dest="config_filename", default=None)

parser.add_argument("-id", "--exp-index", dest="id", default=0, type=int)
parser.add_argument("-root", "--root-path",dest="root_dir", default=".", type=str)
# parser.add_argument("-results", "--results-path",dest="results_dir", default=".", type=str)
parser.add_argument("-log", "--logging", dest="log", default=False, action="store_true")
parser.add_argument("-sv", "--save-model", dest="save_model", default=False, action="store_true")
parser.add_argument("-lo", "--load-model", dest="load_model", default=False, action="store_true")

parser.add_argument("-sp", "--score-path", dest="scorePath", default="score")
parser.add_argument("-c", "--source", dest="source", default="Vox.npy")

parser.add_argument("-it", "--input-type", dest="input_type", default="stft")
parser.add_argument("-tt", "--target-type", dest="target_type", default="mfcc")

parser.add_argument("-hp", "--hybrid-phase", dest="hybrid_phase", default=False, action="store_true")
parser.add_argument("-ts", "--trainset", dest="trainset", default="train")
parser.add_argument("-jp", "--json-path", dest="jsonPath", default=None)
parser.add_argument("-ifs", "--instrument-family-strs", dest="instrument_family_strs", default='all', choices=["all", "bass","brass","flute","guitar","keyboard","mallet","organ","reed","string","synth_lead","vocal"])
parser.add_argument("--notes", dest="notes", default='all', action=eval_action)
parser.add_argument("-vmin", "--velocity-min", dest="velocityMin", default=0, type=int)
parser.add_argument("-vmax", "--velocity-max", dest="velocityMax", default=127, type=int)
parser.add_argument("-iss", "--instrument-source-strs", dest="instrument_source_strs", default='all', choices=["all","acoustic","electronic","synthetic"])
parser.add_argument("-mnof", "--max-number-of-file", dest="maxNumberOfFile", default=127, type=int)

parser.add_argument("--hop-size", dest="hopsize", default=2048)
parser.add_argument("--nfft", dest="nfft", default=2048)
parser.add_argument("--win-len", dest="win_len", default=2048)
parser.add_argument("-sr", "--sample-rate", dest="sample_rate", default=22050, type=int)

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
parser.add_argument("-ad", "--dense-activation", dest="dense_activation", default="tanh", choices=["tanh","relu"])
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
parser.add_argument("-bn", "--batch-norm", dest="batch_norm", default=False, action="store_true")

#RNN
parser.add_argument("-rnn", "--RNN-type", dest="RNN_type", default=None, choices=["LSTM", "SimpleRNN","GRU"])
parser.add_argument("-rns", "--RNN-layer-shape", dest="RNN_layer_shape", default=None, type=int)
parser.add_argument("-cxt", "--frame-context", dest="frame_context", default=None, type=int)

# fit params
parser.add_argument("-e", "--epoch", dest="epoch", default=50, type=int)
parser.add_argument("-ns", "--shuffle", dest="shuffle", default="True", choices=["True","False","batch"])
parser.add_argument("-bsf", "--batch-size-fract", dest="batch_size_fract", default=None, type=float, help='batch size express in % of the trainset')
parser.add_argument("-bse", "--batch-size-effective", dest="batch_size_effective", default=None, type=int, help='batch size in number of sample')

parser.add_argument("-f", "--fit-net", dest="fit_net", default=False, action="store_true")
parser.add_argument("-o", "--optimizer", dest="optimizer", default="adadelta", choices=["adadelta", "adam", "sgd"])
parser.add_argument("-l", "--loss", dest="loss", default="mse", choices=["mse", "msle"])
parser.add_argument("-pt", "--patience", dest="patience", default=20, type=int)
parser.add_argument("-lr", "--learning-rate", dest="learning_rate", default=1.0, type=float)
parser.add_argument("-vl", "--validation-split", dest="val_split", default=0.0, type=float)

#mix reconstruction param
parser.add_argument("-aS", "--a-source", dest="aS", default=0.1, type=float)
parser.add_argument("-aP", "--a-pred", dest="aP", default=None, type=float)
parser.add_argument("-aM", "--a-mix", dest="aM", default=1, type=float)
parser.add_argument("-bS", "--b-source", dest="bS", default=0.1, type=float)
parser.add_argument("-bP", "--b-pred", dest="bP", default=None, type=float)

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

# set mix reconstruction params
if args.shuffle == "True":
    args.shuffle = True
elif args.shuffle == "False":
    args.shuffle = False
if args.aP is None:
    args.aP = 1 - args.aS
if args.bP is None:
    args.bP = 1 - args.bS

if args.batch_size_effective is None and args.batch_size_fract is None:
    print("specify batch-size in % or in absolute number")
    raise ValueError('specify batch-size in % or in absolute number')

if args.batch_size_effective is not None and args.batch_size_fract is not None:
    print("specify batch-size only in % or in absolute number")
    raise ValueError('specify batch-size only in % or in absolute number')



#Feature Params
sample_rate = args.sample_rate
hopsize = int(args.hopsize)
nfft = int(args.nfft)
win_len = int(args.win_len)
###################################################END PARSER ARGUMENT SECTION########################################



###################################################INIT LOG########################################
# redirect all the stream of both standard.out and standard.err to the same logger
strID = str(args.id)

print("init log")
results_dir = os.path.join('experiments', strID)
root_dir = path.realpath(args.root_dir)
baseResultPath = os.path.join(root_dir,     results_dir)
logFolder = os.path.join(baseResultPath,     'logs')
csvFolder = os.path.join(baseResultPath,     'csv')
wavDestPath = os.path.join(baseResultPath,   'reconstructedWav')
modelDestPath = os.path.join(baseResultPath, 'model')
argsFolder = os.path.join(baseResultPath,    'args')
predFolder = os.path.join(baseResultPath,    'preds')

u.makedir(logFolder)
u.makedir(csvFolder)
u.makedir(wavDestPath)
u.makedir(modelDestPath)
u.makedir(argsFolder)
u.makedir(predFolder)

nameFileLog = os.path.join(logFolder, 'process_' + strID + '.log')
nameFileLogCsv = os.path.join(csvFolder, 'process_' + strID + '.csv')  # log in csv file the losses for further analysis
reconstructedFile = os.path.join(wavDestPath, 'process_' + strID + '.wav')
jsonargsFileName = os.path.join(argsFolder, 'process_' + strID + '.json')
jsonargs = json.dumps(args.__dict__, indent=4)


with open(os.path.join(jsonargsFileName), 'w') as file:
    #file.write(json.dump(jsonargs, indent=4))
    json.dump(jsonargs, file,  indent=4)

if args.log:

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

trainStftPath = os.path.join(root_dir, 'dataset', args.trainset, args.input_type)

# LOAD DATASET
if 'nsynth' in args.trainset:
    if args.notes is 'all':
        notes = 'all'
    else:
        notes = dm.parsenotes(args.notes)
    jsonPath = os.path.join(root_dir,  'dataset', args.jsonPath)
    #with open(jsonPath, 'r', encoding='utf-8') as infile:
    with open(jsonPath, 'r') as infile:
        jsonFile = json.load(infile)
    fileslist = dm.scanJson(jsonFile,
                            instrument_family_strs=args.instrument_family_strs,
                            notes=notes,
                            instrument_source_strs=args.instrument_source_strs,
                            velocityMin=args.velocityMin,
                            velocityMax=args.velocityMax,
                            maxNumberOfFile=args.maxNumberOfFile)
    X_data = dm.load_DATASET(trainStftPath, fileslist)
else:
    X_data = dm.load_DATASET(trainStftPath)
# X_data_reshaped, label = dm.reshape_set(X_data, net_type='dense')

X_data_reshaped = X_data[0][1].T
# X_data_reshaped = X_data_reshaped.T.view().T

#load source
sourceStftPath = os.path.join(root_dir, 'dataset', 'source', args.source, args.input_type)

source_stft = dm.load_DATASET(sourceStftPath)
# source = dm.reshape_set(source_stft, net_type='dense')
source_sig = source_stft[0][1].T


if args.hybrid_phase:
    X_data_module = np.absolute(X_data_reshaped)
    module_len = X_data_module.shape[1]
    X_data_phase = np.angle(X_data_reshaped)
    cos_phi_X_data = np.cos(X_data_phase)
    sin_phi_X_data = np.sin(X_data_phase)

    X_data_reshaped = np.hstack([X_data_module, cos_phi_X_data, sin_phi_X_data])
    #X_data_reshaped = np.hstack([X_data_module, X_data_phase])

    source_sig_module = np.absolute(source_sig)
    source_sig_phase = np.angle(source_sig)
    cos_source_sig = np.cos(source_sig_phase)
    sin_source_sig = np.sin(source_sig_phase)
    source_sig_input = np.concatenate([source_sig_module, cos_source_sig, sin_source_sig], axis=1)
    if args.RNN_type is not None:
        source_sig_input, _ = dm.create_context(source_sig_input, look_back=args.frame_context)
        source_sig_module = source_sig_module[: - args.frame_context - 1, :]
        source_sig_phase = source_sig_phase[: - args.frame_context - 1, :]

else:
    X_data_reshaped.dtype = 'float32'

# calcolo il batch size
batch_size = int(len(X_data_reshaped) * args.batch_size_fract)
args.batch_size = batch_size
print("Training on " + str(X_data_reshaped.shape[0]) + " samples")
print("Batch size: " + str(batch_size) + " samples")


#model definition
if args.RNN_type is not None:
    X_data, Y_data = dm.create_context(X_data_reshaped, look_back=args.frame_context)
    args.dense_input_shape = X_data.shape[2]
else:
    args.dense_input_shape = X_data_reshaped.shape[1]
    X_data = X_data_reshaped
    Y_data = X_data

modelName = 'model_' + strID + '.hd5'

if not args.load_model:

    model = autoencoder.autoencoder_fall_detection(strID)
    model.define_sequential_arch(params=args)
    generate_wav_at_each_epoch = GenerateWavCallback(args, source_sig, predFolder, wavDestPath)
    callback_list = [generate_wav_at_each_epoch]
    #model copile
    model.model_compile(optimizer=args.optimizer, loss=args.loss, learning_rate=args.learning_rate)

    #model fit
    m = model.model_fit(X_data, Y_data, validation_split=args.val_split, epochs=args.epoch,
                      batch_size=batch_size, shuffle=args.shuffle,
                      fit_net=args.fit_net, patience=args.patience,
                      nameFileLogCsv=nameFileLogCsv, callback_list=callback_list)

    if args.save_model:
        m.save(os.path.join(modelDestPath, modelName))
        print("model saved at {0}".format(os.path.join(wavDestPath, modelName)))
else:

    m = load_model(os.path.join(modelDestPath, modelName))
    model = autoencoder.autoencoder_fall_detection(strID, model=m)



if args.hybrid_phase:
    #TODO DO it separately for module, sin, cos
    source_sig_module = np.absolute(source_sig)
    source_sig_phase = np.angle(source_sig)
    cos_source_sig = np.cos(source_sig_phase)
    sin_source_sig = np.sin(source_sig_phase)

    source_sig_input = np.concatenate([source_sig_module, cos_source_sig, sin_source_sig], axis=1)
    #source_sig_input = np.hstack([source_sig_module, source_sig_phase])

    if args.RNN_type is not None:
        source_sig_input, _ = dm.create_context(source_sig_input, look_back=args.frame_context)
        source_sig_module = source_sig_module[: - args.frame_context - 1, :]
        source_sig_phase = source_sig_phase[: - args.frame_context - 1, :]

    prediction = np.asarray(model.reconstruct_spectrogram(source_sig_input), order="C")
    pred_name = "prediction_" + strID
    np.save(os.path.join(predFolder, pred_name), prediction)

    prediction_module = prediction[:, 0:module_len]
    prediction_cos = prediction[:, module_len:(module_len*2)]
    prediction_sin = prediction[:, (module_len*2):(module_len*3)]
    prediction_phase = prediction_cos + 1j * prediction_sin
    #prediction_phase = prediction[:, module_len:]


    Mx = args.aS * source_sig_module + args.aP * prediction_module + args.aM * np.sqrt( source_sig_module * prediction_module)
    Phix = args.bS * source_sig_phase + args.bP * prediction_phase

    # Mx = prediction_module
    # Phix = cos_source_sig + 1j * sin_source_sig

    # Phix = prediction_phase
    # prediction_complex = Mx * np.exp(1j*Phix)
    prediction_complex = Mx * Phix
else:
    source_sig.dtype = 'float32'
    source_sig_input = source_sig
    if args.RNN_type is not None:
        source_sig_input, _ = dm.create_context(source_sig, look_back=args.frame_context)
    prediction = np.asarray(model.reconstruct_spectrogram(source_sig_input), order="C")
    prediction_complex = prediction.view()
    prediction_complex.dtype = "complex64"

# prediction_complex = librosa.util.fix_length(prediction_complex, len(prediction_complex) + win_len)
S = librosa.core.istft(prediction_complex.T, hop_length=hopsize, win_length=win_len)
out_filename = "reconstruction_" + strID + ".wav"
librosa.output.write_wav(os.path.join(wavDestPath, out_filename), S, sample_rate)

ts1 = time.time()
tot_time = (ts1-ts0)/60

print("Experiment enlapsed " +str(tot_time) + " minutes.")
print("END.")


