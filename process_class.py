#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:11:09 2017
@author: buckler
"""
import numpy as np
np.random.seed(888)  # for experiment repeatability: this goes here, before importing keras (inside autoencoder
# modele) It works?
import dataset_manupulation as dm
from os import path
import argparse
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import json
import fcntl
import time
import datetime
import utility as u
import librosa
import copy
import importlib
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.metrics import fbeta_score
from experiments.settings import *
import pickle

class eval_action(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(eval_action, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        values = eval(values)
        setattr(namespace, self.dest, values)


##FROM Veleslavia-EUSIPCO2017 code
class Trainer(object):
    init_lr = 0.001

    def __init__(self, X_train, X_val, y_train, y_val, model_module, optimizer, load_to_memory):
        self.model_module = model_module
        #self.dataset_mean = np.load(os.path.join(MODEL_MEANS_BASEPATH, "{}_mean.npy".format(model_module.BASE_NAME)))
        self.optimizer = optimizer if optimizer != 'sgd' else SGD(lr=self.init_lr, momentum=0.9, nesterov=True)
        self.in_memory_data = load_to_memory
        self.y_train = np.asarray(y_train)
        self.y_val = np.asarray(y_val)
        if self.in_memory_data:
            self.X_train = self._load_features(X_train)
            self.X_val = self._load_features(X_val)
        else:
            self.X_train = X_train
            self.X_val = X_val

    def _load_features(self, filenames):
        features = []
        for filename in filenames:
            feature_filename = os.path.join(trainStftPath, filename)
            feature = np.load(feature_filename)
            #feature = np.absolute(feature)
            #feature -= self.dataset_mean
            #LOADS Spectrograms of each sequence --> Logmelspectr --> truncate @ 2 seconds
            melspectr = librosa.feature.melspectrogram(S=feature, n_mels=self.model_module.N_MEL_BANDS, fmax=FS/2)
            logmelspectr = librosa.logamplitude(melspectr**2, ref_power=1.0)
            features.append(logmelspectr[:, 0:SEGMENT_DUR])
        if K.image_dim_ordering() == 'th':
            features = np.array(features).reshape(-1, 1, self.model_module.N_MEL_BANDS, SEGMENT_DUR)
        else:
            features = np.array(features).reshape(-1, self.model_module.N_MEL_BANDS, SEGMENT_DUR, 1)
        return features

    def _batch_generator(self, inputs, targets):
        assert len(inputs) == len(targets)
        while True:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
            for start_idx in range(0, len(inputs) - BATCH_SIZE + 1, BATCH_SIZE):
                excerpt = indices[start_idx:start_idx + BATCH_SIZE]
                if self.in_memory_data:
                    yield inputs[excerpt], targets[excerpt]
                else:
                    yield self._load_features(inputs[excerpt]), targets[excerpt]

    def train(self):
        model = self.model_module.build_model(N_CLASSES)

        early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_EPOCH)
        save_clb = ModelCheckpoint(
            "{weights_basepath}/{model_path}/".format(
                weights_basepath=MODEL_WEIGHT_BASEPATH,
                model_path=self.model_module.BASE_NAME) +
            "epoch.{epoch:02d}-val_loss.{val_loss:.3f}-fbeta.{val_fbeta_score:.3f}" + "-{key}.hdf5".format(
                key=self.model_module.MODEL_KEY),
            monitor='val_loss',
            save_best_only=True)
        lrs = LearningRateScheduler(lambda epoch_n: self.init_lr / (2 ** (epoch_n // SGD_LR_REDUCE)))
        model.summary()
        model.compile(optimizer=self.optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy', fbeta_score])

        if self.in_memory_data:
            history = model.fit(self.X_train, self.y_train, batch_size=BATCH_SIZE, nb_epoch=MAX_EPOCH_NUM,
                                verbose=2, callbacks=[save_clb, early_stopping, lrs],
                                validation_data=(self.X_val,self.y_val), shuffle=True)
        else:
            history = model.fit_generator(self._batch_generator(self.X_train, self.y_train),
                                      samples_per_epoch=self.model_module.SAMPLES_PER_EPOCH,
                                      nb_epoch=MAX_EPOCH_NUM,
                                      verbose=2,
                                      callbacks=[save_clb, early_stopping, lrs],
                                      validation_data=self._batch_generator(self.X_val, self.y_val),
                                      nb_val_samples=self.model_module.SAMPLES_PER_VALIDATION,
                                      class_weight=None,
                                      nb_worker=1)

        pickle.dump(history.history, open('{history_basepath}/{model_path}/history_{model_key}.pkl'.format(
            history_basepath=MODEL_HISTORY_BASEPATH,
            model_path=self.model_module.BASE_NAME,
            model_key=self.model_module.MODEL_KEY),
            'w'))


###################################################PARSER ARGUMENT SECTION########################################
parser = argparse.ArgumentParser(description="Instrument/Timber Classifier")
# Global params
parser.add_argument("-id", "--exp-index", dest="id", default=0, type=int)
parser.add_argument("-root", "--root-path", dest="root_dir", default=".", type=str)
parser.add_argument("-log", "--logging", dest="log", default=False, action="store_true")
parser.add_argument("-sv", "--save-model", dest="save_model", default=False, action="store_true")

parser.add_argument("-cf", "--config-file", dest="config_filename", default=None)
parser.add_argument("-sp", "--score-path", dest="scorePath", default="score")
parser.add_argument("-c", "--source", dest="source", default="Vox.npy")

parser.add_argument("-it", "--input-type", dest="input_type", default="stft")
parser.add_argument("-tt", "--target-type", dest="target_type", default="mfcc")

parser.add_argument('--model', action='store', dest='model', help='-m model to train')
parser.add_argument('--load_to_memory', action='store_true', dest='load_to_memory', default=False,
                    help='-l load dataset to memory')

parser.add_argument("-hp", "--hybrid-phase", dest="hybrid_phase", default=False, action="store_true")
parser.add_argument("-ts", "--trainset", dest="trainset", default="train")
parser.add_argument("-jp", "--json-path", dest="jsonPath", default=None)
# parser.add_argument("-ifs", "--instrument-family-strs", dest="instrument_family_strs", default='all',
#                     choices=["all", "bass", "brass", "flute", "guitar", "keyboard", "mallet", "organ", "reed", "string",
#                              "synth_lead", "vocal"])
parser.add_argument("-ifs", "--instrument-family-strs", dest="instrument_family_strs", default='all', action=eval_action)
parser.add_argument("--notes", dest="notes", default='all', action=eval_action)
parser.add_argument("-vmin", "--velocity-min", dest="velocityMin", default=0, type=int)
parser.add_argument("-vmax", "--velocity-max", dest="velocityMax", default=127, type=int)
parser.add_argument("-iss", "--instrument-source-strs", dest="instrument_source_strs", default='all',
                    choices=["all", "acoustic", "electronic", "synthetic"])
parser.add_argument("-mnof", "--max-number-of-file", dest="maxNumberOfFile", default=127, type=int)

parser.add_argument("-hop", dest="hopsize", default=2048)
parser.add_argument("-sr", "--sample-rate", dest="sample_rate", default=22050, type=int)

# fit params
parser.add_argument("-e", "--epoch", dest="epoch", default=50, type=int)
parser.add_argument("-ns", "--shuffle", dest="shuffle", default="True", choices=["True", "False", "batch"])
parser.add_argument("-bsf", "--batch-size-fract", dest="batch_size_fract", default=None, type=float,
                    help='batch size express in % of the trainset')
parser.add_argument("-bse", "--batch-size-effective", dest="batch_size_effective", default=None, type=int,
                    help='batch size in number of sample')

parser.add_argument("-f", "--fit-net", dest="fit_net", default=False, action="store_true")
parser.add_argument("-o", "--optimizer", dest="optimizer", default="adadelta", choices=["adadelta", "adam", "sgd"])
parser.add_argument("-l", "--loss", dest="loss", default="mse", choices=["mse", "msle"])
parser.add_argument("-pt", "--patiance", dest="patiance", default=20, type=int)
parser.add_argument("-lr", "--learning-rate", dest="learning_rate", default=1.0, type=float)
parser.add_argument("-vl", "--validation-split", dest="val_split", default=0.0, type=float)

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

if args.batch_size_effective is None and args.batch_size_fract is None:
    print("specify batch-size in % or in absolute number")
    raise ValueError('specify batch-size in % or in absolute number')

if args.batch_size_effective is not None and args.batch_size_fract is not None:
    print("specify batch-size only in % or in absolute number")
    raise ValueError('specify batch-size only in % or in absolute number')

if not args.model:
    parser.error('Please, specify the model to train!')
try:
    if args.model in ALLOWED_MODELS:
        model_module = importlib.import_module(".{}".format(args.model), "experiments.models")
        print "{} imported as 'model'".format(args.model)
    else:
        print "The specified model is not allowed"
except ImportError, e:
    print e

# Feature Params
sr = args.sample_rate
hops = args.hopsize
nfft = 4096


###################################################END PARSER ARGUMENT SECTION########################################

# Create context: Mapping of multiple input frames into a single target frame
def create_context(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)


###################################################INIT LOG########################################
# redirect all the stream of both standard.out and standard.err to the same logger
strID = str(args.id)

print("init log")
root_dir = path.realpath(args.root_dir)
baseResultPath = os.path.join(root_dir, 'result')
logFolder = os.path.join(baseResultPath, 'logs')
csvFolder = os.path.join(baseResultPath, 'csv')
argsFolder = os.path.join(baseResultPath, 'args')
predFolder = os.path.join(baseResultPath, 'preds')

u.makedir(logFolder)
u.makedir(csvFolder)
u.makedir(argsFolder)
u.makedir(predFolder)
u.makedir(MODEL_HISTORY_BASEPATH)
u.makedir(MODEL_MEANS_BASEPATH)
u.makedir(MODEL_WEIGHT_BASEPATH)

nameFileLog = os.path.join(logFolder, 'process_' + strID + '.log')
nameFileLogCsv = os.path.join(csvFolder, 'process_' + strID + '.csv')  # log in csv file the losses for further analysis
jsonargsFileName = os.path.join(argsFolder, 'process_' + strID + '.json')
# jsonargs = json.dumps(args.__dict__, indent=4)
#
# # with open(os.path.join(jsonargsFileName), 'w') as file:
# #     # file.write(json.dump(jsonargs, indent=4))
# #     json.dump(jsonargs, file, indent=4)

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

trainStftPath = os.path.join(args.trainset, args.input_type)

# LOAD DATASET
if 'nsynth' in args.trainset:
    if args.notes is 'all':
        notes = 'all'
    else:
        notes = dm.parsenotes(args.notes)

    #notes = [24,25] #DEBUG
    jsonPath = os.path.join(args.jsonPath)
    # with open(jsonPath, 'r', encoding='utf-8') as infile:
    with open(jsonPath, 'r') as infile:
        jsonFile = json.load(infile)
    fileslist, labels = dm.scanJson(jsonFile,
                                    instrument_family_strs=args.instrument_family_strs,
                                    notes=notes,
                                    instrument_source_strs=args.instrument_source_strs,
                                    velocityMin=args.velocityMin,
                                    velocityMax=args.velocityMax,
                                    maxNumberOfFile=args.maxNumberOfFile)

    #HARDCORE CODING (to remove 0-th element (bass family)

    a = np.ones_like(labels)*-1
    labels = labels + a

    X_train, X_val, y_train, y_val = train_test_split(fileslist, to_categorical(np.array(labels, dtype=int)),
                                                      test_size=VALIDATION_SPLIT)
else:
    print "The specified dataset is not allowed"

trainer = Trainer(X_train, X_val, y_train, y_val, model_module, args.optimizer, args.load_to_memory)
trainer.train()

ts1 = time.time()
tot_time = (ts1 - ts0) / 60
print("Experiment emplaced " + str(tot_time) + " minutes.")
print("END.")
