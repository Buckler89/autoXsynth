#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 3 11:19:08 2017

@author: daniele
"""

import argparse
import scipy.stats
import numpy as np
import math
import os
from shutil import copyfile
import utility as u

# container class for single experiments
class experiment:
    pass


class choices:
    def __init__(self, values):
        self.values = values
    def rvs(self):
        return np.random.choice(self.values)


class loguniform_gen:
    def __init__(self, base=2, low=0, high=1, round_exponent=False, round_output=False):
      self.base = base
      self.low = low
      self.high = high
      self.round_exponent = round_exponent
      self.round_output = round_output
    def rvs(self):
      exponent = scipy.stats.uniform.rvs(loc=self.low, scale=(self.high-self.low))
      if self.round_exponent:
        exponent = np.round(exponent)
      value = np.power(self.base, exponent)
      if self.round_output:
        value = int(np.round(value))
      return value


class eval_action(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(eval_action, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        values = eval(values)
        setattr(namespace, self.dest, values)


parser = argparse.ArgumentParser(description="Novelty Deep Fall Detection")

# Global params
parser.add_argument("-rp", "--root-path", dest="root_path", default=None)
parser.add_argument("-log", "--logging", dest="log", default=False, action="store_true")
parser.add_argument("-ss", "--search-strategy", dest="search_strategy", default="grid", choices=["grid", "random"])
parser.add_argument("-rnd", "--rnd-exp-number", dest="N", default=0, type=int)
parser.add_argument("-cf", "--config-file", dest="config_filename", default=None)
parser.add_argument("-sp", "--score-path", dest="scorePath", default="score")
parser.add_argument("-shp", "--script-path", dest="scriptPath", default="scripts")
parser.add_argument("-it", "--input-type", dest="input_type", default="stft", choices=["stft"])
# dense params
parser.add_argument("-nb", "--bias", dest="bias", default=[True], action=eval_action)
parser.add_argument("-da", "--dense-activation", dest="dense_activation", action=eval_action, default=["tanh"])
parser.add_argument("-dln", "--dense-layers-numb", dest="dense_layer_numb", action=eval_action, default=[1])
parser.add_argument("-ds", "--dense-shapes", dest="dense_shapes", action=eval_action, default=[64])
parser.add_argument("-dst", "--dense-shape-type", dest="dense_shape_type", default="any")
parser.add_argument("-dwr", "--d-w-reg", dest="d_w_reg", action=eval_action, default=None) # in autoencoder va usato con eval("funz(parametri)")
parser.add_argument("-dbr", "--d-b-reg", dest="d_b_reg", action=eval_action, default=None)
parser.add_argument("-dar", "--d-act-reg", dest="d_a_reg", action=eval_action, default=None)
parser.add_argument("-dwc", "--d-w-constr", dest="d_w_constr", action=eval_action, default=None)
parser.add_argument("-dbc", "--d-b-constr", dest="d_b_constr", action=eval_action, default=None)
parser.add_argument("-drp", "--dropout", dest="dropout", default=[False], action=eval_action)
parser.add_argument("-drpr", "--drop-rate", dest="drop_rate", default=0.5, type=float)

# fit params
parser.add_argument("-f", "--fit-net", dest="fit_net", default=False, action="store_true")
parser.add_argument("-e", "--epoch", dest="epoch", default=50, type=int)
parser.add_argument("-ns", "--shuffle", dest="shuffle", default=[True], action=eval_action)
parser.add_argument("-bs", "--batch-size-fract", dest="batch_size_fract", action=eval_action, default=["1/10"])
parser.add_argument("-o", "--optimizer", dest="optimizer", action=eval_action, default=["adadelta"])
parser.add_argument("-l", "--loss", dest="loss", action=eval_action, default=["msle"])
parser.add_argument("-lr", "--learning-rate", dest="learning_rate", action=eval_action, default=[1.0])
#parser.add_argument("-pt", "--patience", dest="patience", default=20, type=int)
parser.add_argument("-vl", "--validation-split", dest="val_split", default=0.0, type=float)


args = parser.parse_args()

if args.config_filename is not None:
    with open(args.config_filename, "r") as f:
        lines = f.readlines()
    arguments = []
    for line in lines:
        arguments.extend(line.split("#")[0].split())
    # First parse the arguments specified in the config file
    args = parser.parse_args(args=arguments)
    # Then append the command line arguments
    # Command line arguments have the priority: an argument is specified both
    # in the config file and in the command line, the latter is used
    args = parser.parse_args(namespace=args)



############################################ def grid_search:
def grid_search(args):
    exp_list = []
    n=0

    for cln in args.conv_layer_numb:
        for kn in args.kernel_number:
            for ks in args.kernel_shape:
                for mp in args.m_pool:
                    for p in args.pool_type:
                        for s in args.strides:
                            for dln in args.dense_layer_numb:
                                for ds in args.dense_shapes:
                                    for ci in args.cnn_init:
                                        for ac in args.cnn_conv_activation:
                                            for ad in args.dense_activation:
                                                for bm in args.border_mode:
                                                    for bs in args.batch_size_fract:
                                                        for o in args.optimizer:
                                                            for l in args.loss:
                                                                for nb in args.bias:
                                                                    for ns in args.shuffle:
                                                                        # only sane combination
                                                                        if len(kn) == len(ks) == len(s) == len(mp) == cln:
                                                                            if len(ds) == dln:
                                                                                e = experiment()
                                                                                e.id = n
                                                                                n += 1
                                                                                e.cnn_input_shape = args.cnn_input_shape
                                                                                e.conv_layer_numb = cln
                                                                                e.kernel_number = kn
                                                                                e.kernel_shape = ks
                                                                                e.strides = s
                                                                                e.m_pool = mp
                                                                                e.pool_type = p
                                                                                e.cnn_init = ci
                                                                                e.cnn_conv_activation = ac
                                                                                e.dense_activation = ad
                                                                                e.border_mode = bm
                                                                                e.dense_layer_numb = dln
                                                                                e.dense_shapes = ds
                                                                                e.batch_size_fract = bs
                                                                                e.optimizer = o
                                                                                e.loss = l
                                                                                e.shuffle = ns
                                                                                e.bias = nb

                                                                                exp_list.append(e)

    return exp_list


def gen_with_shape_tie(rng, tie_type, distribution="uniform", base=2, round_exponent=False, round_output=True):
    ack=False
    if distribution == "uniform":
        gen_rows = choices(range(rng[0], rng[1] + 1))
        gen_cols = choices(range(rng[2], rng[3] + 1))
    elif distribution == "loguniform":
        gen_rows = loguniform_gen(base, rng[0], rng[1], round_exponent, round_output)
        gen_cols = loguniform_gen(base, rng[2], rng[3], round_exponent, round_output)
    while not ack:
        rows = gen_rows.rvs()
        cols = gen_cols.rvs()
        if rows == cols and "square" in tie_type or \
            rows <= cols and "+cols" in tie_type or \
            rows >= cols and "+rows" in tie_type or \
            tie_type == "any":
            ack = True
    return [rows, cols]


def gen_with_ties(dim, num, bounds, tie_type, distribution="uniform", base=2, round_exponent=False, round_output=True):
    ties = tie_type.split(",")
    if dim == 1:
        if distribution == "uniform":
            gen = choices(range(bounds[0], bounds[1] + 1))
        elif distribution == "loguniform":
            gen = loguniform_gen(base, bounds[0], bounds[1], round_exponent, round_output)
        if "equal" in ties:
            v = [gen.rvs()] * num
        else:
            v = [gen.rvs()]
            for j in range(1, num):
                ack = False
                c = [gen.rvs()]
                while not ack:
                    if c[0] <= v[j-1] and "decrease" in ties or \
                       c[0] >= v[j-1] and "encrease" in ties or \
                       "any" in ties:
                        ack = True
                    else:
                        c = [gen.rvs()]
                v.extend(c)
    elif dim == 2:
        if ties[1] == ties[2] =="equal":
            v = [gen_with_shape_tie(bounds, ties[0], distribution, base, round_exponent, round_output)] * num
        else:
            v = [gen_with_shape_tie(bounds, ties[0], distribution, base, round_exponent, round_output)]
            for j in range(1, num):
                ack = False
                while not ack:
                    c = gen_with_shape_tie(bounds, ties[0], distribution, base, round_exponent, round_output)
                    ack = True
                    if c[0] > v[j-1][0] and ties[1] == "decrease" or \
                       c[0] < v[j-1][0] and ties[1] == "encrease":
                        ack = False
                    if c[1] > v[j-1][1] and "decrease" in tie_type or \
                       c[1] < v[j-1][1] and "encrease" in tie_type:
                        ack = False
                v.append(c)
    return v


def random_search(args):
    exp_list = []
    for n in range(args.N):
        e = experiment()
        e.id = n

        ################################################################################### Danse layers
        dense_layer_numb = np.random.choice(range(args.dense_layer_numb[0], args.dense_layer_numb[1] + 1))
        e.dense_layer_numb = dense_layer_numb
        e.dense_shapes = gen_with_ties(1, dense_layer_numb, np.log2(args.dense_shapes), args.dense_shape_type, "loguniform", 2)
        e.dense_activation = np.random.choice(args.dense_activation)
        e.dropout = True
        e.drop_rate = args.drop_rate
        ################################################################################### Learning params
        e.shuffle = np.random.choice(args.shuffle)
        e.optimizer = np.random.choice(args.optimizer)
        e.loss = np.random.choice(args.loss)
        b_size = [eval(args.batch_size_fract[0]),eval(args.batch_size_fract[1])]
        e.batch_size_fract = gen_with_ties(1, 1, np.log2(b_size), "any", "loguniform", 2, False, False)[0]
        e.learning_rate = 1 #gen_with_ties(1, 1, np.log10(args.learning_rate), "any", "loguniform", 10, False, False)[0]
        e.shuffle = np.random.choice(args.shuffle)
        e.bias = np.random.choice(args.bias)


        exp_list.append(e)
    return exp_list

############################################ inizializzalizzazioni
experiments = []
if args.root_path is None:
    args.root_path = os.path.realpath(".")

u.makedir(os.path.join(args.root_path, args.scriptPath))

############################################ creazione della lista dei parametri secondo la strategia scelta
if args.search_strategy == "grid":
    print("define point of grid search")
    experiments = grid_search(args)
elif args.search_strategy == "random":
    print("define point of random search")
    experiments = random_search(args)

############################################ creazione dei file per lo scheduler
i=0
for e in experiments:
    script_name = os.path.join(args.scriptPath, str(i).zfill(5) + "_audioXsynth.conf")
    command = " --root-path " +''+ str(args.root_path)+ '' + "\n" + \
              " --input-type " + str(args.input_type) + "\n" + \
              " --exp-index " + str(e.id) + "\n" + \
              " --dense-activation " + str(e.dense_activation) + "\n" + \
              " --d-w-reg " + str(args.d_w_reg) + "\n" + \
              " --d-b-reg " + str(args.d_b_reg) + "\n" + \
              " --d-act-reg " + str(args.d_a_reg) + "\n" + \
              " --d-w-constr " + str(args.d_w_constr) + "\n" + \
              " --d-b-constr " + str(args.d_b_constr) + "\n" + \
              " --dense-shape " + str(e.dense_shapes).replace(" ", "") + "\n" + \
              " --epoch " + str(args.epoch) + "\n" + \
              " --batch-size " + str(e.batch_size_fract) + "\n" + \
              " --optimizer " + str(e.optimizer) + "\n" + \
              " --loss " + str(e.loss) + "\n" + \
              " --drop-rate " + str(e.drop_rate) + "\n" + \
              " --learning-rate " + str(e.learning_rate) + "\n"

    if not e.shuffle:
        command += " --no-shuffle \n"
    if not e.bias:
        command += " --no-bias \n"
    if args.log:
        command += " --logging \n"
    if args.fit_net:
        command += " --fit-net \n"
    if e.dropout:
        command += " --dropout \n"

    with open(script_name, "a") as f:
        f.write(command)

    i+=1

np.save(os.path.join(os.getcwd(), "experiments.npy"), experiments)
