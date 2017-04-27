#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:36:52 2017

@author: buckler
"""
import os
import numpy as np
from collections import Iterable

class Note():
    """
    d: means # (diesis)
    """
    C =  [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
    Cd = [1, 13, 25, 37, 49, 61, 73, 85, 97, 109, 121]
    D =  [2, 14, 26, 38, 50, 62, 74, 86, 98, 110, 122]
    Dd = [3, 15, 27, 39, 51, 63, 75, 87, 99, 111]
    E =  [4, 16, 28, 40, 52, 64, 76, 88, 100, 112, 124]
    F =  [5, 17, 29, 41, 53, 65, 77, 89, 101, 113, 125]
    Fd = [6, 18, 30, 42, 54, 66, 78, 90, 102, 114, 126]
    G =  [7, 19, 31, 43, 55, 67, 79, 91, 103, 115, 127]
    Gd = [8, 20, 32, 44, 56, 68, 80, 92, 104, 116]
    A =  [9, 21, 33, 45, 57, 69, 81, 93, 105, 117]
    Ad = [10, 22, 34, 46, 58, 70, 82, 94, 106, 118]
    B =  [11, 23, 35, 47, 59, 71, 83, 95, 107, 119]


class MajorKey():
    C  = [Note.C, Note.D, Note.E, Note.F, Note.G, Note.A, Note.B]
    Cd = [Note.Cd, Note.Dd, Note.F, Note.Fd, Note.Gd, Note.Ad, Note.C]
    D  = [Note.D, Note.E, Note.Fd, Note.G, Note.A, Note.B, Note.Cd]
    Dd = [Note.Dd, Note.F, Note.G, Note.Gd, Note.Ad, Note.C, Note.D]
    E  = [Note.E, Note.Fd, Note.Gd, Note.A, Note.B, Note.Cd, Note.Dd]
    F  = [Note.F, Note.G, Note.A, Note.Ad, Note.C, Note.D, Note.E]
    Fd = [Note.Fd, Note.Gd, Note.Ad, Note.B, Note.Cd, Note.Dd, Note.F]
    G  = [Note.G, Note.A, Note.B, Note.C, Note.D, Note.E, Note.Fd]
    Gd = [Note.Gd, Note.Ad, Note.C, Note.Cd, Note.Dd, Note.F, Note.G]
    A  = [Note.A, Note.B, Note.Cd, Note.D, Note.E, Note.Fd, Note.Gd]
    Ad = [Note.Ad, Note.C, Note.D, Note.Dd, Note.F, Note.G, Note.A]
    B  = [Note.B, Note.Cd, Note.Dd, Note.E, Note.Fd, Note.Gd, Note.Ad]

def parsenotes(strNoteList):
    """
    take a list of string with note (i.e. G) or key (i.e. Dd_Majkey) ad returns the midi number af all the note 
    :param strNoteList: example ['G','A','Db_MajKey']
    :return: array of int
    """
    notes = []
    for n in strNoteList:
        if n is 'A':
            notes.append(Note.A)
        if n is 'Ad':
            notes.append(Note.Ad)
        if n is 'B':
            notes.append(Note.B)
        if n is 'C':
            notes.append(Note.C)
        if n is 'Cd':
            notes.append(Note.Cd)
        if n is 'D':
            notes.append(Note.D)
        if n is 'Dd':
            notes.append(Note.D)
        if n is 'E':
            notes.append(Note.E)
        if n is 'F':
            notes.append(Note.F)
        if n is 'Fd':
            notes.append(Note.Fd)
        if n is 'G':
            notes.append(Note.G)
        if n is 'Gd':
            notes.append(Note.Gd)

        if n is 'AMaj':
            notes.append(MajorKey.A)
        if n is 'AdMaj':
            notes.append(MajorKey.Ad)
        if n is 'BMaj':
            notes.append(MajorKey.B)
        if n is 'CMaj':
            notes.append(MajorKey.C)
        if n is 'CdMaj':
            notes.append(MajorKey.Cd)
        if n is 'DMaj':
            notes.append(MajorKey.D)
        if n is 'DdMaj':
            notes.append(MajorKey.D)
        if n is 'EMaj':
            notes.append(MajorKey.E)
        if n is 'FMaj':
            notes.append(MajorKey.F)
        if n is 'FdMaj':
            notes.append(MajorKey.Fd)
        if n is 'GMaj':
            notes.append(MajorKey.G)
        if n is 'GdMaj':
            notes.append(MajorKey.Gd)

    return notes
def scanJson(jsonFile, instrument_family_strs='all', notes='all', instrument_source_strs='all', velocityMin=0, velocityMax=127, maxNumberOfFile=500):
    selectedFile = []
    single_dim_array = np.array([])  # TODO find a better way to flat all the note into sigle one dimensional array
    single_dim_array.dtype = np.int8

    if notes is not 'all': #need to put all notes in a one dimensional array
        for x in np.hstack(notes):
            if isinstance(x, Iterable):
                for y in x:

                    single_dim_array = np.append(single_dim_array, y)
            else:
                single_dim_array = np.append(single_dim_array, x)
    i=0
    for key, value in jsonFile.items():
            if value['instrument_family_str'] in instrument_family_strs or instrument_family_strs == 'all':

                if value['pitch'] in single_dim_array or notes is 'all':

                    if value['instrument_source_str'] in instrument_source_strs or instrument_source_strs == 'all':

                        if value['velocity'] >= velocityMin and value['velocity'] <= velocityMax:

                            selectedFile.append(key+'.npy')
                            i+=1
                            if i >= maxNumberOfFile:
                                break


    return selectedFile


# def __init__(id):
#     global logger
#     logger = logging.getLogger(str(id))
#     # def crateLogger(id, logToFile):
#     #     global logger
#     #     logger = u.MyLogger(id, logToFile)

def load_DATASET(datasetPath, fileslist=None, verbose=False):
    """
    Carica tutto il dataset (spettri) in una lista di elementi [filename , matrix ]
    """
    print("Loading dataset from " + datasetPath)
    dataset = list()
    if fileslist is None:
        print('load all file')

        for root, dirnames, filenames in os.walk(datasetPath):
            i = 0
            for file in filenames:
                if verbose:
                    print('load ' + file)
                matrix = np.load(os.path.join(root, file))
                data = [file, matrix]
                dataset.append(data)
                i += 1
    else:
        i = 0
        print('load file from list')
        for file in fileslist:
            if verbose:
                print('load '+file)
            matrix = np.load(os.path.join(datasetPath, file))
            data = [file, matrix]
            dataset.append(data)
            i += 1
    print('{0} file loaded'.format(i))
    return dataset



def awgn_padding_set(set_to_pad, loc=0.0, scale=1.0):
    print("awgn_padding_set")

    # find matrix with biggest second axis
    dim_pad = np.amax([len(k[1][2]) for k in set_to_pad])
    awgn_padded_set = []
    for e in set_to_pad:
        row, col = e[1].shape
        # crete an rowXcol matrix with awgn samples
        awgn_matrix = np.random.normal(loc, scale, size=(row, dim_pad - col))
        awgn_padded_set.append([e[0], np.hstack((e[1], awgn_matrix))])
    return awgn_padded_set


def reshape_set(set_to_reshape, net_type, channels=1):
    """
    reshape the data in a form that keras want:
        -for theano dim ordering: (nsample, channel, row ,col)
        -for tensorflow not supported yet
        -other not specified yet
    :param set_to_reshape:
    :param net_type: is the first type of layer used in the model
    :param channels:
    :return:
    """
    print("reshape_set")

    if net_type is 'convolutional2d':
        n_sample = len(set_to_reshape)
        row, col = set_to_reshape[0][1].shape
        label = []
        shaped_matrix = np.empty((n_sample, channels, row, col))
        for i in range(len(set_to_reshape)):
            label.append(set_to_reshape[i][0])
            shaped_matrix[i][0] = set_to_reshape[i][1]

    if net_type is 'dense':
        #n_sample = len(set_to_reshape)
        #row, col = set_to_reshape[0][1].shape
        label = []
        featSize = set_to_reshape[0][1].shape[0]
        type = set_to_reshape[0][1].dtype
        shaped_matrix = np.empty([1, featSize],dtype=type)
        for sample in (set_to_reshape):
            shaped_matrix = np.vstack([shaped_matrix, sample[1].T])
        shaped_matrix = np.delete(shaped_matrix, 0, 0)

    return shaped_matrix, label


def split_A3FALL_simple(data, train_tag=None):
    '''
    Splitta il dataset in train e test set: train tutti i background, mentre test tutto il resto
    (da amplicare in modo che consenta lo split per la validation)
    '''
    print("split_A3FALL_simple")

    if train_tag == None:
        train_tag = ['classic_', 'rock_', 'ha_']
    #        if test_tag=None:
    #            test_tag=[]

    data_train = [d for d in data if any(word in d[0] for word in
                                         train_tag)]  # controlla se uno dei tag è presente nnel nome del file e lo assegna al trainset
    data_test = [d for d in data if d not in data_train]  # tutto cioò che non è train diventa test

    return data_train, data_test


def split_A3FALL_from_lists(data, listpath, namelist):
    '''
    Richede in ingresso la cartella dove risiedono i file di testo che elencano i vari segnali che farano parte di un voluto set di dati.
    Inltre in namelist vanno specificati i nomi dei file di testo da usare.
    Ritorna una lista contentete le liste dei dataset di shape: (len(namelist),data.shape)
    '''
    print("split_A3FALL_from_lists")

    sets = list()
    for name in namelist:
        sets.append(select_list(os.path.join(listpath, name), data))
    return sets


def select_list(filename, dataset):
    '''
    Dato in ingesso un file di testo, resituisce una array contenete i dati corrispondenti elencati nel file
    '''
    print("select_list")

    subset = list()
    with open(filename) as f:
        content = f.readlines()
        content = [x.strip().replace('.wav', '.npy') for x in content]  # remove the '\n' at the end of string
        subset = [s for s in dataset if
                  any(name in s[0] for name in content)]  # select all the data presetn in the list
    return subset


def normalize_data(data, mean=None, std=None):
    '''
    normalizza media e varianza del dataset passato
    se data=None viene normalizzato tutto il dataset A3FALL
    se mean e variance = None essi vengono calcolati in place sui data
    '''
    print("normalize_data")

    if bool(mean) ^ bool(std):  # xor operator
        raise ("Error!!! Provide both mean and variance")
    elif mean == None and std == None:  # compute mean and variance of the passed data
        data_conc = concatenate_matrix(data)
        mean = np.mean(data_conc)
        std = np.std(data_conc)

    data_std = [[d[0], ((d[1] - mean) / std)] for d in data]  # normalizza i dati: togle mean e divide per std

    return data_std, mean, std


def concatenate_matrix(data):
    """
    concatena gli spettri in un unica matrice: vule una lista e restituisce un array

    :param data:
    :return:
    """

    print("concatenate_matrix")

    data_ = data.copy()
    data_.pop(0)
    matrix = data[0][1]
    for d in data_:
        np.append(matrix, d[1], axis=1)
    return matrix


def labelize_data(y):#TODO usare sklearn.preprocessing.LabelEncoder ?
    """
    labellzza numericamente i nomi dei file
    assegna 1 se è una caduta del manichino, 0 altrimenti
    :param y:
    :return:

    """
    print("labelize_data")

    i = 0
    numeric_labels = list()
    for d in y:
        if 'rndy' in d:
            numeric_labels.append(1)
        else:
            numeric_labels.append(0)
        i += 1

    return numeric_labels