import json
from enum import Enum
import numpy as np
from collections import Iterable


def scanJson(jsonFile, instrument_family_strs='all', notes='all', instrument_source_strs='all'):
    selectedFile = []
    single_dim_array = np.array([])  # TODO find a better way to flat all the note into sigle one dimensional array

    if notes is not 'all':
        for x in np.hstack(notes):
            if isinstance(x, Iterable):
                for y in x:
                    single_dim_array = np.append(single_dim_array, y)
            else:
                single_dim_array = np.append(single_dim_array, x)

    for key, value in jsonFile.items():
        if value['instrument_family_str'] in instrument_family_strs or instrument_family_strs == 'all':

            if value['pitch'] in single_dim_array or notes is 'all':

                if value['instrument_source_str'] in instrument_source_strs or instrument_source_strs == 'all':
                    selectedFile.append(key+'.npy')
    return selectedFile


jsonFilePath = '/media/buckler/DataSSD/Phd/autoXsynthImproved/autoXsynth/dataset/train/NSynth-train/e.json'
with open(jsonFilePath, 'r', encoding='utf-8') as infile:
    jsonFile = json.load(infile);


# TODO keys to note
class Note():
    """
    d: means # (diesis)
    """
    C =  np.array([0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120])
    Cd = np.array([1, 13, 25, 37, 49, 61, 73, 85, 97, 109, 121])
    D =  np.array([2, 14, 26, 38, 50, 62, 74, 86, 98, 110, 122])
    Dd = np.array([3, 15, 27, 39, 51, 63, 75, 87, 99, 111])
    E =  np.array([4, 16, 28, 40, 52, 64, 76, 88, 100, 112, 124])
    F =  np.array([5, 17, 29, 41, 53, 65, 77, 89, 101, 113, 125])
    Fd = np.array([6, 18, 30, 42, 54, 66, 78, 90, 102, 114, 126])
    G =  np.array([7, 19, 31, 43, 55, 67, 79, 91, 103, 115, 127])
    Gd = np.array([8, 20, 32, 44, 56, 68, 80, 92, 104, 116])
    A =  np.array([9, 21, 33, 45, 57, 69, 81, 93, 105, 117])
    Ad = np.array([10, 22, 34, 46, 58, 70, 82, 94, 106, 118])
    B =  np.array([11, 23, 35, 47, 59, 71, 83, 95, 107, 119])


class MajorKey():
    C  = np.array([Note.C, Note.D, Note.E, Note.F, Note.G, Note.A, Note.B])
    Cd = np.array([Note.Cd, Note.Dd, Note.F, Note.Fd, Note.Gd, Note.Ad, Note.C])
    D  = np.array([Note.D, Note.E, Note.Fd, Note.G, Note.A, Note.B, Note.Cd])
    Dd = np.array([Note.Dd, Note.F, Note.G, Note.Gd, Note.Ad, Note.C, Note.D])
    E  = np.array([Note.E, Note.Fd, Note.Gd, Note.A, Note.B, Note.Cd, Note.Dd])
    F  = np.array([Note.F, Note.G, Note.A, Note.Ad, Note.C, Note.D, Note.E])
    Fd = np.array([Note.Fd, Note.Gd, Note.Ad, Note.B, Note.Cd, Note.Dd, Note.F])
    G  = np.array([Note.G, Note.A, Note.B, Note.C, Note.D, Note.E, Note.Fd])
    Gd = np.array([Note.Gd, Note.Ad, Note.C, Note.Cd, Note.Dd, Note.F, Note.G])
    A  = np.array([Note.A, Note.B, Note.Cd, Note.D, Note.E, Note.Fd, Note.Gd])
    Ad = np.array([Note.Ad, Note.C, Note.D, Note.Dd, Note.F, Note.G, Note.A])
    B  = np.array([Note.B, Note.Cd, Note.Dd, Note.E, Note.Fd, Note.Gd, Note.Ad])


selectedFileList = scanJson(jsonFile, instrument_family_strs=['guitar'], notes=[82],
                            instrument_source_strs=['acoustic'])
selectedFileList2 = scanJson(jsonFile)
x = MajorKey.Gd
print()
