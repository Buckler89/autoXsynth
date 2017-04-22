import json
from enum import Enum
import numpy as np

def scanJson(jsonFile, instrument_family_strs='all', notes='all', instrument_source_strs='all'):
    selectedFile = []

    if notes is not 'all':
        notesArray = np.hstack(notes)
    else:
        notesArray = np.array([-1]);

    for key, value in jsonFile.items():
        if value['instrument_family_str'] in instrument_family_strs or instrument_family_strs == 'all':

            if value['pitch'] in notesArray or notes is 'all':

                if value['instrument_source_str'] in instrument_source_strs or instrument_source_strs == 'all':
                    selectedFile.append(key)

                    # for subkey, subvalue in value.items():
                    # print(str(key)+'----------------'+str(value))
    return selectedFile


jsonFilePath = '/media/buckler/Data/Download/nsynth_train_dataset/nsynth-train/e.json'
with open(jsonFilePath, 'r', encoding='utf-8') as infile:
    jsonFile = json.load(infile);


# TODO keys to note
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





selectedFileList = scanJson(jsonFile, instrument_family_strs=['guitar'], notes=MajorKey.Gd,
                            instrument_source_strs=['acoustic'])
selectedFileList2 = scanJson(jsonFile)
x = MajorKey.Gd
print();
