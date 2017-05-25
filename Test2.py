from __future__ import division
from AudioTool import *
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
import numpy as np
import sys


#audio_files_path = '/media/fabio/DATA/Work/GenerativeModels/wavs/Ableton-GrandPiano/'
SCORES_PATH = "./"

test_dataset = pd.read_csv('list_LEO_REC.txt', names=["filename", "class_ids","note"],header=None)
files_list = list(test_dataset.filename)
instrument = test_dataset.class_ids
note = test_dataset.note.tolist()


#test = AudioTool(sys='stft', n_fft=8192, hop=2760, path=audio_files_path, s_rate=22050)
test = AudioTool(sys='pip', n_fft=8192, hop=512, s_list=files_list)
preds = test.audio_analisys()

tp = 0
for n in xrange(0, len(preds)):
    if preds[n] == note[n]:
        tp += 1
    elif preds[n] == note[n]+12:
        tp += 1
    elif preds[n] == note[n]-12:
        tp += 1
    elif preds[n] == note[n]+1:
        tp += 1
    elif preds[n] == note[n]-1:
        tp += 1
accuracy = (tp / (len(preds)))*100
print("Pitch detection accuracy:\t{:.2f}%".format(accuracy))


# labels = np.arange(0, 128)
# sys.stdout = open(os.path.join(SCORES_PATH, 'scores_class.txt'), 'w+')
# print ("Pitch detection with Piptrack\n")
# print("Pitch detection accuracy:\t{:.2f}%".format(accuracy))
# for average_strategy in ["micro", "weighted"]:
#     print("{} average strategy: ".format(average_strategy))
#     print("precision:\t{}".format(precision_score(note, preds, average=average_strategy, labels=labels)))
#     print("recall:\t{}".format(recall_score(note, preds,average=average_strategy, labels=labels)))
#     print("f1:\t{}\n\n".format(f1_score(note, preds, average=average_strategy, labels=labels)))
# CM = np.asarray(confusion_matrix(note, preds, labels=labels), dtype=int)
# np.savetxt(os.path.join(SCORES_PATH, 'CM.txt'), CM)
# print(classification_report(note, preds, labels=labels))
