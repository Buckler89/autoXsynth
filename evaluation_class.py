import argparse
import importlib
import os

import numpy as np
import pandas as pd
from keras import backend as K
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.utils.np_utils import to_categorical
import librosa
import sys
from experiments.settings import *



class Evaluator(object):

    def __init__(self, model_module, weights_path, metrics=True, evaluation_strategy="s2"):
        """
        Test metadata format
        ---------------------
        filename : string
        class_ids: string of ints with space as a delimiter
        """
        #READ FROM CSV "filename;class"
        test_dataset = pd.read_csv(NSYNTH_TESTING_META_PATH, names=["filename", "class_ids"],header=None)
        self.X = list(test_dataset.filename)
        a = test_dataset.class_ids
        b = a.tolist()
        self.y_true = np.array(b, dtype=int)

        self.y_pred = np.zeros(shape=self.y_true.shape)
        self.y_pred_raw = np.zeros(shape=self.y_true.shape)
        self.y_pred_raw_average = np.zeros(shape=self.y_true.shape)
        self.model_module = model_module
        self.weights_path = weights_path

        #TODO check means
        #self.dataset_mean = np.load(os.path.join(MODEL_MEANS_BASEPATH, "{}_mean.npy".format(model_module.BASE_NAME)))

        self.evaluation_strategy = evaluation_strategy
        self.thresholds_s1 = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24]
        self.thresholds_s2 = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

        self.metrics = metrics

    def compute_prediction_scores_raw(self, model, audio_filename):
            data_for_excerpt = self._load_features(audio_filename)
            one_excerpt_prediction = model.predict_on_batch(data_for_excerpt)
            return one_excerpt_prediction
            # if self.evaluation_strategy == "s2":
            #     self.y_pred_raw[i, :] = self._compute_prediction_sum(one_excerpt_prediction)
            # else:
            #     self.y_pred_raw_average[i, :] = self._compute_prediction_sum(one_excerpt_prediction)

    # def report_metrics(self, threshold):
    #     for average_strategy in ["micro", "macro"]:
    #         sys.stdout = open(os.path.join(SCORES_PATH, 'scores_class.txt'), 'a')
    #         print("{} average strategy, threshold {}".format(average_strategy, threshold))
    #         print("precision:\t{}".format(precision_score(self.y_true, self.y_pred, average=average_strategy)))
    #         print("recall:\t{}".format(recall_score(self.y_true, self.y_pred, average=average_strategy)))
    #         print("f1:\t{}".format(f1_score(self.y_true, self.y_pred, average=average_strategy)))

    def compute_metrics(self, decisions):
        sys.stdout = open(os.path.join(SCORES_PATH, 'scores_class.txt'), 'w+')
        for average_strategy in ["micro", "macro", "weighted"]:
            print("{} average strategy: ".format(average_strategy))
            print("precision:\t{}".format(precision_score(self.y_true, decisions, average=average_strategy)))
            print("recall:\t{}".format(recall_score(self.y_true, decisions, average=average_strategy)))
            print("f1:\t{}".format(f1_score(self.y_true, decisions, average=average_strategy)))

    def evaluate(self):
        model = self.model_module.build_model(N_CLASSES)
        model.load_weights(self.weights_path)
        model.compile(optimizer="sgd", loss="categorical_crossentropy")
        preds_tot = np.zeros((1, N_CLASSES))
        for audio_filename in self.X:
            preds_seq = self.compute_prediction_scores_raw(model, audio_filename)
            preds_tot = np.concatenate((preds_tot, preds_seq), axis=0)
        preds_tot = np.delete(preds_tot, 0, 0)
        decisions = np.argmax(preds_tot,axis=1)
        np.savetxt(os.path.join(SCORES_PATH, 'predictions.txt'),  preds_tot)
        np.savetxt(os.path.join(SCORES_PATH, 'decisions.txt'), decisions)
        if self.metrics:
            self.compute_metrics(decisions)

        # if self.evaluation_strategy == "s2":
        #     for threshold in self.thresholds_s2:
        #         self.y_pred = np.copy(self.y_pred_raw)
        #         for i in range(self.y_pred.shape[0]):
        #             self.y_pred[i, :] /= self.y_pred[i, :].max()
        #         self.y_pred[self.y_pred >= threshold] = 1
        #         self.y_pred[self.y_pred < threshold] = 0
        #         self.report_metrics(threshold)
        # else:
        #     for threshold in self.thresholds_s1:
        #         self.y_pred = np.copy(self.y_pred_raw_average)
        #         self.y_pred[self.y_pred < threshold] = 0
        #         self.y_pred[self.y_pred > threshold] = 1
        #         self.report_metrics(threshold)

    def _load_features(self, filename):
        features = []
        feature_filename = os.path.join(NSYNT_TESTING_FEATURES, filename.replace('.wav', '.npy'))
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



    # def _compute_prediction_sum(self, predictions):
    #     prediction_sum = np.zeros(N_CLASSES)
    #     for prediction in predictions:
    #         prediction_sum += prediction
    #     if self.evaluation_strategy == "s1":    # simple averaging strategy
    #         prediction_sum /= predictions.shape[0]
    #     return prediction_sum


def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument("-m",
                         action="store",
                         dest="model",
                         help="-m model to evaluate")
    aparser.add_argument("-w",
                         action="store",
                         dest="weights_path",
                         help="-w path to file with weights for selected model")
    aparser.add_argument("-s",
                         action="store",
                         dest="evaluation_strategy",
                         default="s2",
                         help="-s evaluation strategy: `s1` (simple averaging and thresholding) or `s2` ("
                              "summarization, normalization by max probability and thresholding)")

    args = aparser.parse_args()

    if not (args.model and args.weights_path):
        aparser.error("Please, specify the model and the weights path to evaluate!")
    try:
        if args.model in ALLOWED_MODELS:
            model_module = importlib.import_module(".{}".format(args.model), "experiments.models")
            print "{} imported as 'model'".format(args.model)
        else:
            print "The specified model is not allowed"
        if not os.path.exists(args.weights_path):
            print "The specified weights path doesn't exist"
    except ImportError, e:
        print e

    evaluator = Evaluator(model_module, args.weights_path, args.evaluation_strategy)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
