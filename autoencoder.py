#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:43:32 2017

@author: buckler
"""

import os

from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, Dense, Dropout, Flatten, Reshape, Convolution2D, MaxPooling2D, UpSampling2D, \
    ZeroPadding2D, Cropping2D, Concatenate, BatchNormalization, LSTM, SimpleRNN, GRU
from tensorflow.python.keras.optimizers import Adam, Adadelta, SGD
from tensorflow.python.keras.callbacks import Callback, ProgbarLogger, CSVLogger, EarlyStopping, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, f1_score
import matplotlib
import math
from scipy.spatial.distance import euclidean
import dataset_manupulation as dm
import json
import utility as u


# import matplotlib.image as img

def compute_distances(x_test, decoded_images):#TODO adatta per i channel > 1
    """
    calcola le distanze euclide tra 2 vettori di immagini con shape (n_img,1,row,col)
    ritorna un vettore con le distanze con shape (n_img,1)
    """
    print("compute_distance")

    # e_d2d = np.zeros(x_test.shape)
    e_d = np.zeros(x_test.shape[0])

    for i in range(decoded_images.shape[0]):
        # e_d2d[i,0,:,:] = euclidean_distances(decoded_images[i,0,:,:],x_test[i,0,:,:])
        # e_d[i] = euclidean_distances(decoded_images[i,0,:,:],x_test[i,0,:,:]).sum()
        e_d[i] = euclidean(decoded_images[i, 0, :, :].flatten(), x_test[i, 0, :, :].flatten())

    return e_d


def compute_optimal_th(fpr, tpr, thresholds, method='std'):
    """
    http://medind.nic.in/ibv/t11/i4/ibvt11i4p277.pdf
    ci sono molti metodi per trovare l ottima th:
        1-'std' minumum of distances from point (0,1)
            min(d^2), d^2=[(0-fpr)^2+(1-tpr)^2]
        2-'xxx' definire delle funzioni costo TODO
    """
    if method == 'std':
        indx = ((0 - fpr) ** 2 + (1 - tpr) ** 2).argmin()
        optimal_th = thresholds[indx]
        return optimal_th, indx


def compute_score(original_image, decoded_images, labels):
    """

    :param original_image:
    :param decoded_images:
    :param labels:
    :return:
    """
    print("compute_score")

    true_numeric_labels = dm.labelize_data(labels)
    euclidean_distances = compute_distances(original_image, decoded_images)

    fpr, tpr, roc_auc, thresholds = ROCCurve(true_numeric_labels, euclidean_distances, pos_label=1,
                                             makeplot='no', opt_th_plot='yes') #TODO need a ebetter system for makeplot
    if max(fpr) != 1 or max(tpr) != 1 or min(fpr) != 0 or min(
            tpr) != 0:  # in teoria questi mi e max dovrebbero essere sempre 1 e 0 rispettivamente
        print("max min tpr fpr error")
    optimal_th, indx = compute_optimal_th(fpr, tpr, thresholds, method='std')
    ROCCurve(true_numeric_labels, euclidean_distances, indx, pos_label=1, makeplot='no', opt_th_plot='yes')

    # compute tpr fpr fnr tnr metrics
    #        npoint=5000
    #        minth=min(euclidean_distances)
    #        maxth=max(euclidean_distances)
    #        step=(maxth-minth)/npoint
    #        ths=np.arange(minth,maxth,step)
    #        tp=np.zeros(len(ths))
    #        fn=np.zeros(len(ths))
    #        tn=np.zeros(len(ths))
    #        fp=np.zeros(len(ths))
    #
    #        k=0
    #        for th in ths:
    #            i=0
    #            for d in euclidean_distances:
    #                if d > th:
    #                    if true_numeric_labels[i]==1:
    #                        tp[k]+=1
    #                    else:
    #                        fp[k]+=1
    #                else:
    #                    if true_numeric_labels[i]==1:
    #                        fn[k]+=1
    #                    else:
    #                        tn[k]+=1
    #                i+=1
    #            k+=1
    #        tpr=tp/(tp+fn)
    #        tnr=tn/(tn+fp)
    #        fpr=fp/(fp+tn)
    #        fnr=fn/(fn+tp)
    # ---------------------------DET----------------------

    # DETCurve(fpr,fnr)

    # ---------------------------myROC----------------------

    #        plt.figure()
    #        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    #        plt.plot([0, 1], [0, 1], 'k--')
    #        plt.xlim([0.0, 1.0])
    #        plt.ylim([0.0, 1.05])
    #        plt.xlabel('False Positive Rate')
    #        plt.ylabel('True Positive Rate')
    #        plt.title('Receiver operating characteristic')
    #        plt.legend(loc="lower right")
    #        plt.show()

    # --------------------------CONFUSION MATRIX---------------------
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    i = 0
    y_pred = np.zeros(len(euclidean_distances))
    for d in euclidean_distances:
        if d > optimal_th:
            y_pred[i] = 1
            if true_numeric_labels[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            y_pred[i] = 0
            if true_numeric_labels[i] == 1:
                fn += 1
            else:
                tn += 1
        i += 1
    # tpr=tp/(tp+fn)
    #        tnr=tn/(tn+fp)
    #        fpr=fp/(fp+tn)
    #        fnr=fn/(fn+tp)
    print("confusion matrix:")
    # sk_cm=confusion_matrix(true_numeric_labels,y_pred)
    my_cm = np.array([[tp, fn], [fp, tn]])
    print("\t Fall \t NoFall")
    print("Fall \t" + str(tp) + "\t" + str(fn))
    print("NoFall \t" + str(fp) + "\t" + str(tn))
    print("F1measure: " + str(f1_score(true_numeric_labels, y_pred, pos_label=1)))
    print(classification_report(true_numeric_labels, y_pred, target_names=['NoFall', 'Fall']))

    return roc_auc, optimal_th, my_cm, true_numeric_labels, y_pred


def ROCCurve(y_true, y_score, indx=None, pos_label=1, makeplot='yes', opt_th_plot='no'):
    print("roc curve:")
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    if makeplot == 'yes':
        # Plot of a ROC curve for a specific class

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        if opt_th_plot == 'yes' and indx != None:
            plt.plot(fpr[indx], tpr[indx], 'ro')
        plt.show()

    return fpr, tpr, roc_auc, thresholds


def DETCurve(fpr, fnr):
    """
    Given false positive and false negative rates, produce a DET Curve.
    The false positive rate is assumed to be increasing while the false
    negative rate is assumed to be decreasing.
    """
    print("DETCurve")
    plt.figure()
    # axis_min = min(fps[0],fns[-1])
    fig, ax = plt.subplots(figsize=(10, 10), dpi=600)
    plt.plot(fpr, fnr)
    plt.yscale('log')
    plt.xscale('log')
    ticks_to_use = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)
    plt.axis([0.001, 50, 0.001, 50])
    plt.show()


def print_score(cm, y_pred, y_true):
    """
    print the final results for the all fold test
    """
    cm = cm.astype(int)
    print("FINAL REPORT")
    print("\t\t Fall \t NoFall")
    print("Fall \t" + str(cm[0, 0]) + "\t" + str(cm[0, 1]))
    print("NoFall \t" + str(cm[1, 0]) + "\t" + str(cm[1, 1]))

    print("F1measure: " + str(f1_score(y_true, y_pred, pos_label=1)))
    print(classification_report(y_true, y_pred, target_names=['NoFall', 'Fall']))


def plot_decoded_imgs(original, decoded_imgs, n=2):
   #plot reconstructed image ( mnist_dataset only )

    plt.figure(figsize=(20*4, 4*4))
    for i in range(1, n):
        # display original
        ax = plt.subplot(2, n, i)
        plt.imshow(original[i].reshape(129, 197))
        #plt.gray()
        plt.colors()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(129, 197))
        plt.colors()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def plot_decoded_img(label, original, decoded_img, destSavePath):
    # plot reconstructed image ( mnist_dataset only )
    plt.ioff() #turn off interactive mode otherwise plt plot the figure when it wants

    fig = plt.figure()
    fig.suptitle(label)
    # display original
    ax = plt.subplot(2, 1, 1)
    plt.imshow(original.reshape(129, 197))
    # plt.gray()
    plt.colors()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, 1, 2)
    plt.imshow(decoded_img.reshape(129, 197))
    plt.colors()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #plt.show()

    fig.savefig(destSavePath)
    plt.close(fig)
    plt.ion()#turn on interactive mode

class autoencoder_fall_detection:
    def __init__(self, id_process, case=None, fold=None, model=0):
        """

        :param id: The id of the experiment. Is also the name of the logger that must be used!
        :param fit: useful in debug mode, if there is a model already fitted
        """
        print("__init__")
        self._autoencoder = model
        self._case = case
        self._id = id_process
        self._fold = fold

    def define_static_arch(self):
        """
        E' TEMPORANEA:QUESTA FUNZIONE VA ELIMINATA ALLA FINE
        QUESTa è usata solo per bypassare la creazione dinamica che vuole tutti i parametri!
        """
        print('define TEST arch ')
        ks = [3, 3]  # serve solo su def_static_arch
        nk = [16, 8, 8]  # serve solo su def_static_arch
        input_img = Input(shape=(1, 129, 197))

        x = Convolution2D(nk[0], ks[0], ks[1], activation='tanh', border_mode='same')(input_img)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(nk[1], ks[0], ks[1], activation='tanh', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(nk[2], ks[0], ks[1], activation='tanh', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        # at this point the representation is (8, 4, 4) i.e. 128-dimensional

        x = Flatten()(x)
        x = Dense(3400, activation='tanh')(x)
        encoded = Dense(64, activation='tanh')(x)
        # -------------------------------------
        x = Dense(3400, activation='tanh')(encoded)
        x = Reshape((8, 17, 25))(x)

        x = Convolution2D(nk[2], ks[0], ks[1], activation='tanh', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(nk[1], ks[0], ks[1], activation='tanh', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(nk[0], ks[0], ks[1], activation='tanh')(x)
        x = UpSampling2D((2, 2))(x)
        x = ZeroPadding2D(padding=(0, 0, 0, 1))(x)
        x = Cropping2D(cropping=((1, 2), (0, 0)))(x)

        decoded = Convolution2D(1, ks[0], ks[1], activation='tanh', border_mode='same')(x)

        #        layer1 = Model(input_img, decoded)
        #        layer1.summary()
        self._autoencoder = Model(input_img, decoded)

        return self._autoencoder

    def define_cnn_arch(self, params):
        print("define_arch")
        # ---------------------------------------------------------- Encoding
        d = params.cnn_input_shape[0]
        h = params.cnn_input_shape[1]
        w = params.cnn_input_shape[2]
        print("(" + str(d) + ", " + str(h) + ", " + str(w) + ")")

        input_img = Input(shape=params.cnn_input_shape)
        x = input_img

        for i in range(len(params.kernel_number)):
            x = Convolution2D(params.kernel_number[i],
                              params.kernel_shape[i][0],
                              params.kernel_shape[i][1],
                              kernel_initializer=params.cnn_init,
                              activation=params.cnn_conv_activation,
                              border_mode=params.border_mode,
                              subsample=tuple(params.strides[i]),
                              kernel_regularizer=eval(params.cnn_w_reg),
                              bias_regularizer=eval(params.cnn_b_reg),
                              activity_regularizer=eval(params.cnn_a_reg),
                              kernel_constraint=eval(params.cnn_w_constr),
                              bias_constraint=eval(params.cnn_b_constr),
                              bias=params.bias)(x)

            if params.border_mode == 'same':
                ph = params.kernel_shape[i][0] - 1
                pw = params.kernel_shape[i][1] - 1
            else:
                ph = pw = 0
            h = int((h - params.kernel_shape[i][0] + ph) / params.strides[i][0]) + 1
            w = int((w - params.kernel_shape[i][1] + pw) / params.strides[i][1]) + 1
            d = params.kernel_number[i]
            print("conv(" + str(i) + ") -> (" + str(d) + ", " + str(h) + ", " + str(w) + ")")

            if params.pool_type=="all":
                x = MaxPooling2D(params.m_pool[i], border_mode='same')(x)
                # if MaxPooling border=='valid' h=int(h/params.params.m_pool[i][0])
                h = math.ceil(h / params.m_pool[i][0])
                w = math.ceil(w / params.m_pool[i][1])
                print("pool(" + str(i) + ") -> (" + str(d) + ", " + str(h) + ", " + str(w) + ")")

        if params.pool_type=="only_end":
            x = MaxPooling2D(params.m_pool[0], border_mode='same')(x)
            # if MaxPooling border=='valid' h=int(h/params.params.m_pool[i][0])
            h = math.ceil(h / params.m_pool[-1][0])
            w = math.ceil(w / params.m_pool[-1][1])
            print("pool  -> (" + str(d) + ", " + str(h) + ", " + str(w) + ")")

        x = Flatten()(x)

        inputs = [d*h*w]
        inputs.extend(params.dense_shapes)

        for i in range(len(inputs)):
            x = Dense(inputs[i],
                      kernel_initializer=params.cnn_init,
                      activation=params.cnn_dense_activation,
                      kernel_regularizer=eval(params.d_w_reg),
                      bias_regularizer=eval(params.d_b_reg),
                      activity_regularizer=eval(params.d_a_reg),
                      kernel_constraint=eval(params.d_w_constr),
                      bias_constraint=eval(params.d_b_constr),
                      bias=params.bias)(x)
            print("dense[" + str(i) + "] -> (" + str(inputs[i]) + ")")
            if (params.dropout):
                x = Dropout(params.drop_rate)(x)

        # ---------------------------------------------------------- Decoding

        for i in range(len(inputs) - 2, -1, -1):  # backwards indices last excluded
            x = Dense(inputs[i],
                      kernel_initializer=params.cnn_init,
                      activation=params.cnn_dense_activation,
                      kernel_regularizer=eval(params.d_w_reg),
                      bias_regularizer=eval(params.d_b_reg),
                      activity_regularizer=eval(params.d_a_reg),
                      kernel_constraint=eval(params.d_w_constr),
                      bias_constraint=eval(params.d_b_constr),
                      bias=params.bias)(x)
            print("dense[" + str(i) + "] -> (" + str(inputs[i]) + ")")
            if (params.dropout):
                x = Dropout(params.drop_rate)(x)#ATTENZIONE: nostra versione keras1.2. nella documentazione ufficiale dropout è cambiato ma a noi serve il vecchio ovverro quello con il parametro "p"

        x = Reshape((d, h, w))(x)
        print("----------------------------------->(" + str(d) + ", " + str(h) + ", " + str(w) + ")")

        for i in range(len(params.kernel_number) - 1, -1, -1):

            x = Convolution2D(params.kernel_number[i],
                              params.kernel_shape[i][0],
                              params.kernel_shape[i][1],
                              kernel_initializer=params.cnn_init,
                              activation=params.cnn_conv_activation,
                              border_mode=params.border_mode,
                              subsample=tuple(params.strides[i]),
                              kernel_regularizer=eval(params.cnn_w_reg),
                              bias_regularizer=eval(params.cnn_b_reg),
                              activity_regularizer=eval(params.cnn_a_reg),
                              kernel_constraint=eval(params.cnn_w_constr),
                              bias_constraint=eval(params.cnn_b_constr),
                              bias=params.bias)(x)

            if params.border_mode == 'same':
                ph = params.kernel_shape[i][0] - 1
                pw = params.kernel_shape[i][1] - 1
            else:
                ph = pw = 0
            h = int((h - params.kernel_shape[i][0] + ph) / params.strides[i][0]) + 1
            w = int((w - params.kernel_shape[i][1] + pw) / params.strides[i][1]) + 1
            d = params.kernel_number[i]
            print("conv " + str(i) + "->(" + str(d) + ", " + str(h) + ", " + str(w) + ")")

            if params.pool_type == "only_end" and i == len(params.kernel_number) - 1:
                x = UpSampling2D(params.m_pool[i])(x)
                h = h * params.m_pool[i][0]
                w = w * params.m_pool[i][1]
                print("up->   (" + str(d) + ", " + str(h) + ", " + str(w) + ")")
            elif params.pool_type=="all":
                x = UpSampling2D(params.m_pool[i])(x)
                h = h * params.m_pool[i][0]
                w = w * params.m_pool[i][1]
                print("up " + str(i) + "->  (" + str(d) + ", " + str(h) + ", " + str(w) + ")")

        dh = h - params.cnn_input_shape[1]
        dw = w - params.cnn_input_shape[2]
        # print(h, params.cnn_input_shape[1], w, params.cnn_input_shape[2])
        print(str(h) + " " + str(params.cnn_input_shape[1]) + " " + str(w) + " " + str(params.cnn_input_shape[2]))

        h_zp = h_cr = w_zp = w_cr = (0, 0)
        if dh > 0:
            h_cr = (int(dh / 2), dh - int(dh / 2))
        else:
            h_zp = (-int(dh / 2), int(dh / 2) - dh)
        if dw > 0:
            w_cr = (int(dw / 2), dw - int(dw / 2))
        else:
            w_zp = (-int(dw / 2), int(dw / 2) - dw)

        # print(h_zp, w_zp, type(h_zp), type(w_zp), )
        print(str(h_zp) + " " + str(w_zp) + " " + str(type(h_zp)) + " " + str(type(w_zp)))
        # print(h_cr, w_cr, type(h_cr), type(w_cr), )
        print(str(h_cr) + " " + str(w_cr) + " " + str(type(h_cr)) + " " + str(type(w_cr)))

        x = ZeroPadding2D(padding=(h_zp[0], h_zp[1], w_zp[0], w_zp[1]))(x)
        x = Cropping2D(cropping=(h_cr, w_cr))(x)

        decoded = Convolution2D(params.cnn_input_shape[0],
                                params.kernel_shape[0][0],
                                params.kernel_shape[0][1],
                                kernel_initializer=params.cnn_init,
                                activation=params.cnn_conv_activation,
                                border_mode=params.border_mode,
                                subsample=(1,1),
                                kernel_regularizer=eval(params.cnn_w_reg),
                                bias_regularizer=eval(params.cnn_b_reg),
                                activity_regularizer=eval(params.cnn_a_reg),
                                kernel_constraint=eval(params.cnn_w_constr),
                                bias_constraint=eval(params.cnn_b_constr),
                                bias=params.bias)(x)

        self._autoencoder = Model(input_img, decoded)
        self._autoencoder.summary()
        self._autoencoder.name = 'AlreadyDefModel'

        return self._autoencoder

    def define_sequential_arch(self, params=None, path=None):
        if params is not None:
            if params.RNN_type is not None:
                #bottleneck indicates if the central layer of autoencoder is not a dense layer, in order to obtain a symmentric autoencoder:
                # dense-Rnn-dense. In this case the last number in dense_shape isn't the encoding layer.
                bottlneck=1 #TODO FIND ETTER WAY TO DO THIS
                input_img = Input(shape=(params.frame_context, params.dense_input_shape))
            else:
                bottlneck=0 #TODO FIND ETTER WAY TO DO THIS
                input_img = Input(shape=(params.dense_input_shape,))

            x = input_img
            for i in range(len(params.dense_shapes)):
                x = Dense(params.dense_shapes[i],
                          kernel_initializer=params.init,
                          activation=params.dense_activation,
                          kernel_regularizer=eval(params.d_w_reg),
                          bias_regularizer=eval(params.d_b_reg),
                          activity_regularizer=eval(params.d_a_reg),
                          kernel_constraint=eval(params.d_w_constr),
                          bias_constraint=eval(params.d_b_constr),
                          use_bias=params.bias)(x)
                print("dense[" + str(i) + "] -> (" + str(params.dense_shapes[i]) + ")")
                if (params.dropout):
                    x = Dropout(params.drop_rate)(x)
                if (params.batch_norm):
                    x = BatchNormalization()(x)

            if params.RNN_type is not None:
            # ---------------------------------------------------------- RNN Bottleneck

                if params.RNN_type == 'LSTM':
                    x = LSTM(params.RNN_layer_shape,
                             kernel_initializer=params.init,
                             activation=params.dense_activation,
                             return_sequences=False)(x)

                elif params.RNN_type == 'SimpleRNN':
                    x = SimpleRNN(params.RNN_layer_shape,
                                  kernel_initializer=params.init,
                                  activation=params.dense_activation,
                                  return_sequences=False)(x)

                elif params.RNN_type == 'GRU':
                    x = GRU(params.RNN_layer_shape,
                            kernel_initializer=params.init,
                            activation=params.dense_activation,
                            return_sequences=False)(x)

                if (params.dropout):
                    x = Dropout(params.drop_rate)(x)
                if (params.batch_norm):
                    x = BatchNormalization()(x)

                print("RNN_layer[" + params.RNN_type + "] -> (" + str(params.RNN_layer_shape) + ")")
            # ---------------------------------------------------------- Decoding

            for i in range(len(params.dense_shapes) - (2-bottlneck), -1, -1):  # backwards indices last excluded

                if i is not 0:
                    x = Dense(params.dense_shapes[i],
                              kernel_initializer=params.init,
                              activation=params.dense_activation,
                              kernel_regularizer=eval(params.d_w_reg),
                              bias_regularizer=eval(params.d_b_reg),
                              activity_regularizer=eval(params.d_a_reg),
                              kernel_constraint=eval(params.d_w_constr),
                              bias_constraint=eval(params.d_b_constr),
                              use_bias=params.bias)(x)
                    if (params.batch_norm):
                        x = BatchNormalization()(x)
                # last dense with linear activation
                elif params.hybrid_phase:
                    mod = Dense((int(params.dense_input_shape / 3)),
                                kernel_initializer=params.init,
                                activation='relu', # because the module is always positive
                                kernel_regularizer=eval(params.d_w_reg),
                                bias_regularizer=eval(params.d_b_reg),
                                activity_regularizer=eval(params.d_a_reg),
                                kernel_constraint=eval(params.d_w_constr),
                                bias_constraint=eval(params.d_b_constr),
                                use_bias=params.bias)(x)
                    cos = Dense((int(params.dense_input_shape / 3)),
                                kernel_initializer=params.init,
                                activation='tanh',
                                kernel_regularizer=eval(params.d_w_reg),
                                bias_regularizer=eval(params.d_b_reg),
                                activity_regularizer=eval(params.d_a_reg),
                                kernel_constraint=eval(params.d_w_constr),
                                bias_constraint=eval(params.d_b_constr),
                                use_bias=params.bias)(x)
                    sin = Dense((int(params.dense_input_shape / 3)),
                                kernel_initializer=params.init,
                                activation='tanh',
                                kernel_regularizer=eval(params.d_w_reg),
                                bias_regularizer=eval(params.d_b_reg),
                                activity_regularizer=eval(params.d_a_reg),
                                kernel_constraint=eval(params.d_w_constr),
                                bias_constraint=eval(params.d_b_constr),
                                use_bias=params.bias)(x)
                    x = Concatenate()([mod, cos, sin])
                else:
                    x = Dense(params.dense_input_shape,
                              kernel_initializer=params.init,
                              activation='linear',
                              kernel_regularizer=eval(params.d_w_reg),
                              bias_regularizer=eval(params.d_b_reg),
                              activity_regularizer=eval(params.d_a_reg),
                              kernel_constraint=eval(params.d_w_constr),
                              bias_constraint=eval(params.d_b_constr),
                              use_bias=params.bias)(x)

                print("dense[" + str(i) + "] -> (" + str(params.dense_shapes[i]) + ")")
                if (params.dropout):
                    x = Dropout(params.drop_rate)(x)

            #ATTENZIONE: nostra versione keras1.2. nella documentazione ufficiale dropout è cambiato ma a noi serve il vecchio ovverro quello con il parametro "p"


            decoded = x
            self._autoencoder = Model(input_img, decoded)

        elif path is not None:
            self._autoencoder = load_model(path)

        self._autoencoder.summary()
        # self._autoencoder.name = 'Autoencoder'
        return self._autoencoder


    def model_compile(self, model=None, optimizer='adadelta', learning_rate=1.0, loss='mse'):
        """
        compila il modello con i parametri passati: se non viene passato compila il modello istanziato dalla classe

        :param model:
        :param optimizer:
        :param learning_rate:
        :param loss:
        :return:
        """

        print("model_compile")

        if optimizer == "adadelta":
            opti = Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-06)
        if optimizer == "adam":
            opti = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        if optimizer == 'sgd':
            opti = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        if model is None:
            self._autoencoder.compile(optimizer=opti, loss=loss)
        else:
            model.compile(optimizer=opti, loss=loss)

        return self._autoencoder

    def model_fit(self, x_train, y_train, x_dev=None, y_dev=None, validation_split=0.0, epochs=50, batch_size=128, shuffle=True, model=None,
                  fit_net=True, patience=20, nameFileLogCsv='losses.csv', callback_list=[]):
        print("model_fit")

        if model is not None:
            self._autoencoder = model

        if not fit_net:  # take a model already fitted
            self.load_model('my_model.h5')

        else:

            csv_logger = CSVLogger(nameFileLogCsv)
            early_stopping = EarlyStopping(patience=patience)
            tensor_board = TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=batch_size, write_graph=True,
                                        write_grads=False, write_images=True)
            callback_list.append(csv_logger)
            callback_list.append(early_stopping)
            if (x_dev is not None and y_dev is not None) or validation_split > 0.0:  # se ho a disposizione un validation set allora faccio anche l'early stopping

                self._autoencoder.fit(x_train, y_train,
                                      epochs=epochs,
                                      validation_split=validation_split,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      # callbacks=[csv_logger, early_stopping, tensor_board],
                                      # callbacks=[csv_logger, early_stopping],
                                      callbacks=callback_list,
                                      verbose=2)  # with a value != 1 ProbarLogging is not called

            else:
                self._autoencoder.fit(x_train, y_train,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      callbacks=[csv_logger],
                                      verbose=1)
                # save the model an weights on disk
                # self.save_model(self._autoencoder)

        # self._autoencoder.save('my_model.h5')
        #            self._autoencoder.save_weights('my_model_weights.h5')
        # save the model and wetight on varibles
        #        self._config = self._autoencoder.get_config()
        #        self._weight = self._autoencoder.get_weights()

        # self._fit_net = False
        return self._autoencoder

    def load_model(self, model, weights=None):
        #load from disk the model
        autoencoder = load_model(model)
        if weights is not None:
            autoencoder.load_weights(weights) #é inutile: load_model carica anche i pesi
        self._autoencoder = autoencoder

        return autoencoder

    def save_model(self, model=None, path='.', name='my_model'):
        """
        salva il modello e i pesi.
        Se non è passato nessun modello, viene salvato il modello che è istanziato attualmente nella classe
        """
        if model is None:
            model = self._autoencoder

        model.save(os.path.join(path, name + '.h5'))
        #model.save_weights(os.path.join(path, name + '_weights.h5')) è inutile: save salva anche i pesi
        json_string = model.to_json()
        with open(os.path.join(path, name + '.json'), "w") as text_file:
            text_file.write(json.dumps(json_string, indent=4, sort_keys=True))

        return

    def reconstruct_spectrogram(self, x_test, model=None):
        """
        Decode the input data

        :param x_test: The data to be decoded
        :param model: autoencoder model to use for decode the input data. If None a self model of class is used
        :return: the decoded data
        """
        print("reconstruct_spectrogram")
        if model is None:
            decoded_imgs = self._autoencoder.predict(x_test)
        else:
            decoded_imgs = model.predict(x_test)

        return decoded_imgs


    # def reconstruct_handwritedigit_mnist(self, x_test):  # @Diego -> da cancellare?
    #     """
    #     vuole in ingresso un vettore con shape (1,1,28,28), la configurazione del modello e i pesi
    #     """
    #     print("reconstruct_handwritedigit_mnist")
    #
    #     decoded_imgs = self._autoencoder.predict(x_test)
    #
    #     plt.figure()
    #     # display original
    #     ax = plt.subplot(2, 1, 1)
    #     plt.imshow(x_test.reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #
    #     # display reconstruction
    #     ax = plt.subplot(2, 1, 2)
    #     plt.imshow(decoded_imgs.reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #     plt.show()



