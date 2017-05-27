MODEL_WEIGHT_BASEPATH = './weights'
MODEL_HISTORY_BASEPATH = './history'
MODEL_MEANS_BASEPATH = './means'
N_CLASSES = 8
TRAIN_SPLIT = 0.85
VALIDATION_SPLIT = 0.15
N_TRAINING_SET = 4
MAX_EPOCH_NUM = 400
EARLY_STOPPING_EPOCH = 20
SGD_LR_REDUCE = 5
BATCH_SIZE = 16
N_MEL_BANDS = 128
SEGMENT_DUR = 16
FS = 16000
ALLOWED_MODELS = ['han16', 'singlelayer', 'multilayer']
#NSYNTH_TESTING_META_PATH = '/media/fabio/DATA/Work/crossSynth/autoXsynth/dataset/Autoencoder-Dataset/list.txt'
#NSYNT_TESTING_FEATURES = '/media/fabio/DATA/Work/crossSynth/autoXsynth/dataset/Autoencoder-Dataset/stft-2048'
NSYNTH_TESTING_META_PATH = '/media/fabio/DATA/Work/crossSynth/autoXsynth/dataset/reconstructedWav_for_paper/listNOLAST2.txt'
NSYNT_TESTING_FEATURES = '/media/fabio/DATA/Work/crossSynth/autoXsynth/dataset/reconstructedWav_for_paper/trimmed'
SCORES_PATH = './Scores_NOLAST2'
WEIGHTS_PATH = '/media/fabio/DATA/Work/crossSynth/autoXsynth/weights/multilayer/epoch.52-val_loss.0.346-fbeta.0.903-multi_layer_musically_motivated_model.hdf5'

