MODEL_WEIGHT_BASEPATH = './weights'
MODEL_HISTORY_BASEPATH = './history'
MODEL_MEANS_BASEPATH = './means'
N_CLASSES = 10
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
NSYNTH_TESTING_META_PATH = '/media/fabio/DATA/Work/crossSynth/autoXsynth/dataset/reconstructedWav_for_paper/list709.txt'
NSYNT_TESTING_FEATURES = '/media/fabio/DATA/Work/crossSynth/autoXsynth/dataset/reconstructedWav_for_paper/'
SCORES_PATH = './Scores_VOCAL'

