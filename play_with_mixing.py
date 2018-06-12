
import numpy as np
import os
import librosa
import utility as u
sr = 44100
hops = 1024
nfft = 2048
win_len = 2048
aS = 0
aP = 1-aS
aM = 0.0
bS = 1
bP = 1-bS
frame_context = 2

root_path = os.getcwd()
expid = '5'
destFold = os.path.join(root_path, 'experiments', expid, 'mix_analysis')
u.makedir(destFold)
epoch='20'
out_filename = os.path.join(destFold,"reconstructed_"+epoch+"_ap"+str(aP)+".wav")

predictName = "prediction_"+epoch+".npy"
predicPathfile = os.path.join('experiments', expid, 'preds', predictName)
sourceName = 'Vox.npy'
sourcePathfile = os.path.join('dataset', 'source', 'vox',  'stft-2048', sourceName)

source = np.load(sourcePathfile)
predict = np.load(predicPathfile)


source_sig = source.T.view().T
source_sig_module = np.absolute(source_sig)
source_sig_phase = np.angle(source_sig)
source_sig_module = source_sig_module[:, : - frame_context - 1]
source_sig_phase = source_sig_phase[:, : - frame_context - 1]
# cos_source_sig = np.cos(source_sig_phase)
# sin_source_sig = np.sin(source_sig_phase)

module_len = len(source_sig_module)
predict_sig_module_ = predict[:, 0:module_len]
cos_predict_sig = predict[:, module_len:(module_len * 2)]
sin_predict_sig = predict[:, (module_len * 2):(module_len * 3)]
predict_sig_phase_ = cos_predict_sig + 1j * sin_predict_sig

predict_sig_module = predict_sig_module_.T.view()#.T
predict_sig_phase = predict_sig_phase_.T.view()#.T
L = source_sig_module.shape[1]

Mx = aS * source_sig_module + aP * predict_sig_module[:,:L] + aM * np.sqrt(source_sig_module * predict_sig_module[:,:L])
Phix = bS * source_sig_phase + bP * predict_sig_phase[:,:L]

prediction_complex = Mx * Phix

S = librosa.core.istft(prediction_complex, hop_length=hops, win_length=win_len)
librosa.output.write_wav(os.path.join(out_filename,out_filename), S, sr)