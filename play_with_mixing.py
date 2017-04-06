
import numpy as np
import os
import librosa
import utility as u
sr = 22050
hops = 2048
nfft = 4096
aS = 0.05
aP = 1-aS
aM = 0.3
bS = 0.5
bP = 1-bS
frame_context = 2

root_path = os.getcwd()
destFold = os.path.join(root_path,'mix_analysis')
u.makedir(destFold)
id='408'
out_filename = os.path.join(destFold,"reconstructed_"+id+"_P.wav")

predictName = "prediction_"+id+".npy"
predicPathfile = os.path.join('result', 'preds', predictName)
sourceName = 'Vox.npy'
sourcePathfile = os.path.join('dataset','source', 'stft-2048', sourceName)

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

Mx = aS * source_sig_module + aP * predict_sig_module + aM * np.sqrt(source_sig_module * predict_sig_module)
Phix = bS * source_sig_phase + bP * predict_sig_phase

prediction_complex = Mx * Phix

S = librosa.core.istft(prediction_complex, hop_length=hops, win_length=nfft)
librosa.output.write_wav(os.path.join(out_filename,out_filename), S, sr)