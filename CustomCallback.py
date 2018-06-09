
from tensorflow.python.keras.callbacks import Callback
import librosa
import numpy as np
import os
import dataset_manupulation as dm

class GenerateWavCallback(Callback):

    def __init__(self, args, source_sig, pred_folder, wav_dest_path):
        self.args = args
        self.source_sig = source_sig
        self.pred_folder = pred_folder
        self.wav_dest_path = wav_dest_path


    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            if self.args.hybrid_phase:
                # TODO DO it separately for module, sin, cos
                self.source_sig_module = np.absolute(self.source_sig)
                self.module_len = self.source_sig_module.shape[1]
                self.source_sig_phase = np.angle(self.source_sig)
                cos_source_sig = np.cos(self.source_sig_phase)
                sin_source_sig = np.sin(self.source_sig_phase)

                self.source_sig_input = np.concatenate([self.source_sig_module, cos_source_sig, sin_source_sig], axis=1)
                # self.source_sig_input = np.hstack([self.source_sig_module, self.source_sig_phase])

                if self.args.RNN_type is not None:
                    self.source_sig_input, _ = dm.create_context(self.source_sig_input, look_back=self.args.frame_context)
                    self.source_sig_module = self.source_sig_module[: - self.args.frame_context - 1, :]
                    self.source_sig_phase = self.source_sig_phase[: - self.args.frame_context - 1, :]

                prediction = np.asarray(self.model.predict(self.source_sig_input), order="C")
                pred_name = "prediction_" + str(epoch)
                np.save(os.path.join(self.pred_folder, pred_name), prediction)

                prediction_module = prediction[:, 0:self.module_len]
                prediction_cos = prediction[:, self.module_len:(self.module_len * 2)]
                prediction_sin = prediction[:, (self.module_len * 2):(self.module_len * 3)]
                prediction_phase = prediction_cos + 1j * prediction_sin
                # prediction_phase = prediction[:, module_len:]

                Mx = self.args.aS * self.source_sig_module + self.args.aP * prediction_module + self.args.aM * np.sqrt(
                    self.source_sig_module * prediction_module)
                Phix = self.args.bS * self.source_sig_phase + self.args.bP * prediction_phase

                prediction_complex = Mx * Phix
            else:
                self.source_sig.dtype = 'float32'
                self.source_sig_input = self.source_sig
                if self.args.RNN_type is not None:
                    self.source_sig_input, _ = dm.create_context(self.source_sig, look_back=self.args.frame_context)
                prediction = np.asarray(self.model.reconstruct_spectrogram(self.source_sig_input), order="C")
                prediction_complex = prediction.view()
                prediction_complex.dtype = "complex64"

            # prediction_complex = librosa.util.fix_length(prediction_complex, len(prediction_complex) + win_len)
            S = librosa.core.istft(prediction_complex.T, hop_length=int(self.args.hopsize), win_length=int(self.args.win_len))
            out_filename = "reconstruction_" + str(epoch) + ".wav"
            print("saving wav at {}".format(os.path.join(self.wav_dest_path, out_filename)))
            librosa.output.write_wav(os.path.join(self.wav_dest_path, out_filename), S, self.args.sample_rate)
