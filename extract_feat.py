import AudioFeatures

# path = 'dataset/target/sweet_child_o_mine/wav/'
# path = 'dataset/source/vox/wav/'
path = 'dataset/okapi_sample/'
# path = '/media/buckler/DataSSD/Phd/autoXsynthImproved/data/target/sweet_child_o_mine/wav/'
extractor = AudioFeatures.AudioFeatures(feature='stft', n_fft=2048, win_len=2048, hop=1024, path=path, extension='.wav',
                                        s_rate=None, free_disk=False, norm=False)

extractor.feat_extract()

print()