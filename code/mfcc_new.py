from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy

audio = ['mandarin29', 'arabic16', 'english23'];
AUDIO_LOC = '../data/audio/{}.wav'
(rate,sig) = wav.read(AUDIO_LOC.format(audio[1]))
mfcc_feat = mfcc(sig,rate)
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(sig,rate)

mfcc = mfcc_feat

print mfcc.shape
mfcc= numpy.swapaxes(mfcc, 0, 1)
fig, ax = plt.subplots()
im = ax.imshow(mfcc, cmap='hot', interpolation='nearest', origin='lower', aspect='auto')
plt.colorbar(im)
plt.savefig('../images/arabic_t.png')