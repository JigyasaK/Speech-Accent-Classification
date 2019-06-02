import matplotlib.pyplot as plt
import librosa
import librosa.display as display
from scipy.io import wavfile

import common
from scipy import signal
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct

audio = ['mandarin29', 'arabic16', 'english23'];
AUDIO_LOC = '../data/audio/{}.wav'
pre_emphasis = 0.97
frame_stride = 0.01
frame_size = 0.025


sample_rate, samples = wavfile.read(AUDIO_LOC.format(audio[0]))
y = common.get_wav(audio[1])
duration = librosa.core.get_duration(y)
mfcc = common.to_mfcc(y)


def imshow(mfcc):
    extent = [0, duration, 0, mfcc.shape[0]]
    # mfcc= numpy.swapaxes(mfcc, 0, 1)
    print mfcc.shape
    fig, ax = plt.subplots()
    im = ax.imshow(mfcc, cmap='coolwarm', interpolation='nearest', origin='lower', aspect='auto', extent=extent)
    plt.colorbar(im)
    plt.savefig('../images/arabic_t.png')

def libr(y):
    S = librosa.feature.melspectrogram(y, sr=24000, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_S, sr=24000, x_axis='time', y_axis='mel')
    plt.title('Mel power spectrogram ')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()

def lib_mfcc(mfcc):
    # mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

    # Let's pad on the first and second deltas while we're at it
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(delta2_mfcc)
    plt.ylabel('MFCC coeffs')
    plt.xlabel('Time')
    plt.title('MFCC')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('../images/arabic_mfcc.png')

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

freqs, times, spectrogram = log_specgram(samples, sample_rate)
print sample_rate
print samples.shape
fig = plt.figure(figsize=(14, 8))
ax2 = fig.add_subplot(212)
ax2.imshow(spectrogram.T, aspect='auto', origin='lower',
           extent=[times.min(), abs(times.max()), freqs.min(), freqs.max()])
# ax2.set_yticks(freqs[::16])
# ax2.set_xticks(times[::16])
ax2.set_title('Spectrogram of ' + audio[0])
ax2.set_ylabel('Freqs in Hz')
ax2.set_xlabel('Seconds')
plt.savefig('../images/arabic_t.png')

mean = np.mean(spectrogram, axis=0)
std = np.std(spectrogram, axis=0)
spectrogram = (spectrogram - mean) / std

print mean
print std

lib_mfcc(mfcc)
