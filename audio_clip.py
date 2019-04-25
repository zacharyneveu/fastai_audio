from fastai.torch_core import *
from IPython.display import display, Audio
import torchaudio as ta
import torchaudio.transforms as tatfms
import librosa
import librosa.display


__all__ = ['AudioClip', 'open_audio']


class AudioClip(ItemBase):
    def __init__(self, signal, sample_rate):
        self.data = signal
        self.sample_rate = sample_rate

    def __str__(self):
        return '(duration={}s, sample_rate={:.1f}KHz)'.format(
            self.duration, self.sample_rate/1000)

    def clone(self):
        return self.__class__(self.data.clone(), self.sample_rate)

    def apply_tfms(self, tfms, **kwargs):
        x = self.clone()
        for tfm in tfms:
            x.data = tfm(x.data)
        return x

    @property
    def num_samples(self):
        #return len(self.data)
        return self.data.size(0)

    @property
    def duration(self):
        return self.num_samples / self.sample_rate

    def show(self, ax=None, figsize=(5, 1), player=True, spec=True, title=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        if title:
            ax.set_title(title)
        timesteps = np.arange(len(self.data)) / self.sample_rate
        ax.plot(timesteps, self.data)
        ax.set_xlabel('Time (s)')
        plt.show()
        if player:
            # unable to display an IPython 'Audio' player in plt axes
            display(Audio(self.data, rate=self.sample_rate))
        if spec:
            spectfm = tatfms.Compose([
                tatfms.MelSpectrogram(sr=self.sample_rate, n_mels=128, n_fft=2048, hop=512),
                tatfms.SpectrogramToDB()
                ])
            x = self.data.unsqueeze(0)
            spec = spectfm(x).squeeze(0).transpose(0,1)
            nparr = spec.cpu().numpy()
            librosa.display.specshow(nparr)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            plt.tight_layout()



def open_audio(fn):
    x, sr = ta.load(fn)
    # Get only first channel for now
    x = x[0,:]

    return AudioClip(x, sr)
