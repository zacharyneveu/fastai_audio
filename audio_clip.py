from fastai.torch_core import *
from IPython.display import display, Audio
import soundfile as sf
# debug
import sys

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
        return len(self.data)

    @property
    def duration(self):
        return self.num_samples / self.sample_rate

    def show(self, ax=None, figsize=(5, 1), player=True, title=None, **kwargs):
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


def open_audio(fn):
    # at first, use wavfile.read because it is significantly faster
    x, sr = sf.read(fn, dtype='float32', always_2d=True)
    x = x[:,0]

    t = torch.from_numpy(x.astype(np.float32, copy=False))
    if x.dtype == np.int16:
        t.div_(32767)
    # not sure WHY but these show up sometimes
    elif x.dtype == np.uint8:
        t.sub_(127).div_(128)
        n8 += 1
    elif x.dtype != np.float32:
        raise OSError('Encountered unexpected dtype: {}'.format(x.dtype))
    return AudioClip(t, sr)
