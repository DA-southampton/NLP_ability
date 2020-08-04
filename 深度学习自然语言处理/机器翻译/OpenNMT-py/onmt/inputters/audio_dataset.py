# -*- coding: utf-8 -*-
import os
from tqdm import tqdm

import torch
from torchtext.data import Field

from onmt.inputters.datareader_base import DataReaderBase

# imports of datatype-specific dependencies
try:
    import torchaudio
    import librosa
    import numpy as np
except ImportError:
    torchaudio, librosa, np = None, None, None


class AudioDataReader(DataReaderBase):
    """Read audio data from disk.

    Args:
        sample_rate (int): sample_rate.
        window_size (float) : window size for spectrogram in seconds.
        window_stride (float): window stride for spectrogram in seconds.
        window (str): window type for spectrogram generation. See
            :func:`librosa.stft()` ``window`` for more details.
        normalize_audio (bool): subtract spectrogram by mean and divide
            by std or not.
        truncate (int or NoneType): maximum audio length
            (0 or None for unlimited).

    Raises:
        onmt.inputters.datareader_base.MissingDependencyException: If
            importing any of ``torchaudio``, ``librosa``, or ``numpy`` fail.
    """

    def __init__(self, sample_rate=0, window_size=0, window_stride=0,
                 window=None, normalize_audio=True, truncate=None):
        self._check_deps()
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.normalize_audio = normalize_audio
        self.truncate = truncate

    @classmethod
    def from_opt(cls, opt):
        return cls(sample_rate=opt.sample_rate, window_size=opt.window_size,
                   window_stride=opt.window_stride, window=opt.window)

    @classmethod
    def _check_deps(cls):
        if any([torchaudio is None, librosa is None, np is None]):
            cls._raise_missing_dep(
                "torchaudio", "librosa", "numpy")

    def extract_features(self, audio_path):
        # torchaudio loading options recently changed. It's probably
        # straightforward to rewrite the audio handling to make use of
        # up-to-date torchaudio, but in the meantime there is a legacy
        # method which uses the old defaults
        sound, sample_rate_ = torchaudio.legacy.load(audio_path)
        if self.truncate and self.truncate > 0:
            if sound.size(0) > self.truncate:
                sound = sound[:self.truncate]

        assert sample_rate_ == self.sample_rate, \
            'Sample rate of %s != -sample_rate (%d vs %d)' \
            % (audio_path, sample_rate_, self.sample_rate)

        sound = sound.numpy()
        if len(sound.shape) > 1:
            if sound.shape[1] == 1:
                sound = sound.squeeze()
            else:
                sound = sound.mean(axis=1)  # average multiple channels

        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        d = librosa.stft(sound, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, _ = librosa.magphase(d)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize_audio:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
        return spect

    def read(self, data, side, src_dir=None):
        """Read data into dicts.

        Args:
            data (str or Iterable[str]): Sequence of audio paths or
                path to file containing audio paths.
                In either case, the filenames may be relative to ``src_dir``
                (default behavior) or absolute.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            src_dir (str): Location of source audio files. See ``data``.

        Yields:
            A dictionary containing audio data for each line.
        """

        assert src_dir is not None and os.path.exists(src_dir),\
            "src_dir must be a valid directory if data_type is audio"

        if isinstance(data, str):
            data = DataReaderBase._read_file(data)

        for i, line in enumerate(tqdm(data)):
            line = line.decode("utf-8").strip()
            audio_path = os.path.join(src_dir, line)
            if not os.path.exists(audio_path):
                audio_path = line

            assert os.path.exists(audio_path), \
                'audio path %s not found' % line

            spect = self.extract_features(audio_path)
            yield {side: spect, side + '_path': line, 'indices': i}


def audio_sort_key(ex):
    """Sort using duration time of the sound spectrogram."""
    return ex.src.size(1)


class AudioSeqField(Field):
    """Defines an audio datatype and instructions for converting to Tensor.

    See :class:`Fields` for attribute descriptions.
    """

    def __init__(self, preprocessing=None, postprocessing=None,
                 include_lengths=False, batch_first=False, pad_index=0,
                 is_target=False):
        super(AudioSeqField, self).__init__(
            sequential=True, use_vocab=False, init_token=None,
            eos_token=None, fix_length=False, dtype=torch.float,
            preprocessing=preprocessing, postprocessing=postprocessing,
            lower=False, tokenize=None, include_lengths=include_lengths,
            batch_first=batch_first, pad_token=pad_index, unk_token=None,
            pad_first=False, truncate_first=False, stop_words=None,
            is_target=is_target
        )

    def pad(self, minibatch):
        """Pad a batch of examples to the length of the longest example.

        Args:
            minibatch (List[torch.FloatTensor]): A list of audio data,
                each having shape 1 x n_feats x len where len is variable.

        Returns:
            torch.FloatTensor or Tuple[torch.FloatTensor, List[int]]: The
                padded tensor of shape ``(batch_size, 1, n_feats, max_len)``.
                and a list of the lengths if `self.include_lengths` is `True`
                else just returns the padded tensor.
        """

        assert not self.pad_first and not self.truncate_first \
            and not self.fix_length and self.sequential
        minibatch = list(minibatch)
        lengths = [x.size(1) for x in minibatch]
        max_len = max(lengths)
        nfft = minibatch[0].size(0)
        sounds = torch.full((len(minibatch), 1, nfft, max_len), self.pad_token)
        for i, (spect, len_) in enumerate(zip(minibatch, lengths)):
            sounds[i, :, :, 0:len_] = spect
        if self.include_lengths:
            return (sounds, lengths)
        return sounds

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.

        If the field has ``include_lengths=True``, a tensor of lengths will be
        included in the return value.

        Args:
            arr (torch.FloatTensor or Tuple(torch.FloatTensor, List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True. Examples have shape
                ``(batch_size, 1, n_feats, max_len)`` if `self.batch_first`
                else ``(max_len, batch_size, 1, n_feats)``.
            device (str or torch.device): See `Field.numericalize`.
        """

        assert self.use_vocab is False
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=torch.int, device=device)

        if self.postprocessing is not None:
            arr = self.postprocessing(arr, None)

        if self.sequential and not self.batch_first:
            arr = arr.permute(3, 0, 1, 2)
        if self.sequential:
            arr = arr.contiguous()
        arr = arr.to(device)
        if self.include_lengths:
            return arr, lengths
        return arr


def audio_fields(**kwargs):
    audio = AudioSeqField(pad_index=0, batch_first=True, include_lengths=True)
    return audio
