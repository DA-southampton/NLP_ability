import os

import torch
from torchtext.data import Field

from onmt.inputters.datareader_base import DataReaderBase

try:
    import numpy as np
except ImportError:
    np = None


class VecDataReader(DataReaderBase):
    """Read feature vector data from disk.
    Raises:
        onmt.inputters.datareader_base.MissingDependencyException: If
            importing ``np`` fails.
    """

    def __init__(self):
        self._check_deps()

    @classmethod
    def _check_deps(cls):
        if np is None:
            cls._raise_missing_dep("np")

    def read(self, vecs, side, vec_dir=None):
        """Read data into dicts.
        Args:
            vecs (str or Iterable[str]): Sequence of feature vector paths or
                path to file containing feature vector paths.
                In either case, the filenames may be relative to ``vec_dir``
                (default behavior) or absolute.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            vec_dir (str): Location of source vectors. See ``vecs``.
        Yields:
            A dictionary containing feature vector data.
        """

        if isinstance(vecs, str):
            vecs = DataReaderBase._read_file(vecs)

        for i, filename in enumerate(vecs):
            filename = filename.decode("utf-8").strip()
            vec_path = os.path.join(vec_dir, filename)
            if not os.path.exists(vec_path):
                vec_path = filename

            assert os.path.exists(vec_path), \
                'vec path %s not found' % filename

            vec = np.load(vec_path)
            yield {side: torch.from_numpy(vec),
                   side + "_path": filename, "indices": i}


def vec_sort_key(ex):
    """Sort using the length of the vector sequence."""
    return ex.src.shape[0]


class VecSeqField(Field):
    """Defines an vector datatype and instructions for converting to Tensor.
    See :class:`Fields` for attribute descriptions.
    """

    def __init__(self, preprocessing=None, postprocessing=None,
                 include_lengths=False, batch_first=False, pad_index=0,
                 is_target=False):
        super(VecSeqField, self).__init__(
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
                each having shape ``(len, n_feats, feat_dim)``
                where len is variable.
        Returns:
            torch.FloatTensor or Tuple[torch.FloatTensor, List[int]]: The
                padded tensor of shape
                ``(batch_size, max_len, n_feats, feat_dim)``.
                and a list of the lengths if `self.include_lengths` is `True`
                else just returns the padded tensor.
        """

        assert not self.pad_first and not self.truncate_first \
            and not self.fix_length and self.sequential
        minibatch = list(minibatch)
        lengths = [x.size(0) for x in minibatch]
        max_len = max(lengths)
        nfeats = minibatch[0].size(1)
        feat_dim = minibatch[0].size(2)
        feats = torch.full((len(minibatch), max_len, nfeats, feat_dim),
                           self.pad_token)
        for i, (feat, len_) in enumerate(zip(minibatch, lengths)):
            feats[i, 0:len_, :, :] = feat
        if self.include_lengths:
            return (feats, lengths)
        return feats

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.
        If the field has ``include_lengths=True``, a tensor of lengths will be
        included in the return value.
        Args:
            arr (torch.FloatTensor or Tuple(torch.FloatTensor, List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
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
        arr = arr.to(device)

        if self.postprocessing is not None:
            arr = self.postprocessing(arr, None)

        if self.sequential and not self.batch_first:
            arr = arr.permute(1, 0, 2, 3)
        if self.sequential:
            arr = arr.contiguous()

        if self.include_lengths:
            return arr, lengths
        return arr


def vec_fields(**kwargs):
    vec = VecSeqField(pad_index=0, include_lengths=True)
    return vec
