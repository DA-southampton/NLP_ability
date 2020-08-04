# -*- coding: utf-8 -*-
import glob
import os
import codecs
import math

from collections import Counter, defaultdict
from itertools import chain, cycle

import torch
import torchtext.data
from torchtext.data import Field, RawField, LabelField
from torchtext.vocab import Vocab
from torchtext.data.utils import RandomShuffler

from onmt.inputters.text_dataset import text_fields, TextMultiField
from onmt.inputters.image_dataset import image_fields
from onmt.inputters.audio_dataset import audio_fields
from onmt.inputters.vec_dataset import vec_fields
from onmt.utils.logging import logger
# backwards compatibility
from onmt.inputters.text_dataset import _feature_tokenize  # noqa: F401
from onmt.inputters.image_dataset import (  # noqa: F401
    batch_img as make_img)

import gc


# monkey-patch to make torchtext Vocab's pickleable
def _getstate(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


Vocab.__getstate__ = _getstate
Vocab.__setstate__ = _setstate


def make_src(data, vocab):
    src_size = max([t.size(0) for t in data])
    src_vocab_size = max([t.max() for t in data]) + 1
    alignment = torch.zeros(src_size, len(data), src_vocab_size)
    for i, sent in enumerate(data):
        for j, t in enumerate(sent):
            alignment[j, i, t] = 1
    return alignment


def make_tgt(data, vocab):
    tgt_size = max([t.size(0) for t in data])
    alignment = torch.zeros(tgt_size, len(data)).long()
    for i, sent in enumerate(data):
        alignment[:sent.size(0), i] = sent
    return alignment


class AlignField(LabelField):
    """
    Parse ['<src>-<tgt>', ...] into ['<src>','<tgt>', ...]
    """
    def __init__(self, **kwargs):
        kwargs['use_vocab'] = False
        kwargs['preprocessing'] = parse_align_idx
        super(AlignField, self).__init__(**kwargs)

    def process(self, batch, device=None):
        """ Turn a batch of align-idx to a sparse align idx Tensor"""
        sparse_idx = []
        for i, example in enumerate(batch):
            for src, tgt in example:
                # +1 for tgt side to keep coherent after "bos" padding,
                # register ['N°_in_batch', 'tgt_id+1', 'src_id']
                sparse_idx.append([i, tgt + 1, src])

        align_idx = torch.tensor(sparse_idx, dtype=self.dtype, device=device)

        return align_idx


def parse_align_idx(align_pharaoh):
    """
    Parse Pharaoh alignment into [[<src>, <tgt>], ...]
    """
    align_list = align_pharaoh.strip().split(' ')
    flatten_align_idx = []
    for align in align_list:
        try:
            src_idx, tgt_idx = align.split('-')
        except ValueError:
            logger.warning("{} in `{}`".format(align, align_pharaoh))
            logger.warning("Bad alignement line exists. Please check file!")
            raise
        flatten_align_idx.append([int(src_idx), int(tgt_idx)])
    return flatten_align_idx


def get_fields(
    src_data_type,
    n_src_feats,
    n_tgt_feats,
    pad='<blank>',
    bos='<s>',
    eos='</s>',
    dynamic_dict=False,
    with_align=False,
    src_truncate=None,
    tgt_truncate=None
):
    """
    Args:
        src_data_type: type of the source input. Options are [text|img|audio].
        n_src_feats (int): the number of source features (not counting tokens)
            to create a :class:`torchtext.data.Field` for. (If
            ``src_data_type=="text"``, these fields are stored together
            as a ``TextMultiField``).
        n_tgt_feats (int): See above.
        pad (str): Special pad symbol. Used on src and tgt side.
        bos (str): Special beginning of sequence symbol. Only relevant
            for tgt.
        eos (str): Special end of sequence symbol. Only relevant
            for tgt.
        dynamic_dict (bool): Whether or not to include source map and
            alignment fields.
        with_align (bool): Whether or not to include word align.
        src_truncate: Cut off src sequences beyond this (passed to
            ``src_data_type``'s data reader - see there for more details).
        tgt_truncate: Cut off tgt sequences beyond this (passed to
            :class:`TextDataReader` - see there for more details).

    Returns:
        A dict mapping names to fields. These names need to match
        the dataset example attributes.
    """

    assert src_data_type in ['text', 'img', 'audio', 'vec'], \
        "Data type not implemented"
    assert not dynamic_dict or src_data_type == 'text', \
        'it is not possible to use dynamic_dict with non-text input'
    fields = {}

    fields_getters = {"text": text_fields,
                      "img": image_fields,
                      "audio": audio_fields,
                      "vec": vec_fields}

    src_field_kwargs = {"n_feats": n_src_feats,
                        "include_lengths": True,
                        "pad": pad, "bos": None, "eos": None,
                        "truncate": src_truncate,
                        "base_name": "src"}
    fields["src"] = fields_getters[src_data_type](**src_field_kwargs)

    tgt_field_kwargs = {"n_feats": n_tgt_feats,
                        "include_lengths": False,
                        "pad": pad, "bos": bos, "eos": eos,
                        "truncate": tgt_truncate,
                        "base_name": "tgt"}
    fields["tgt"] = fields_getters["text"](**tgt_field_kwargs)

    indices = Field(use_vocab=False, dtype=torch.long, sequential=False)
    fields["indices"] = indices

    if dynamic_dict:
        src_map = Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_src, sequential=False)
        fields["src_map"] = src_map

        src_ex_vocab = RawField()
        fields["src_ex_vocab"] = src_ex_vocab

        align = Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False)
        fields["alignment"] = align

    if with_align:
        word_align = AlignField()
        fields["align"] = word_align

    return fields


def load_old_vocab(vocab, data_type="text", dynamic_dict=False):
    """Update a legacy vocab/field format.

    Args:
        vocab: a list of (field name, torchtext.vocab.Vocab) pairs. This is the
            format formerly saved in *.vocab.pt files. Or, text data
            not using a :class:`TextMultiField`.
        data_type (str): text, img, or audio
        dynamic_dict (bool): Used for copy attention.

    Returns:
        a dictionary whose keys are the field names and whose values Fields.
    """

    if _old_style_vocab(vocab):
        # List[Tuple[str, Vocab]] -> List[Tuple[str, Field]]
        # -> dict[str, Field]
        vocab = dict(vocab)
        n_src_features = sum('src_feat_' in k for k in vocab)
        n_tgt_features = sum('tgt_feat_' in k for k in vocab)
        fields = get_fields(
            data_type, n_src_features, n_tgt_features,
            dynamic_dict=dynamic_dict)
        for n, f in fields.items():
            try:
                f_iter = iter(f)
            except TypeError:
                f_iter = [(n, f)]
            for sub_n, sub_f in f_iter:
                if sub_n in vocab:
                    sub_f.vocab = vocab[sub_n]
        return fields

    if _old_style_field_list(vocab):  # upgrade to multifield
        # Dict[str, List[Tuple[str, Field]]]
        # doesn't change structure - don't return early.
        fields = vocab
        for base_name, vals in fields.items():
            if ((base_name == 'src' and data_type == 'text') or
                    base_name == 'tgt'):
                assert not isinstance(vals[0][1], TextMultiField)
                fields[base_name] = [(base_name, TextMultiField(
                    vals[0][0], vals[0][1], vals[1:]))]

    if _old_style_nesting(vocab):
        # Dict[str, List[Tuple[str, Field]]] -> List[Tuple[str, Field]]
        # -> dict[str, Field]
        fields = dict(list(chain.from_iterable(vocab.values())))

    return fields


def _old_style_vocab(vocab):
    """Detect old-style vocabs (``List[Tuple[str, torchtext.data.Vocab]]``).

    Args:
        vocab: some object loaded from a *.vocab.pt file

    Returns:
        Whether ``vocab`` is a list of pairs where the second object
        is a :class:`torchtext.vocab.Vocab` object.

    This exists because previously only the vocab objects from the fields
    were saved directly, not the fields themselves, and the fields needed to
    be reconstructed at training and translation time.
    """

    return isinstance(vocab, list) and \
        any(isinstance(v[1], Vocab) for v in vocab)


def _old_style_nesting(vocab):
    """Detect old-style nesting (``dict[str, List[Tuple[str, Field]]]``)."""
    return isinstance(vocab, dict) and \
        any(isinstance(v, list) for v in vocab.values())


def _old_style_field_list(vocab):
    """Detect old-style text fields.

    Not old style vocab, old nesting, and text-type fields not using
    ``TextMultiField``.

    Args:
        vocab: some object loaded from a *.vocab.pt file

    Returns:
        Whether ``vocab`` is not an :func:`_old_style_vocab` and not
        a :class:`TextMultiField` (using an old-style text representation).
    """

    # if tgt isn't using TextMultiField, then no text field is.
    return (not _old_style_vocab(vocab)) and _old_style_nesting(vocab) and \
        (not isinstance(vocab['tgt'][0][1], TextMultiField))


def old_style_vocab(vocab):
    """The vocab/fields need updated."""
    return _old_style_vocab(vocab) or _old_style_field_list(vocab) or \
        _old_style_nesting(vocab)


def filter_example(ex, use_src_len=True, use_tgt_len=True,
                   min_src_len=1, max_src_len=float('inf'),
                   min_tgt_len=1, max_tgt_len=float('inf')):
    """Return whether an example is an acceptable length.

    If used with a dataset as ``filter_pred``, use :func:`partial()`
    for all keyword arguments.

    Args:
        ex (torchtext.data.Example): An object with a ``src`` and ``tgt``
            property.
        use_src_len (bool): Filter based on the length of ``ex.src``.
        use_tgt_len (bool): Similar to above.
        min_src_len (int): A non-negative minimally acceptable length
            (examples of exactly this length will be included).
        min_tgt_len (int): Similar to above.
        max_src_len (int or float): A non-negative (possibly infinite)
            maximally acceptable length (examples of exactly this length
            will be included).
        max_tgt_len (int or float): Similar to above.
    """

    src_len = len(ex.src[0])
    tgt_len = len(ex.tgt[0])
    return (not use_src_len or min_src_len <= src_len <= max_src_len) and \
        (not use_tgt_len or min_tgt_len <= tgt_len <= max_tgt_len)


def _pad_vocab_to_multiple(vocab, multiple):
    vocab_size = len(vocab)
    if vocab_size % multiple == 0:
        return
    target_size = int(math.ceil(vocab_size / multiple)) * multiple
    padding_tokens = [
        "averyunlikelytoken%d" % i for i in range(target_size - vocab_size)]
    vocab.extend(Vocab(Counter(), specials=padding_tokens))
    return vocab


def _build_field_vocab(field, counter, size_multiple=1, **kwargs):
    # this is basically copy-pasted from torchtext.
    all_specials = [
        field.unk_token, field.pad_token, field.init_token, field.eos_token
    ]
    specials = [tok for tok in all_specials if tok is not None]
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)
    if size_multiple > 1:
        _pad_vocab_to_multiple(field.vocab, size_multiple)


def _load_vocab(vocab_path, name, counters, min_freq):
    # counters changes in place
    vocab = _read_vocab_file(vocab_path, name)
    vocab_size = len(vocab)
    logger.info('Loaded %s vocab has %d tokens.' % (name, vocab_size))
    for i, token in enumerate(vocab):
        # keep the order of tokens specified in the vocab file by
        # adding them to the counter with decreasing counting values
        counters[name][token] = vocab_size - i + min_freq
    return vocab, vocab_size


def _build_fv_from_multifield(multifield, counters, build_fv_args,
                              size_multiple=1):
    for name, field in multifield:
        _build_field_vocab(
            field,
            counters[name],
            size_multiple=size_multiple,
            **build_fv_args[name])
        logger.info(" * %s vocab size: %d." % (name, len(field.vocab)))


def _build_fields_vocab(fields, counters, data_type, share_vocab,
                        vocab_size_multiple,
                        src_vocab_size, src_words_min_frequency,
                        tgt_vocab_size, tgt_words_min_frequency):
    build_fv_args = defaultdict(dict)
    build_fv_args["src"] = dict(
        max_size=src_vocab_size, min_freq=src_words_min_frequency)
    build_fv_args["tgt"] = dict(
        max_size=tgt_vocab_size, min_freq=tgt_words_min_frequency)
    tgt_multifield = fields["tgt"]
    _build_fv_from_multifield(
        tgt_multifield,
        counters,
        build_fv_args,
        size_multiple=vocab_size_multiple if not share_vocab else 1)
    if data_type == 'text':
        src_multifield = fields["src"]
        _build_fv_from_multifield(
            src_multifield,
            counters,
            build_fv_args,
            size_multiple=vocab_size_multiple if not share_vocab else 1)
        if share_vocab:
            # `tgt_vocab_size` is ignored when sharing vocabularies
            logger.info(" * merging src and tgt vocab...")
            src_field = src_multifield.base_field
            tgt_field = tgt_multifield.base_field
            _merge_field_vocabs(
                src_field, tgt_field, vocab_size=src_vocab_size,
                min_freq=src_words_min_frequency,
                vocab_size_multiple=vocab_size_multiple)
            logger.info(" * merged vocab size: %d." % len(src_field.vocab))

    return fields


def build_vocab(train_dataset_files, fields, data_type, share_vocab,
                src_vocab_path, src_vocab_size, src_words_min_frequency,
                tgt_vocab_path, tgt_vocab_size, tgt_words_min_frequency,
                vocab_size_multiple=1):
    """Build the fields for all data sides.

    Args:
        train_dataset_files: a list of train dataset pt file.
        fields (dict[str, Field]): fields to build vocab for.
        data_type (str): A supported data type string.
        share_vocab (bool): share source and target vocabulary?
        src_vocab_path (str): Path to src vocabulary file.
        src_vocab_size (int): size of the source vocabulary.
        src_words_min_frequency (int): the minimum frequency needed to
            include a source word in the vocabulary.
        tgt_vocab_path (str): Path to tgt vocabulary file.
        tgt_vocab_size (int): size of the target vocabulary.
        tgt_words_min_frequency (int): the minimum frequency needed to
            include a target word in the vocabulary.
        vocab_size_multiple (int): ensure that the vocabulary size is a
            multiple of this value.

    Returns:
        Dict of Fields
    """

    counters = defaultdict(Counter)

    if src_vocab_path:
        try:
            logger.info("Using existing vocabulary...")
            vocab = torch.load(src_vocab_path)
            # return vocab to dump with standard name
            return vocab
        except torch.serialization.pickle.UnpicklingError:
            logger.info("Building vocab from text file...")
            # empty train_dataset_files so that vocab is only loaded from
            # given paths in src_vocab_path, tgt_vocab_path
            train_dataset_files = []

    # Load vocabulary
    if src_vocab_path:
        src_vocab, src_vocab_size = _load_vocab(
            src_vocab_path, "src", counters,
            src_words_min_frequency)
    else:
        src_vocab = None

    if tgt_vocab_path:
        tgt_vocab, tgt_vocab_size = _load_vocab(
            tgt_vocab_path, "tgt", counters,
            tgt_words_min_frequency)
    else:
        tgt_vocab = None

    for i, path in enumerate(train_dataset_files):
        dataset = torch.load(path)
        logger.info(" * reloading %s." % path)
        for ex in dataset.examples:
            for name, field in fields.items():
                try:
                    f_iter = iter(field)
                except TypeError:
                    f_iter = [(name, field)]
                    all_data = [getattr(ex, name, None)]
                else:
                    all_data = getattr(ex, name)
                for (sub_n, sub_f), fd in zip(
                        f_iter, all_data):
                    has_vocab = (sub_n == 'src' and src_vocab) or \
                                (sub_n == 'tgt' and tgt_vocab)
                    if sub_f.sequential and not has_vocab:
                        val = fd
                        counters[sub_n].update(val)

        # Drop the none-using from memory but keep the last
        if i < len(train_dataset_files) - 1:
            dataset.examples = None
            gc.collect()
            del dataset.examples
            gc.collect()
            del dataset
            gc.collect()

    fields = _build_fields_vocab(
        fields, counters, data_type,
        share_vocab, vocab_size_multiple,
        src_vocab_size, src_words_min_frequency,
        tgt_vocab_size, tgt_words_min_frequency)

    return fields  # is the return necessary?


def _merge_field_vocabs(src_field, tgt_field, vocab_size, min_freq,
                        vocab_size_multiple):
    # in the long run, shouldn't it be possible to do this by calling
    # build_vocab with both the src and tgt data?
    specials = [tgt_field.unk_token, tgt_field.pad_token,
                tgt_field.init_token, tgt_field.eos_token]
    merged = sum(
        [src_field.vocab.freqs, tgt_field.vocab.freqs], Counter()
    )
    merged_vocab = Vocab(
        merged, specials=specials,
        max_size=vocab_size, min_freq=min_freq
    )
    if vocab_size_multiple > 1:
        _pad_vocab_to_multiple(merged_vocab, vocab_size_multiple)
    src_field.vocab = merged_vocab
    tgt_field.vocab = merged_vocab
    assert len(src_field.vocab) == len(tgt_field.vocab)


def _read_vocab_file(vocab_path, tag):
    """Loads a vocabulary from the given path.

    Args:
        vocab_path (str): Path to utf-8 text file containing vocabulary.
            Each token should be on a line by itself. Tokens must not
            contain whitespace (else only before the whitespace
            is considered).
        tag (str): Used for logging which vocab is being read.
    """

    logger.info("Loading {} vocabulary from {}".format(tag, vocab_path))

    if not os.path.exists(vocab_path):
        raise RuntimeError(
            "{} vocabulary not found at {}".format(tag, vocab_path))
    else:
        with codecs.open(vocab_path, 'r', 'utf-8') as f:
            return [line.strip().split()[0] for line in f if line.strip()]


def batch_iter(data, batch_size, batch_size_fn=None, batch_size_multiple=1):
    """Yield elements from data in chunks of batch_size, where each chunk size
    is a multiple of batch_size_multiple.

    This is an extended version of torchtext.data.batch.
    """
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return count
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far >= batch_size:
            overflowed = 0
            if size_so_far > batch_size:
                overflowed += 1
            if batch_size_multiple > 1:
                overflowed += (
                    (len(minibatch) - overflowed) % batch_size_multiple)
            if overflowed == 0:
                yield minibatch
                minibatch, size_so_far = [], 0
            else:
                if overflowed == len(minibatch):
                    logger.warning(
                        "An example was ignored, more tokens"
                        " than allowed by tokens batch_size")
                else:
                    yield minibatch[:-overflowed]
                    minibatch = minibatch[-overflowed:]
                    size_so_far = 0
                    for i, ex in enumerate(minibatch):
                        size_so_far = batch_size_fn(ex, i + 1, size_so_far)
    if minibatch:
        yield minibatch


def _pool(data, batch_size, batch_size_fn, batch_size_multiple,
          sort_key, random_shuffler, pool_factor):
    for p in torchtext.data.batch(
            data, batch_size * pool_factor,
            batch_size_fn=batch_size_fn):
        p_batch = list(batch_iter(
            sorted(p, key=sort_key),
            batch_size,
            batch_size_fn=batch_size_fn,
            batch_size_multiple=batch_size_multiple))
        for b in random_shuffler(p_batch):
            yield b


class OrderedIterator(torchtext.data.Iterator):

    def __init__(self,
                 dataset,
                 batch_size,
                 pool_factor=1,
                 batch_size_multiple=1,
                 yield_raw_example=False,
                 **kwargs):
        super(OrderedIterator, self).__init__(dataset, batch_size, **kwargs)
        self.batch_size_multiple = batch_size_multiple
        self.yield_raw_example = yield_raw_example
        self.dataset = dataset
        self.pool_factor = pool_factor

    def create_batches(self):
        if self.train:
            if self.yield_raw_example:
                self.batches = batch_iter(
                    self.data(),
                    1,
                    batch_size_fn=None,
                    batch_size_multiple=1)
            else:
                self.batches = _pool(
                    self.data(),
                    self.batch_size,
                    self.batch_size_fn,
                    self.batch_size_multiple,
                    self.sort_key,
                    self.random_shuffler,
                    self.pool_factor)
        else:
            self.batches = []
            for b in batch_iter(
                    self.data(),
                    self.batch_size,
                    batch_size_fn=self.batch_size_fn,
                    batch_size_multiple=self.batch_size_multiple):
                self.batches.append(sorted(b, key=self.sort_key))

    def __iter__(self):
        """
        Extended version of the definition in torchtext.data.Iterator.
        Added yield_raw_example behaviour to yield a torchtext.data.Example
        instead of a torchtext.data.Batch object.
        """
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a
                    # minibatch be sorted by decreasing order, which
                    #  requires reversing relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                if self.yield_raw_example:
                    yield minibatch[0]
                else:
                    yield torchtext.data.Batch(
                        minibatch,
                        self.dataset,
                        self.device)
            if not self.repeat:
                return


class MultipleDatasetIterator(object):
    """
    This takes a list of iterable objects (DatasetLazyIter) and their
    respective weights, and yields a batch in the wanted proportions.
    """
    def __init__(self,
                 train_shards,
                 fields,
                 device,
                 opt):
        self.index = -1
        self.iterables = []
        for shard in train_shards:
            self.iterables.append(
                build_dataset_iter(shard, fields, opt, multi=True))
        self.init_iterators = True
        self.weights = opt.data_weights
        self.batch_size = opt.batch_size
        self.batch_size_fn = max_tok_len \
            if opt.batch_type == "tokens" else None
        self.batch_size_multiple = 8 if opt.model_dtype == "fp16" else 1
        self.device = device
        # Temporarily load one shard to retrieve sort_key for data_type
        temp_dataset = torch.load(self.iterables[0]._paths[0])
        self.sort_key = temp_dataset.sort_key
        self.random_shuffler = RandomShuffler()
        self.pool_factor = opt.pool_factor
        del temp_dataset

    def _iter_datasets(self):
        if self.init_iterators:
            self.iterators = [iter(iterable) for iterable in self.iterables]
            self.init_iterators = False
        for weight in self.weights:
            self.index = (self.index + 1) % len(self.iterators)
            for i in range(weight):
                yield self.iterators[self.index]

    def _iter_examples(self):
        for iterator in cycle(self._iter_datasets()):
            yield next(iterator)

    def __iter__(self):
        while True:
            for minibatch in _pool(
                    self._iter_examples(),
                    self.batch_size,
                    self.batch_size_fn,
                    self.batch_size_multiple,
                    self.sort_key,
                    self.random_shuffler,
                    self.pool_factor):
                minibatch = sorted(minibatch, key=self.sort_key, reverse=True)
                yield torchtext.data.Batch(minibatch,
                                           self.iterables[0].dataset,
                                           self.device)


class DatasetLazyIter(object):
    """Yield data from sharded dataset files.

    Args:
        dataset_paths: a list containing the locations of dataset files.
        fields (dict[str, Field]): fields dict for the
            datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: See :class:`OrderedIterator` ``device``.
        is_train (bool): train or valid?
    """

    def __init__(self, dataset_paths, fields, batch_size, batch_size_fn,
                 batch_size_multiple, device, is_train, pool_factor,
                 repeat=True, num_batches_multiple=1, yield_raw_example=False):
        self._paths = dataset_paths
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.batch_size_multiple = batch_size_multiple
        self.device = device
        self.is_train = is_train
        self.repeat = repeat
        self.num_batches_multiple = num_batches_multiple
        self.yield_raw_example = yield_raw_example
        self.pool_factor = pool_factor

    def _iter_dataset(self, path):
        logger.info('Loading dataset from %s' % path)
        cur_dataset = torch.load(path)
        logger.info('number of examples: %d' % len(cur_dataset))
        cur_dataset.fields = self.fields
        cur_iter = OrderedIterator(
            dataset=cur_dataset,
            batch_size=self.batch_size,
            pool_factor=self.pool_factor,
            batch_size_multiple=self.batch_size_multiple,
            batch_size_fn=self.batch_size_fn,
            device=self.device,
            train=self.is_train,
            sort=False,
            sort_within_batch=True,
            repeat=False,
            yield_raw_example=self.yield_raw_example
        )
        for batch in cur_iter:
            self.dataset = cur_iter.dataset
            yield batch

        # NOTE: This is causing some issues for consumer/producer,
        # as we may still have some of those examples in some queue
        # cur_dataset.examples = None
        # gc.collect()
        # del cur_dataset
        # gc.collect()

    def __iter__(self):
        num_batches = 0
        paths = self._paths
        if self.is_train and self.repeat:
            # Cycle through the shards indefinitely.
            paths = cycle(paths)
        for path in paths:
            for batch in self._iter_dataset(path):
                yield batch
                num_batches += 1
        if self.is_train and not self.repeat and \
           num_batches % self.num_batches_multiple != 0:
            # When the dataset is not repeated, we might need to ensure that
            # the number of returned batches is the multiple of a given value.
            # This is important for multi GPU training to ensure that all
            # workers have the same number of batches to process.
            for path in paths:
                for batch in self._iter_dataset(path):
                    yield batch
                    num_batches += 1
                    if num_batches % self.num_batches_multiple == 0:
                        return


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch, max_tgt_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt[0]) + 1)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def build_dataset_iter(corpus_type, fields, opt, is_train=True, multi=False):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over. We implement simple ordered iterator strategy here,
    but more sophisticated strategy like curriculum learning is ok too.
    """
    dataset_paths = list(sorted(
        glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt')))
    if not dataset_paths:
        if is_train:
            raise ValueError('Training data %s not found' % opt.data)
        else:
            return None
    if multi:
        batch_size = 1
        batch_fn = None
        batch_size_multiple = 1
    else:
        batch_size = opt.batch_size if is_train else opt.valid_batch_size
        batch_fn = max_tok_len \
            if is_train and opt.batch_type == "tokens" else None
        batch_size_multiple = 8 if opt.model_dtype == "fp16" else 1

    device = "cuda" if opt.gpu_ranks else "cpu"

    return DatasetLazyIter(
        dataset_paths,
        fields,
        batch_size,
        batch_fn,
        batch_size_multiple,
        device,
        is_train,
        opt.pool_factor,
        repeat=not opt.single_pass,
        num_batches_multiple=max(opt.accum_count) * opt.world_size,
        yield_raw_example=multi)


def build_dataset_iter_multiple(train_shards, fields, opt):
    return MultipleDatasetIterator(
        train_shards, fields, "cuda" if opt.gpu_ranks else "cpu", opt)
