import torch


class DecodeStrategy(object):
    """Base class for generation strategies.

    Args:
        pad (int): Magic integer in output vocab.
        bos (int): Magic integer in output vocab.
        eos (int): Magic integer in output vocab.
        batch_size (int): Current batch size.
        parallel_paths (int): Decoding strategies like beam search
            use parallel paths. Each batch is repeated ``parallel_paths``
            times in relevant state tensors.
        min_length (int): Shortest acceptable generation, not counting
            begin-of-sentence or end-of-sentence.
        max_length (int): Longest acceptable sequence, not counting
            begin-of-sentence (presumably there has been no EOS
            yet if max_length is used as a cutoff).
        block_ngram_repeat (int): Block beams where
            ``block_ngram_repeat``-grams repeat.
        exclusion_tokens (set[int]): If a gram contains any of these
            tokens, it may repeat.
        return_attention (bool): Whether to work with attention too. If this
            is true, it is assumed that the decoder is attentional.

    Attributes:
        pad (int): See above.
        bos (int): See above.
        eos (int): See above.
        predictions (list[list[LongTensor]]): For each batch, holds a
            list of beam prediction sequences.
        scores (list[list[FloatTensor]]): For each batch, holds a
            list of scores.
        attention (list[list[FloatTensor or list[]]]): For each
            batch, holds a list of attention sequence tensors
            (or empty lists) having shape ``(step, inp_seq_len)`` where
            ``inp_seq_len`` is the length of the sample (not the max
            length of all inp seqs).
        alive_seq (LongTensor): Shape ``(B x parallel_paths, step)``.
            This sequence grows in the ``step`` axis on each call to
            :func:`advance()`.
        is_finished (ByteTensor or NoneType): Shape
            ``(B, parallel_paths)``. Initialized to ``None``.
        alive_attn (FloatTensor or NoneType): If tensor, shape is
            ``(step, B x parallel_paths, inp_seq_len)``, where ``inp_seq_len``
            is the (max) length of the input sequence.
        min_length (int): See above.
        max_length (int): See above.
        block_ngram_repeat (int): See above.
        exclusion_tokens (set[int]): See above.
        return_attention (bool): See above.
        done (bool): See above.
    """

    def __init__(self, pad, bos, eos, batch_size, parallel_paths,
                 min_length, block_ngram_repeat, exclusion_tokens,
                 return_attention, max_length):

        # magic indices
        self.pad = pad
        self.bos = bos
        self.eos = eos

        self.batch_size = batch_size
        self.parallel_paths = parallel_paths
        # result caching
        self.predictions = [[] for _ in range(batch_size)]
        self.scores = [[] for _ in range(batch_size)]
        self.attention = [[] for _ in range(batch_size)]

        self.alive_attn = None

        self.min_length = min_length
        self.max_length = max_length
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens
        self.return_attention = return_attention

        self.done = False

    def initialize(self, memory_bank, src_lengths, src_map=None, device=None):
        """DecodeStrategy subclasses should override :func:`initialize()`.

        `initialize` should be called before all actions.
        used to prepare necessary ingredients for decode.
        """
        if device is None:
            device = torch.device('cpu')
        self.alive_seq = torch.full(
            [self.batch_size * self.parallel_paths, 1], self.bos,
            dtype=torch.long, device=device)
        self.is_finished = torch.zeros(
            [self.batch_size, self.parallel_paths],
            dtype=torch.uint8, device=device)
        return None, memory_bank, src_lengths, src_map

    def __len__(self):
        return self.alive_seq.shape[1]

    def ensure_min_length(self, log_probs):
        if len(self) <= self.min_length:
            log_probs[:, self.eos] = -1e20

    def ensure_max_length(self):
        # add one to account for BOS. Don't account for EOS because hitting
        # this implies it hasn't been found.
        if len(self) == self.max_length + 1:
            self.is_finished.fill_(1)

    def block_ngram_repeats(self, log_probs):
        cur_len = len(self)
        if self.block_ngram_repeat > 0 and cur_len > 1:
            for path_idx in range(self.alive_seq.shape[0]):
                # skip BOS
                hyp = self.alive_seq[path_idx, 1:]
                ngrams = set()
                fail = False
                gram = []
                for i in range(cur_len - 1):
                    # Last n tokens, n = block_ngram_repeat
                    gram = (gram + [hyp[i].item()])[-self.block_ngram_repeat:]
                    # skip the blocking if any token in gram is excluded
                    if set(gram) & self.exclusion_tokens:
                        continue
                    if tuple(gram) in ngrams:
                        fail = True
                    ngrams.add(tuple(gram))
                if fail:
                    log_probs[path_idx] = -10e20

    def advance(self, log_probs, attn):
        """DecodeStrategy subclasses should override :func:`advance()`.

        Advance is used to update ``self.alive_seq``, ``self.is_finished``,
        and, when appropriate, ``self.alive_attn``.
        """

        raise NotImplementedError()

    def update_finished(self):
        """DecodeStrategy subclasses should override :func:`update_finished()`.

        ``update_finished`` is used to update ``self.predictions``,
        ``self.scores``, and other "output" attributes.
        """

        raise NotImplementedError()
