import torch
from onmt.translate import penalties
from onmt.translate.decode_strategy import DecodeStrategy
from onmt.utils.misc import tile

import warnings


class BeamSearch(DecodeStrategy):
    """Generation beam search.

    Note that the attributes list is not exhaustive. Rather, it highlights
    tensors to document their shape. (Since the state variables' "batch"
    size decreases as beams finish, we denote this axis with a B rather than
    ``batch_size``).

    Args:
        beam_size (int): Number of beams to use (see base ``parallel_paths``).
        batch_size (int): See base.
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        n_best (int): Don't stop until at least this many beams have
            reached EOS.
        global_scorer (onmt.translate.GNMTGlobalScorer): Scorer instance.
        min_length (int): See base.
        max_length (int): See base.
        return_attention (bool): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.

    Attributes:
        top_beam_finished (ByteTensor): Shape ``(B,)``.
        _batch_offset (LongTensor): Shape ``(B,)``.
        _beam_offset (LongTensor): Shape ``(batch_size x beam_size,)``.
        alive_seq (LongTensor): See base.
        topk_log_probs (FloatTensor): Shape ``(B x beam_size,)``. These
            are the scores used for the topk operation.
        memory_lengths (LongTensor): Lengths of encodings. Used for
            masking attentions.
        select_indices (LongTensor or NoneType): Shape
            ``(B x beam_size,)``. This is just a flat view of the
            ``_batch_index``.
        topk_scores (FloatTensor): Shape
            ``(B, beam_size)``. These are the
            scores a sequence will receive if it finishes.
        topk_ids (LongTensor): Shape ``(B, beam_size)``. These are the
            word indices of the topk predictions.
        _batch_index (LongTensor): Shape ``(B, beam_size)``.
        _prev_penalty (FloatTensor or NoneType): Shape
            ``(B, beam_size)``. Initialized to ``None``.
        _coverage (FloatTensor or NoneType): Shape
            ``(1, B x beam_size, inp_seq_len)``.
        hypotheses (list[list[Tuple[Tensor]]]): Contains a tuple
            of score (float), sequence (long), and attention (float or None).
    """

    def __init__(self, beam_size, batch_size, pad, bos, eos, n_best,
                 global_scorer, min_length, max_length, return_attention,
                 block_ngram_repeat, exclusion_tokens,
                 stepwise_penalty, ratio):
        super(BeamSearch, self).__init__(
            pad, bos, eos, batch_size, beam_size, min_length,
            block_ngram_repeat, exclusion_tokens, return_attention,
            max_length)
        # beam parameters
        self.global_scorer = global_scorer
        self.beam_size = beam_size
        self.n_best = n_best
        self.ratio = ratio

        # result caching
        self.hypotheses = [[] for _ in range(batch_size)]

        # beam state
        self.top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
        # BoolTensor was introduced in pytorch 1.2
        try:
            self.top_beam_finished = self.top_beam_finished.bool()
        except AttributeError:
            pass
        self._batch_offset = torch.arange(batch_size, dtype=torch.long)

        self.select_indices = None
        self.done = False
        # "global state" of the old beam
        self._prev_penalty = None
        self._coverage = None

        self._stepwise_cov_pen = (
                stepwise_penalty and self.global_scorer.has_cov_pen)
        self._vanilla_cov_pen = (
            not stepwise_penalty and self.global_scorer.has_cov_pen)
        self._cov_pen = self.global_scorer.has_cov_pen

    def initialize(self, memory_bank, src_lengths, src_map=None, device=None):
        """Initialize for decoding.
        Repeat src objects `beam_size` times.
        """

        def fn_map_state(state, dim):
            return tile(state, self.beam_size, dim=dim)

        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, self.beam_size, dim=1)
                                for x in memory_bank)
            mb_device = memory_bank[0].device
        else:
            memory_bank = tile(memory_bank, self.beam_size, dim=1)
            mb_device = memory_bank.device
        if src_map is not None:
            src_map = tile(src_map, self.beam_size, dim=1)
        if device is None:
            device = mb_device

        self.memory_lengths = tile(src_lengths, self.beam_size)
        super(BeamSearch, self).initialize(
            memory_bank, self.memory_lengths, src_map, device)
        self.best_scores = torch.full(
            [self.batch_size], -1e10, dtype=torch.float, device=device)
        self._beam_offset = torch.arange(
            0, self.batch_size * self.beam_size, step=self.beam_size,
            dtype=torch.long, device=device)
        self.topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (self.beam_size - 1), device=device
        ).repeat(self.batch_size)
        # buffers for the topk scores and 'backpointer'
        self.topk_scores = torch.empty((self.batch_size, self.beam_size),
                                       dtype=torch.float, device=device)
        self.topk_ids = torch.empty((self.batch_size, self.beam_size),
                                    dtype=torch.long, device=device)
        self._batch_index = torch.empty([self.batch_size, self.beam_size],
                                        dtype=torch.long, device=device)
        return fn_map_state, memory_bank, self.memory_lengths, src_map

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def current_backptr(self):
        # for testing
        return self.select_indices.view(self.batch_size, self.beam_size)\
            .fmod(self.beam_size)

    @property
    def batch_offset(self):
        return self._batch_offset

    def advance(self, log_probs, attn):
        vocab_size = log_probs.size(-1)

        # using integer division to get an integer _B without casting
        _B = log_probs.shape[0] // self.beam_size

        if self._stepwise_cov_pen and self._prev_penalty is not None:
            self.topk_log_probs += self._prev_penalty
            self.topk_log_probs -= self.global_scorer.cov_penalty(
                self._coverage + attn, self.global_scorer.beta).view(
                _B, self.beam_size)

        # force the output to be longer than self.min_length
        step = len(self)
        self.ensure_min_length(log_probs)

        # Multiply probs by the beam probability.
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)

        self.block_ngram_repeats(log_probs)

        # if the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token
        length_penalty = self.global_scorer.length_penalty(
            step + 1, alpha=self.global_scorer.alpha)

        # Flatten probs into a list of possibilities.
        curr_scores = log_probs / length_penalty
        curr_scores = curr_scores.reshape(_B, self.beam_size * vocab_size)
        torch.topk(curr_scores,  self.beam_size, dim=-1,
                   out=(self.topk_scores, self.topk_ids))

        # Recover log probs.
        # Length penalty is just a scalar. It doesn't matter if it's applied
        # before or after the topk.
        torch.mul(self.topk_scores, length_penalty, out=self.topk_log_probs)

        # Resolve beam origin and map to batch index flat representation.
        torch.div(self.topk_ids, vocab_size, out=self._batch_index)
        self._batch_index += self._beam_offset[:_B].unsqueeze(1)
        self.select_indices = self._batch_index.view(_B * self.beam_size)
        self.topk_ids.fmod_(vocab_size)  # resolve true word ids

        # Append last prediction.
        self.alive_seq = torch.cat(
            [self.alive_seq.index_select(0, self.select_indices),
             self.topk_ids.view(_B * self.beam_size, 1)], -1)
        if self.return_attention or self._cov_pen:
            current_attn = attn.index_select(1, self.select_indices)
            if step == 1:
                self.alive_attn = current_attn
                # update global state (step == 1)
                if self._cov_pen:  # coverage penalty
                    self._prev_penalty = torch.zeros_like(self.topk_log_probs)
                    self._coverage = current_attn
            else:
                self.alive_attn = self.alive_attn.index_select(
                    1, self.select_indices)
                self.alive_attn = torch.cat([self.alive_attn, current_attn], 0)
                # update global state (step > 1)
                if self._cov_pen:
                    self._coverage = self._coverage.index_select(
                        1, self.select_indices)
                    self._coverage += current_attn
                    self._prev_penalty = self.global_scorer.cov_penalty(
                        self._coverage, beta=self.global_scorer.beta).view(
                            _B, self.beam_size)

        if self._vanilla_cov_pen:
            # shape: (batch_size x beam_size, 1)
            cov_penalty = self.global_scorer.cov_penalty(
                self._coverage,
                beta=self.global_scorer.beta)
            self.topk_scores -= cov_penalty.view(_B, self.beam_size).float()

        self.is_finished = self.topk_ids.eq(self.eos)
        self.ensure_max_length()

    def update_finished(self):
        # Penalize beams that finished.
        _B_old = self.topk_log_probs.shape[0]
        step = self.alive_seq.shape[-1]  # 1 greater than the step in advance
        self.topk_log_probs.masked_fill_(self.is_finished, -1e10)
        # on real data (newstest2017) with the pretrained transformer,
        # it's faster to not move this back to the original device
        self.is_finished = self.is_finished.to('cpu')
        self.top_beam_finished |= self.is_finished[:, 0].eq(1)
        predictions = self.alive_seq.view(_B_old, self.beam_size, step)
        attention = (
            self.alive_attn.view(
                step - 1, _B_old, self.beam_size, self.alive_attn.size(-1))
            if self.alive_attn is not None else None)
        non_finished_batch = []
        for i in range(self.is_finished.size(0)):  # Batch level
            b = self._batch_offset[i]
            finished_hyp = self.is_finished[i].nonzero().view(-1)
            # Store finished hypotheses for this batch.
            for j in finished_hyp:  # Beam level: finished beam j in batch i
                if self.ratio > 0:
                    s = self.topk_scores[i, j] / (step + 1)
                    if self.best_scores[b] < s:
                        self.best_scores[b] = s
                self.hypotheses[b].append((
                    self.topk_scores[i, j],
                    predictions[i, j, 1:],  # Ignore start_token.
                    attention[:, i, j, :self.memory_lengths[i]]
                    if attention is not None else None))
            # End condition is the top beam finished and we can return
            # n_best hypotheses.
            if self.ratio > 0:
                pred_len = self.memory_lengths[i] * self.ratio
                finish_flag = ((self.topk_scores[i, 0] / pred_len)
                               <= self.best_scores[b]) or \
                    self.is_finished[i].all()
            else:
                finish_flag = self.top_beam_finished[i] != 0
            if finish_flag and len(self.hypotheses[b]) >= self.n_best:
                best_hyp = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)
                for n, (score, pred, attn) in enumerate(best_hyp):
                    if n >= self.n_best:
                        break
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)  # ``(batch, n_best,)``
                    self.attention[b].append(
                        attn if attn is not None else [])
            else:
                non_finished_batch.append(i)
        non_finished = torch.tensor(non_finished_batch)
        # If all sentences are translated, no need to go further.
        if len(non_finished) == 0:
            self.done = True
            return

        _B_new = non_finished.shape[0]
        # Remove finished batches for the next step.
        self.top_beam_finished = self.top_beam_finished.index_select(
            0, non_finished)
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        non_finished = non_finished.to(self.topk_ids.device)
        self.topk_log_probs = self.topk_log_probs.index_select(0,
                                                               non_finished)
        self._batch_index = self._batch_index.index_select(0, non_finished)
        self.select_indices = self._batch_index.view(_B_new * self.beam_size)
        self.alive_seq = predictions.index_select(0, non_finished) \
            .view(-1, self.alive_seq.size(-1))
        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.topk_ids = self.topk_ids.index_select(0, non_finished)
        if self.alive_attn is not None:
            inp_seq_len = self.alive_attn.size(-1)
            self.alive_attn = attention.index_select(1, non_finished) \
                .view(step - 1, _B_new * self.beam_size, inp_seq_len)
            if self._cov_pen:
                self._coverage = self._coverage \
                    .view(1, _B_old, self.beam_size, inp_seq_len) \
                    .index_select(1, non_finished) \
                    .view(1, _B_new * self.beam_size, inp_seq_len)
                if self._stepwise_cov_pen:
                    self._prev_penalty = self._prev_penalty.index_select(
                        0, non_finished)


class GNMTGlobalScorer(object):
    """NMT re-ranking.

    Args:
       alpha (float): Length parameter.
       beta (float):  Coverage parameter.
       length_penalty (str): Length penalty strategy.
       coverage_penalty (str): Coverage penalty strategy.

    Attributes:
        alpha (float): See above.
        beta (float): See above.
        length_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        coverage_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        has_cov_pen (bool): See :class:`penalties.PenaltyBuilder`.
        has_len_pen (bool): See :class:`penalties.PenaltyBuilder`.
    """

    @classmethod
    def from_opt(cls, opt):
        return cls(
            opt.alpha,
            opt.beta,
            opt.length_penalty,
            opt.coverage_penalty)

    def __init__(self, alpha, beta, length_penalty, coverage_penalty):
        self._validate(alpha, beta, length_penalty, coverage_penalty)
        self.alpha = alpha
        self.beta = beta
        penalty_builder = penalties.PenaltyBuilder(coverage_penalty,
                                                   length_penalty)
        self.has_cov_pen = penalty_builder.has_cov_pen
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty

        self.has_len_pen = penalty_builder.has_len_pen
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty

    @classmethod
    def _validate(cls, alpha, beta, length_penalty, coverage_penalty):
        # these warnings indicate that either the alpha/beta
        # forces a penalty to be a no-op, or a penalty is a no-op but
        # the alpha/beta would suggest otherwise.
        if length_penalty is None or length_penalty == "none":
            if alpha != 0:
                warnings.warn("Non-default `alpha` with no length penalty. "
                              "`alpha` has no effect.")
        else:
            # using some length penalty
            if length_penalty == "wu" and alpha == 0.:
                warnings.warn("Using length penalty Wu with alpha==0 "
                              "is equivalent to using length penalty none.")
        if coverage_penalty is None or coverage_penalty == "none":
            if beta != 0:
                warnings.warn("Non-default `beta` with no coverage penalty. "
                              "`beta` has no effect.")
        else:
            # using some coverage penalty
            if beta == 0.:
                warnings.warn("Non-default coverage penalty with beta==0 "
                              "is equivalent to using coverage penalty none.")
