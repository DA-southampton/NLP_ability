"""Ensemble decoding.

Decodes using multiple models simultaneously,
combining their prediction distributions by averaging.
All models in the ensemble must share a target vocabulary.
"""

import torch
import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.decoders.decoder import DecoderBase
from onmt.models import NMTModel
import onmt.model_builder


class EnsembleDecoderOutput(object):
    """Wrapper around multiple decoder final hidden states."""
    def __init__(self, model_dec_outs):
        self.model_dec_outs = tuple(model_dec_outs)

    def squeeze(self, dim=None):
        """Delegate squeeze to avoid modifying
        :func:`onmt.translate.translator.Translator.translate_batch()`
        """
        return EnsembleDecoderOutput([
            x.squeeze(dim) for x in self.model_dec_outs])

    def __getitem__(self, index):
        return self.model_dec_outs[index]


class EnsembleEncoder(EncoderBase):
    """Dummy Encoder that delegates to individual real Encoders."""
    def __init__(self, model_encoders):
        super(EnsembleEncoder, self).__init__()
        self.model_encoders = nn.ModuleList(model_encoders)

    def forward(self, src, lengths=None):
        enc_hidden, memory_bank, _ = zip(*[
            model_encoder(src, lengths)
            for model_encoder in self.model_encoders])
        return enc_hidden, memory_bank, lengths


class EnsembleDecoder(DecoderBase):
    """Dummy Decoder that delegates to individual real Decoders."""
    def __init__(self, model_decoders):
        model_decoders = nn.ModuleList(model_decoders)
        attentional = any([dec.attentional for dec in model_decoders])
        super(EnsembleDecoder, self).__init__(attentional)
        self.model_decoders = model_decoders

    def forward(self, tgt, memory_bank, memory_lengths=None, step=None,
                **kwargs):
        """See :func:`onmt.decoders.decoder.DecoderBase.forward()`."""
        # Memory_lengths is a single tensor shared between all models.
        # This assumption will not hold if Translator is modified
        # to calculate memory_lengths as something other than the length
        # of the input.
        dec_outs, attns = zip(*[
            model_decoder(
                tgt, memory_bank[i],
                memory_lengths=memory_lengths, step=step)
            for i, model_decoder in enumerate(self.model_decoders)])
        mean_attns = self.combine_attns(attns)
        return EnsembleDecoderOutput(dec_outs), mean_attns

    def combine_attns(self, attns):
        result = {}
        for key in attns[0].keys():
            result[key] = torch.stack(
                [attn[key] for attn in attns if attn[key] is not None]).mean(0)
        return result

    def init_state(self, src, memory_bank, enc_hidden):
        """ See :obj:`RNNDecoderBase.init_state()` """
        for i, model_decoder in enumerate(self.model_decoders):
            model_decoder.init_state(src, memory_bank[i], enc_hidden[i])

    def map_state(self, fn):
        for model_decoder in self.model_decoders:
            model_decoder.map_state(fn)


class EnsembleGenerator(nn.Module):
    """
    Dummy Generator that delegates to individual real Generators,
    and then averages the resulting target distributions.
    """
    def __init__(self, model_generators, raw_probs=False):
        super(EnsembleGenerator, self).__init__()
        self.model_generators = nn.ModuleList(model_generators)
        self._raw_probs = raw_probs

    def forward(self, hidden, attn=None, src_map=None):
        """
        Compute a distribution over the target dictionary
        by averaging distributions from models in the ensemble.
        All models in the ensemble must share a target vocabulary.
        """
        distributions = torch.stack(
                [mg(h) if attn is None else mg(h, attn, src_map)
                 for h, mg in zip(hidden, self.model_generators)]
            )
        if self._raw_probs:
            return torch.log(torch.exp(distributions).mean(0))
        else:
            return distributions.mean(0)


class EnsembleModel(NMTModel):
    """Dummy NMTModel wrapping individual real NMTModels."""
    def __init__(self, models, raw_probs=False):
        encoder = EnsembleEncoder(model.encoder for model in models)
        decoder = EnsembleDecoder(model.decoder for model in models)
        super(EnsembleModel, self).__init__(encoder, decoder)
        self.generator = EnsembleGenerator(
            [model.generator for model in models], raw_probs)
        self.models = nn.ModuleList(models)


def load_test_model(opt):
    """Read in multiple models for ensemble."""
    shared_fields = None
    shared_model_opt = None
    models = []
    for model_path in opt.models:
        fields, model, model_opt = \
            onmt.model_builder.load_test_model(opt, model_path=model_path)
        if shared_fields is None:
            shared_fields = fields
        else:
            for key, field in fields.items():
                try:
                    f_iter = iter(field)
                except TypeError:
                    f_iter = [(key, field)]
                for sn, sf in f_iter:
                    if sf is not None and 'vocab' in sf.__dict__:
                        sh_field = shared_fields[key]
                        try:
                            sh_f_iter = iter(sh_field)
                        except TypeError:
                            sh_f_iter = [(key, sh_field)]
                        sh_f_dict = dict(sh_f_iter)
                        assert sf.vocab.stoi == sh_f_dict[sn].vocab.stoi, \
                            "Ensemble models must use the same " \
                            "preprocessed data"
        models.append(model)
        if shared_model_opt is None:
            shared_model_opt = model_opt
    ensemble_model = EnsembleModel(models, opt.avg_raw_probs)
    return shared_fields, ensemble_model, shared_model_opt
