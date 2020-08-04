"""Define a minimal encoder."""
from onmt.encoders.encoder import EncoderBase
from onmt.utils.misc import sequence_mask
import torch


class MeanEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       num_layers (int): number of replicated layers
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, num_layers, embeddings):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            embeddings)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        _, batch, emb_dim = emb.size()

        if lengths is not None:
            # we avoid padding while mean pooling
            mask = sequence_mask(lengths).float()
            mask = mask / lengths.unsqueeze(1).float()
            mean = torch.bmm(mask.unsqueeze(1), emb.transpose(0, 1)).squeeze(1)
        else:
            mean = emb.mean(0)

        mean = mean.expand(self.num_layers, batch, emb_dim)
        memory_bank = emb
        encoder_final = (mean, mean)
        return encoder_final, memory_bank, lengths
