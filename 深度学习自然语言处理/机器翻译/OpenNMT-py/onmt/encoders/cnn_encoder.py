"""
Implementation of "Convolutional Sequence to Sequence Learning"
"""
import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.utils.cnn_factory import shape_transform, StackedCNN

SCALE_WEIGHT = 0.5 ** 0.5


class CNNEncoder(EncoderBase):
    """Encoder based on "Convolutional Sequence to Sequence Learning"
    :cite:`DBLP:journals/corr/GehringAGYD17`.
    """

    def __init__(self, num_layers, hidden_size,
                 cnn_kernel_width, dropout, embeddings):
        super(CNNEncoder, self).__init__()

        self.embeddings = embeddings
        input_size = embeddings.embedding_size
        self.linear = nn.Linear(input_size, hidden_size)
        self.cnn = StackedCNN(num_layers, hidden_size,
                              cnn_kernel_width, dropout)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.cnn_kernel_width,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings)

    def forward(self, input, lengths=None, hidden=None):
        """See :class:`onmt.modules.EncoderBase.forward()`"""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        # s_len, batch, emb_dim = emb.size()

        emb = emb.transpose(0, 1).contiguous()
        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        emb_remap = self.linear(emb_reshape)
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1)
        emb_remap = shape_transform(emb_remap)
        out = self.cnn(emb_remap)

        return emb_remap.squeeze(3).transpose(0, 1).contiguous(), \
            out.squeeze(3).transpose(0, 1).contiguous(), lengths

    def update_dropout(self, dropout):
        self.cnn.dropout.p = dropout
