"""Audio encoder"""
import math

import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.utils.rnn_factory import rnn_factory
from onmt.encoders.encoder import EncoderBase


class AudioEncoder(EncoderBase):
    """A simple encoder CNN -> RNN for audio input.

    Args:
        rnn_type (str): Type of RNN (e.g. GRU, LSTM, etc).
        enc_layers (int): Number of encoder layers.
        dec_layers (int): Number of decoder layers.
        brnn (bool): Bidirectional encoder.
        enc_rnn_size (int): Size of hidden states of the rnn.
        dec_rnn_size (int): Size of the decoder hidden states.
        enc_pooling (str): A comma separated list either of length 1
            or of length ``enc_layers`` specifying the pooling amount.
        dropout (float): dropout probablity.
        sample_rate (float): input spec
        window_size (int): input spec
    """

    def __init__(self, rnn_type, enc_layers, dec_layers, brnn,
                 enc_rnn_size, dec_rnn_size, enc_pooling, dropout,
                 sample_rate, window_size):
        super(AudioEncoder, self).__init__()
        self.enc_layers = enc_layers
        self.rnn_type = rnn_type
        self.dec_layers = dec_layers
        num_directions = 2 if brnn else 1
        self.num_directions = num_directions
        assert enc_rnn_size % num_directions == 0
        enc_rnn_size_real = enc_rnn_size // num_directions
        assert dec_rnn_size % num_directions == 0
        self.dec_rnn_size = dec_rnn_size
        dec_rnn_size_real = dec_rnn_size // num_directions
        self.dec_rnn_size_real = dec_rnn_size_real
        self.dec_rnn_size = dec_rnn_size
        input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        enc_pooling = enc_pooling.split(',')
        assert len(enc_pooling) == enc_layers or len(enc_pooling) == 1
        if len(enc_pooling) == 1:
            enc_pooling = enc_pooling * enc_layers
        enc_pooling = [int(p) for p in enc_pooling]
        self.enc_pooling = enc_pooling

        if type(dropout) is not list:
            dropout = [dropout]
        if max(dropout) > 0:
            self.dropout = nn.Dropout(dropout[0])
        else:
            self.dropout = None
        self.W = nn.Linear(enc_rnn_size, dec_rnn_size, bias=False)
        self.batchnorm_0 = nn.BatchNorm1d(enc_rnn_size, affine=True)
        self.rnn_0, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=input_size,
                        hidden_size=enc_rnn_size_real,
                        num_layers=1,
                        dropout=dropout[0],
                        bidirectional=brnn)
        self.pool_0 = nn.MaxPool1d(enc_pooling[0])
        for l in range(enc_layers - 1):
            batchnorm = nn.BatchNorm1d(enc_rnn_size, affine=True)
            rnn, _ = \
                rnn_factory(rnn_type,
                            input_size=enc_rnn_size,
                            hidden_size=enc_rnn_size_real,
                            num_layers=1,
                            dropout=dropout[0],
                            bidirectional=brnn)
            setattr(self, 'rnn_%d' % (l + 1), rnn)
            setattr(self, 'pool_%d' % (l + 1),
                    nn.MaxPool1d(enc_pooling[l + 1]))
            setattr(self, 'batchnorm_%d' % (l + 1), batchnorm)

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        """Alternate constructor."""
        if embeddings is not None:
            raise ValueError("Cannot use embeddings with AudioEncoder.")
        return cls(
            opt.rnn_type,
            opt.enc_layers,
            opt.dec_layers,
            opt.brnn,
            opt.enc_rnn_size,
            opt.dec_rnn_size,
            opt.audio_enc_pooling,
            opt.dropout,
            opt.sample_rate,
            opt.window_size)

    def forward(self, src, lengths=None):
        """See :func:`onmt.encoders.encoder.EncoderBase.forward()`"""
        batch_size, _, nfft, t = src.size()
        src = src.transpose(0, 1).transpose(0, 3).contiguous() \
                 .view(t, batch_size, nfft)
        orig_lengths = lengths
        lengths = lengths.view(-1).tolist()

        for l in range(self.enc_layers):
            rnn = getattr(self, 'rnn_%d' % l)
            pool = getattr(self, 'pool_%d' % l)
            batchnorm = getattr(self, 'batchnorm_%d' % l)
            stride = self.enc_pooling[l]
            packed_emb = pack(src, lengths)
            memory_bank, tmp = rnn(packed_emb)
            memory_bank = unpack(memory_bank)[0]
            t, _, _ = memory_bank.size()
            memory_bank = memory_bank.transpose(0, 2)
            memory_bank = pool(memory_bank)
            lengths = [int(math.floor((length - stride) / stride + 1))
                       for length in lengths]
            memory_bank = memory_bank.transpose(0, 2)
            src = memory_bank
            t, _, num_feat = src.size()
            src = batchnorm(src.contiguous().view(-1, num_feat))
            src = src.view(t, -1, num_feat)
            if self.dropout and l + 1 != self.enc_layers:
                src = self.dropout(src)

        memory_bank = memory_bank.contiguous().view(-1, memory_bank.size(2))
        memory_bank = self.W(memory_bank).view(-1, batch_size,
                                               self.dec_rnn_size)

        state = memory_bank.new_full((self.dec_layers * self.num_directions,
                                      batch_size, self.dec_rnn_size_real), 0)
        if self.rnn_type == 'LSTM':
            # The encoder hidden is  (layers*directions) x batch x dim.
            encoder_final = (state, state)
        else:
            encoder_final = state
        return encoder_final, memory_bank, orig_lengths.new_tensor(lengths)

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        for i in range(self.enc_layers - 1):
            getattr(self, 'rnn_%d' % i).dropout = dropout
