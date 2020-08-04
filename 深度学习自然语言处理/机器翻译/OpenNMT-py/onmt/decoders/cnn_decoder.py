"""Implementation of the CNN Decoder part of
"Convolutional Sequence to Sequence Learning"
"""
import torch
import torch.nn as nn

from onmt.modules import ConvMultiStepAttention, GlobalAttention
from onmt.utils.cnn_factory import shape_transform, GatedConv
from onmt.decoders.decoder import DecoderBase

SCALE_WEIGHT = 0.5 ** 0.5


class CNNDecoder(DecoderBase):
    """Decoder based on "Convolutional Sequence to Sequence Learning"
    :cite:`DBLP:journals/corr/GehringAGYD17`.

    Consists of residual convolutional layers, with ConvMultiStepAttention.
    """

    def __init__(self, num_layers, hidden_size, attn_type,
                 copy_attn, cnn_kernel_width, dropout, embeddings,
                 copy_attn_type):
        super(CNNDecoder, self).__init__()

        self.cnn_kernel_width = cnn_kernel_width
        self.embeddings = embeddings

        # Decoder State
        self.state = {}

        input_size = self.embeddings.embedding_size
        self.linear = nn.Linear(input_size, hidden_size)
        self.conv_layers = nn.ModuleList(
            [GatedConv(hidden_size, cnn_kernel_width, dropout, True)
             for i in range(num_layers)]
        )
        self.attn_layers = nn.ModuleList(
            [ConvMultiStepAttention(hidden_size) for i in range(num_layers)]
        )

        # CNNDecoder has its own attention mechanism.
        # Set up a separate copy attention layer if needed.
        assert not copy_attn, "Copy mechanism not yet tested in conv2conv"
        if copy_attn:
            self.copy_attn = GlobalAttention(
                hidden_size, attn_type=copy_attn_type)
        else:
            self.copy_attn = None

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.global_attention,
            opt.copy_attn,
            opt.cnn_kernel_width,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.copy_attn_type)

    def init_state(self, _, memory_bank, enc_hidden):
        """Init decoder state."""
        self.state["src"] = (memory_bank + enc_hidden) * SCALE_WEIGHT
        self.state["previous_input"] = None

    def map_state(self, fn):
        self.state["src"] = fn(self.state["src"], 1)
        if self.state["previous_input"] is not None:
            self.state["previous_input"] = fn(self.state["previous_input"], 1)

    def detach_state(self):
        self.state["previous_input"] = self.state["previous_input"].detach()

    def forward(self, tgt, memory_bank, step=None, **kwargs):
        """ See :obj:`onmt.modules.RNNDecoderBase.forward()`"""

        if self.state["previous_input"] is not None:
            tgt = torch.cat([self.state["previous_input"], tgt], 0)

        dec_outs = []
        attns = {"std": []}
        if self.copy_attn is not None:
            attns["copy"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        tgt_emb = emb.transpose(0, 1).contiguous()
        # The output of CNNEncoder.
        src_memory_bank_t = memory_bank.transpose(0, 1).contiguous()
        # The combination of output of CNNEncoder and source embeddings.
        src_memory_bank_c = self.state["src"].transpose(0, 1).contiguous()

        emb_reshape = tgt_emb.contiguous().view(
            tgt_emb.size(0) * tgt_emb.size(1), -1)
        linear_out = self.linear(emb_reshape)
        x = linear_out.view(tgt_emb.size(0), tgt_emb.size(1), -1)
        x = shape_transform(x)

        pad = torch.zeros(x.size(0), x.size(1), self.cnn_kernel_width - 1, 1)

        pad = pad.type_as(x)
        base_target_emb = x

        for conv, attention in zip(self.conv_layers, self.attn_layers):
            new_target_input = torch.cat([pad, x], 2)
            out = conv(new_target_input)
            c, attn = attention(base_target_emb, out,
                                src_memory_bank_t, src_memory_bank_c)
            x = (x + (c + out) * SCALE_WEIGHT) * SCALE_WEIGHT
        output = x.squeeze(3).transpose(1, 2)

        # Process the result and update the attentions.
        dec_outs = output.transpose(0, 1).contiguous()
        if self.state["previous_input"] is not None:
            dec_outs = dec_outs[self.state["previous_input"].size(0):]
            attn = attn[:, self.state["previous_input"].size(0):].squeeze()
            attn = torch.stack([attn])
        attns["std"] = attn
        if self.copy_attn is not None:
            attns["copy"] = attn

        # Update the state.
        self.state["previous_input"] = tgt
        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def update_dropout(self, dropout):
        for layer in self.conv_layers:
            layer.dropout.p = dropout
