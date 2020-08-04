"""  Attention and normalization modules  """
from onmt.modules.util_class import Elementwise
from onmt.modules.gate import context_gate_factory, ContextGate
from onmt.modules.global_attention import GlobalAttention
from onmt.modules.conv_multi_step_attention import ConvMultiStepAttention
from onmt.modules.copy_generator import CopyGenerator, CopyGeneratorLoss, \
    CopyGeneratorLossCompute
from onmt.modules.multi_headed_attn import MultiHeadedAttention
from onmt.modules.embeddings import Embeddings, PositionalEncoding, \
    VecEmbedding
from onmt.modules.weight_norm import WeightNormConv2d
from onmt.modules.average_attn import AverageAttention

__all__ = ["Elementwise", "context_gate_factory", "ContextGate",
           "GlobalAttention", "ConvMultiStepAttention", "CopyGenerator",
           "CopyGeneratorLoss", "CopyGeneratorLossCompute",
           "MultiHeadedAttention", "Embeddings", "PositionalEncoding",
           "WeightNormConv2d", "AverageAttention", "VecEmbedding"]
