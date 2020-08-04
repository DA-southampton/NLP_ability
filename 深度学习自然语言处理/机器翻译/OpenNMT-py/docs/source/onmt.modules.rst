Modules
=============

Core Modules
------------

.. autoclass:: onmt.modules.Embeddings
    :members:


Encoders
---------

.. autoclass:: onmt.encoders.EncoderBase
    :members:

.. autoclass:: onmt.encoders.MeanEncoder
    :members:

.. autoclass:: onmt.encoders.RNNEncoder
    :members:


Decoders
---------


.. autoclass:: onmt.decoders.DecoderBase
    :members:
    
.. autoclass:: onmt.decoders.decoder.RNNDecoderBase
    :members:

.. autoclass:: onmt.decoders.StdRNNDecoder
    :members:

.. autoclass:: onmt.decoders.InputFeedRNNDecoder
    :members:

Attention
----------

.. autoclass:: onmt.modules.AverageAttention
    :members:

.. autoclass:: onmt.modules.GlobalAttention
    :members:



Architecture: Transformer
----------------------------

.. autoclass:: onmt.modules.PositionalEncoding
    :members:

.. autoclass:: onmt.modules.position_ffn.PositionwiseFeedForward
    :members:

.. autoclass:: onmt.encoders.TransformerEncoder
    :members:

.. autoclass:: onmt.decoders.TransformerDecoder
    :members:

.. autoclass:: onmt.modules.MultiHeadedAttention
    :members:
    :undoc-members:


Architecture: Conv2Conv
----------------------------

(These methods are from a user contribution
and have not been thoroughly tested.)


.. autoclass:: onmt.encoders.CNNEncoder
    :members:


.. autoclass:: onmt.decoders.CNNDecoder
    :members:

.. autoclass:: onmt.modules.ConvMultiStepAttention
    :members:

.. autoclass:: onmt.modules.WeightNormConv2d
    :members:

Architecture: SRU
----------------------------

.. autoclass:: onmt.models.sru.SRU
    :members:


Alternative Encoders
--------------------

onmt\.modules\.AudioEncoder

.. autoclass:: onmt.encoders.AudioEncoder
    :members:


onmt\.modules\.ImageEncoder

.. autoclass:: onmt.encoders.ImageEncoder
    :members:


Copy Attention
--------------

.. autoclass:: onmt.modules.CopyGenerator
    :members:


Structured Attention
-------------------------------------------

.. autoclass:: onmt.modules.structured_attention.MatrixTree
    :members:
