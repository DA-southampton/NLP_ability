# Library

For this example, we will assume that we have run preprocess to
create our datasets. For instance

> onmt_preprocess -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/data -src_vocab_size 10000 -tgt_vocab_size 10000



```python
import torch
import torch.nn as nn

import onmt
import onmt.inputters
import onmt.modules
import onmt.utils
```

We begin by loading in the vocabulary for the model of interest. This will let us check vocab size and to get the special ids for padding.


```python
vocab_fields = torch.load("data/data.vocab.pt")

src_text_field = vocab_fields["src"].base_field
src_vocab = src_text_field.vocab
src_padding = src_vocab.stoi[src_text_field.pad_token]

tgt_text_field = vocab_fields['tgt'].base_field
tgt_vocab = tgt_text_field.vocab
tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]
```

Next we specify the core model itself. Here we will build a small model with an encoder and an attention based input feeding decoder. Both models will be RNNs and the encoder will be bidirectional


```python
emb_size = 100
rnn_size = 500
# Specify the core model.

encoder_embeddings = onmt.modules.Embeddings(emb_size, len(src_vocab),
                                             word_padding_idx=src_padding)

encoder = onmt.encoders.RNNEncoder(hidden_size=rnn_size, num_layers=1,
                                   rnn_type="LSTM", bidirectional=True,
                                   embeddings=encoder_embeddings)

decoder_embeddings = onmt.modules.Embeddings(emb_size, len(tgt_vocab),
                                             word_padding_idx=tgt_padding)
decoder = onmt.decoders.decoder.InputFeedRNNDecoder(
    hidden_size=rnn_size, num_layers=1, bidirectional_encoder=True, 
    rnn_type="LSTM", embeddings=decoder_embeddings)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = onmt.models.model.NMTModel(encoder, decoder)
model.to(device)

# Specify the tgt word generator and loss computation module
model.generator = nn.Sequential(
    nn.Linear(rnn_size, len(tgt_vocab)),
    nn.LogSoftmax(dim=-1)).to(device)

loss = onmt.utils.loss.NMTLossCompute(
    criterion=nn.NLLLoss(ignore_index=tgt_padding, reduction="sum"),
    generator=model.generator)
```

Now we set up the optimizer. Our wrapper around a core torch optim class handles learning rate updates and gradient normalization automatically.


```python
lr = 1
torch_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optim = onmt.utils.optimizers.Optimizer(
    torch_optimizer, learning_rate=lr, max_grad_norm=2)
```

Now we load the data from disk with the associated vocab fields. To iterate through the data itself we use a wrapper around a torchtext iterator class. We specify one for both the training and test data.


```python
# Load some data
from itertools import chain
train_data_file = "data/data.train.0.pt"
valid_data_file = "data/data.valid.0.pt"
train_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[train_data_file],
                                                     fields=vocab_fields,
                                                     batch_size=50,
                                                     batch_size_multiple=1,
                                                     batch_size_fn=None,
                                                     device=device,
                                                     is_train=True,
                                                     repeat=True)

valid_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[valid_data_file],
                                                     fields=vocab_fields,
                                                     batch_size=10,
                                                     batch_size_multiple=1,
                                                     batch_size_fn=None,
                                                     device=device,
                                                     is_train=False,
                                                     repeat=False)
```

Finally we train. Keeping track of the output requires a report manager.


```python
report_manager = onmt.utils.ReportMgr(
    report_every=50, start_time=None, tensorboard_writer=None)
trainer = onmt.Trainer(model=model,
                       train_loss=loss,
                       valid_loss=loss,
                       optim=optim,
                       report_manager=report_manager)
trainer.train(train_iter=train_iter,
              train_steps=400,
              valid_iter=valid_iter,
              valid_steps=200)
```

```
[2019-02-15 16:34:17,475 INFO] Start training loop and validate every 200 steps...
[2019-02-15 16:34:17,601 INFO] Loading dataset from data/data.train.0.pt, number of examples: 10000
[2019-02-15 16:35:43,873 INFO] Step 50/  400; acc:  11.54; ppl: 1714.07; xent: 7.45; lr: 1.00000; 662/656 tok/s;     86 sec
[2019-02-15 16:37:05,965 INFO] Step 100/  400; acc:  13.75; ppl: 534.80; xent: 6.28; lr: 1.00000; 675/671 tok/s;    168 sec
[2019-02-15 16:38:31,289 INFO] Step 150/  400; acc:  15.02; ppl: 439.96; xent: 6.09; lr: 1.00000; 675/668 tok/s;    254 sec
[2019-02-15 16:39:56,715 INFO] Step 200/  400; acc:  16.08; ppl: 357.62; xent: 5.88; lr: 1.00000; 642/647 tok/s;    339 sec
[2019-02-15 16:39:56,811 INFO] Loading dataset from data/data.valid.0.pt, number of examples: 3000
[2019-02-15 16:41:13,415 INFO] Validation perplexity: 208.73
[2019-02-15 16:41:13,415 INFO] Validation accuracy: 23.3507
[2019-02-15 16:41:13,567 INFO] Loading dataset from data/data.train.0.pt, number of examples: 10000
[2019-02-15 16:42:41,562 INFO] Step 250/  400; acc:  17.07; ppl: 310.41; xent: 5.74; lr: 1.00000; 347/344 tok/s;    504 sec
[2019-02-15 16:44:04,899 INFO] Step 300/  400; acc:  19.17; ppl: 262.81; xent: 5.57; lr: 1.00000; 665/661 tok/s;    587 sec
[2019-02-15 16:45:33,653 INFO] Step 350/  400; acc:  19.38; ppl: 244.81; xent: 5.50; lr: 1.00000; 649/642 tok/s;    676 sec
[2019-02-15 16:47:06,141 INFO] Step 400/  400; acc:  20.44; ppl: 214.75; xent: 5.37; lr: 1.00000; 593/598 tok/s;    769 sec
[2019-02-15 16:47:06,265 INFO] Loading dataset from data/data.valid.0.pt, number of examples: 3000
[2019-02-15 16:48:27,328 INFO] Validation perplexity: 150.277
[2019-02-15 16:48:27,328 INFO] Validation accuracy: 24.2132
```

To use the model, we need to load up the translation functions. A Translator object requires the vocab fields, readers for source and target and a global scorer.


```python
import onmt.translate

src_reader = onmt.inputters.str2reader["text"]
tgt_reader = onmt.inputters.str2reader["text"]
scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7, 
                                         beta=0., 
                                         length_penalty="avg", 
                                         coverage_penalty="none")
gpu = 0 if torch.cuda.is_available() else -1
translator = onmt.translate.Translator(model=model, 
                                       fields=vocab_fields, 
                                       src_reader=src_reader, 
                                       tgt_reader=tgt_reader, 
                                       global_scorer=scorer,
                                       gpu=gpu)
builder = onmt.translate.TranslationBuilder(data=torch.load(valid_data_file), 
                                            fields=vocab_fields)


for batch in valid_iter:
    trans_batch = translator.translate_batch(
        batch=batch, src_vocabs=[src_vocab],
        attn_debug=False)
    translations = builder.from_batch(trans_batch)
    for trans in translations:
        print(trans.log(0))
```
```
[2019-02-15 16:48:27,419 INFO] Loading dataset from data/data.valid.0.pt, number of examples: 3000


SENT 0: ['Parliament', 'Does', 'Not', 'Support', 'Amendment', 'Freeing', 'Tymoshenko']
PRED 0: <unk> ist ein <unk> <unk> <unk> .
PRED SCORE: -1.0983


SENT 0: ['Today', ',', 'the', 'Ukraine', 'parliament', 'dismissed', ',', 'within', 'the', 'Code', 'of', 'Criminal', 'Procedure', 'amendment', ',', 'the', 'motion', 'to', 'revoke', 'an', 'article', 'based', 'on', 'which', 'the', 'opposition', 'leader', ',', 'Yulia', 'Tymoshenko', ',', 'was', 'sentenced', '.']
PRED 0: <unk> ist das <unk> <unk> .
PRED SCORE: -1.5950


SENT 0: ['The', 'amendment', 'that', 'would', 'lead', 'to', 'freeing', 'the', 'imprisoned', 'former', 'Prime', 'Minister', 'was', 'revoked', 'during', 'second', 'reading', 'of', 'the', 'proposal', 'for', 'mitigation', 'of', 'sentences', 'for', 'economic', 'offences', '.']
PRED 0: Es gibt es das <unk> der <unk> für <unk> <unk> .
PRED SCORE: -1.5128


SENT 0: ['In', 'October', ',', 'Tymoshenko', 'was', 'sentenced', 'to', 'seven', 'years', 'in', 'prison', 'for', 'entering', 'into', 'what', 'was', 'reported', 'to', 'be', 'a', 'disadvantageous', 'gas', 'deal', 'with', 'Russia', '.']
PRED 0: <unk> ist ein <unk> <unk> .
PRED SCORE: -1.5578


SENT 0: ['The', 'verdict', 'is', 'not', 'yet', 'final;', 'the', 'court', 'will', 'hear', 'Tymoshenko', '&apos;s', 'appeal', 'in', 'December', '.']
PRED 0: <unk> ist nicht <unk> .
PRED SCORE: -0.9623


SENT 0: ['Tymoshenko', 'claims', 'the', 'verdict', 'is', 'a', 'political', 'revenge', 'of', 'the', 'regime;', 'in', 'the', 'West', ',', 'the', 'trial', 'has', 'also', 'evoked', 'suspicion', 'of', 'being', 'biased', '.']
PRED 0: <unk> ist ein <unk> <unk> .
PRED SCORE: -0.8703


SENT 0: ['The', 'proposal', 'to', 'remove', 'Article', '365', 'from', 'the', 'Code', 'of', 'Criminal', 'Procedure', ',', 'upon', 'which', 'the', 'former', 'Prime', 'Minister', 'was', 'sentenced', ',', 'was', 'supported', 'by', '147', 'members', 'of', 'parliament', '.']
PRED 0: <unk> Sie sich mit <unk> <unk> .
PRED SCORE: -1.4778


SENT 0: ['Its', 'ratification', 'would', 'require', '226', 'votes', '.']
PRED 0: <unk> Sie sich <unk> .
PRED SCORE: -1.3341


SENT 0: ['Libya', '&apos;s', 'Victory']
PRED 0: <unk> Sie die <unk> <unk> .
PRED SCORE: -1.5192


SENT 0: ['The', 'story', 'of', 'Libya', '&apos;s', 'liberation', ',', 'or', 'rebellion', ',', 'already', 'has', 'its', 'defeated', '.']
PRED 0: <unk> ist ein <unk> <unk> .
PRED SCORE: -1.2772

...
