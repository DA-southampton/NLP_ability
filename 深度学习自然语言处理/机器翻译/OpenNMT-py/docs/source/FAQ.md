# FAQ

## How do I use Pretrained embeddings (e.g. GloVe)?

Using vocabularies from OpenNMT-py preprocessing outputs, `embeddings_to_torch.py` to generate encoder and decoder embeddings initialized with GloVe's values.

the script is a slightly modified version of ylhsieh's one2.

Usage:

```shell
embeddings_to_torch.py [-h] [-emb_file_both EMB_FILE_BOTH]
                       [-emb_file_enc EMB_FILE_ENC]
                       [-emb_file_dec EMB_FILE_DEC] -output_file
                       OUTPUT_FILE -dict_file DICT_FILE [-verbose]
                       [-skip_lines SKIP_LINES]
                       [-type {GloVe,word2vec}]
```

Run embeddings_to_torch.py -h for more usagecomplete info.

Example

1) get GloVe files:

```shell
mkdir "glove_dir"
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d "glove_dir"
```

2) prepare data:

```shell
onmt_preprocess \
-train_src data/train.src.txt \
-train_tgt data/train.tgt.txt \
-valid_src data/valid.src.txt \
-valid_tgt data/valid.tgt.txt \
-save_data data/data
```

3) prepare embeddings:

```shell
./tools/embeddings_to_torch.py -emb_file_both "glove_dir/glove.6B.100d.txt" \
-dict_file "data/data.vocab.pt" \
-output_file "data/embeddings"
```

4) train using pre-trained embeddings:

```shell
onmt_train -save_model data/model \
           -batch_size 64 \
           -layers 2 \
           -rnn_size 200 \
           -word_vec_size 100 \
           -pre_word_vecs_enc "data/embeddings.enc.pt" \
           -pre_word_vecs_dec "data/embeddings.dec.pt" \
           -data data/data
```

## How do I use the Transformer model?

The transformer model is very sensitive to hyperparameters. To run it
effectively you need to set a bunch of different options that mimic the Google
setup. We have confirmed the following command can replicate their WMT results.

```shell
python  train.py -data /tmp/de2/data -save_model /tmp/extra \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
        -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 \
        -world_size 4 -gpu_ranks 0 1 2 3
```

Here are what each of the parameters mean:

* `param_init_glorot` `-param_init 0`: correct initialization of parameters
* `position_encoding`: add sinusoidal position encoding to each embedding
* `optim adam`, `decay_method noam`, `warmup_steps 8000`: use special learning rate.
* `batch_type tokens`, `normalization tokens`, `accum_count 4`: batch and normalize based on number of tokens and not sentences. Compute gradients based on four batches.
* `label_smoothing 0.1`: use label smoothing loss.

## Do you support multi-gpu?

First you need to make sure you `export CUDA_VISIBLE_DEVICES=0,1,2,3`.

If you want to use GPU id 1 and 3 of your OS, you will need to `export CUDA_VISIBLE_DEVICES=1,3`

Both `-world_size` and `-gpu_ranks` need to be set. E.g. `-world_size 4 -gpu_ranks 0 1 2 3` will use 4 GPU on this node only.

If you want to use 2 nodes with 2 GPU each, you need to set `-master_ip` and `-master_port`, and

* `-world_size 4 -gpu_ranks 0 1`: on the first node
* `-world_size 4 -gpu_ranks 2 3`: on the second node
* `-accum_count 2`: This will accumulate over 2 batches before updating parameters.

if you use a regular network card (1 Gbps) then we suggest to use a higher `-accum_count` to minimize the inter-node communication.

**Note:**

When training on several GPUs, you can't have them in 'Exclusive' compute mode (`nvidia-smi -c 3`).

The multi-gpu setup relies on a Producer/Consumer setup. This setup means there will be `2<n_gpu> + 1` processes spawned, with 2 processes per GPU, one for model training and one (Consumer) that hosts a `Queue` of batches that will be processed next. The additional process is the Producer, creating batches and sending them to the Consumers. This setup is beneficial for both wall time and memory, since it loads data shards 'in advance', and does not require to load it for each GPU process.

## How can I ensemble Models at inference?

You can specify several models in the translate.py command line: -model model1_seed1 model2_seed2
Bear in mind that your models must share the same target vocabulary.

## How can I weight different corpora at training?

### Preprocessing

We introduced `-train_ids` which is a list of IDs that will be given to the preprocessed shards.

E.g. we have two corpora : `parallel.en` and  `parallel.de` + `from_backtranslation.en` `from_backtranslation.de`, we can pass the following in the `preprocess.py` command:

```shell
...
-train_src parallel.en from_backtranslation.en \
-train_tgt parallel.de from_backtranslation.de \
-train_ids A B \
-save_data my_data \
...
```

and it will dump `my_data.train_A.X.pt` based on `parallel.en`//`parallel.de` and `my_data.train_B.X.pt` based on `from_backtranslation.en`//`from_backtranslation.de`.

### Training

We introduced `-data_ids` based on the same principle as above, as well as `-data_weights`, which is the list of the weight each corpus should have.
E.g.

```shell
...
-data my_data \
-data_ids A B \
-data_weights 1 7 \
...
```

will mean that we'll look for `my_data.train_A.*.pt` and `my_data.train_B.*.pt`, and that when building batches, we'll take 1 example from corpus A, then 7 examples from corpus B, and so on.

**Warning**: This means that we'll load as many shards as we have `-data_ids`, in order to produce batches containing data from every corpus. It may be a good idea to reduce the `-shard_size` at preprocessing.

## Can I get word alignment while translating?

### Raw alignments from averaging Transformer attention heads

Currently, we support producing word alignment while translating for Transformer based models. Using `-report_align` when calling `translate.py` will output the inferred alignments in Pharaoh format. Those alignments are computed from an argmax on the average of the attention heads of the *second to last* decoder layer. The resulting alignment src-tgt (Pharaoh) will be pasted to the translation sentence, separated by ` ||| `.
Note: The *second to last* default behaviour was empirically determined. It is not the same as the paper (they take the *penultimate* layer), probably because of light differences in the architecture.

* alignments use the standard "Pharaoh format", where a pair `i-j` indicates the i<sub>th</sub> word of source language is aligned to j<sub>th</sub> word of target language.
* Example: {'src': 'das stimmt nicht !'; 'output': 'that is not true ! ||| 0-0 0-1 1-2 2-3 1-4 1-5 3-6'}
* Using the`-tgt` option when calling `translate.py`, we output alignments between the source and the gold target rather than the inferred target, assuming we're doing evaluation.
* To convert subword alignments to word alignments, or symetrize bidirectional alignments, please refer to the [lilt scripts](https://github.com/lilt/alignment-scripts).

### Supervised learning on a specific head

The quality of output alignments can be further improved by providing reference alignments while training. This will invoke multi-task learning on translation and alignment. This is an implementation based on the paper [Jointly Learning to Align and Translate with Transformer Models](https://arxiv.org/abs/1909.02074).

The data need to be preprocessed with the reference alignments in order to learn the supervised task.

When calling `preprocess.py`, add:

* `--train_align <path>`: path(s) to the training alignments in Pharaoh format
* `--valid_align <path>`: path to the validation set alignments in Pharaoh format (optional).
The reference alignment file(s) could be generated by [GIZA++](https://github.com/moses-smt/mgiza/) or [fast_align](https://github.com/clab/fast_align).

Note: There should be no blank lines in the alignment files provided.

Options to learn such alignments are:

* `-lambda_align`: set the value > 0.0 to enable joint align training, the paper suggests 0.05;
* `-alignment_layer`: indicate the index of the decoder layer;
* `-alignment_heads`:  number of alignment heads for the alignment task - should be set to 1 for the supervised task, and preferably kept to default (or same as `num_heads`) for the average task;
* `-full_context_alignment`: do full context decoder pass (no future mask) when computing alignments. This will slow down the training (~12% in terms of tok/s) but will be beneficial to generate better alignment.
