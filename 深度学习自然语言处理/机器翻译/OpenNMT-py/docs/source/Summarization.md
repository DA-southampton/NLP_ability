# Summarization

Note: The process and results below are presented in our paper `Bottom-Up Abstractive Summarization`. Please consider citing it if you follow these instructions. 

```
@inproceedings{gehrmann2018bottom,
  title={Bottom-Up Abstractive Summarization},
  author={Gehrmann, Sebastian and Deng, Yuntian and Rush, Alexander},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={4098--4109},
  year={2018}
}
```


This document describes how to replicate summarization experiments on the CNN-DM and gigaword datasets using OpenNMT-py.
In the following, we assume access to a tokenized form of the corpus split into train/valid/test set. You can find the data [here](https://github.com/harvardnlp/sent-summary).

An example article-title pair from Gigaword should look like this:

**Input**
*australia 's current account deficit shrunk by a record #.## billion dollars -lrb- #.## billion us -rrb- in the june quarter due to soaring commodity prices , figures released monday showed .*

**Output**
*australian current account deficit narrows sharply*


### Preprocessing the data

Since we are using copy-attention [1] in the model, we need to preprocess the dataset such that source and target are aligned and use the same dictionary. This is achieved by using the options `dynamic_dict` and `share_vocab`.
We additionally turn off truncation of the source to ensure that inputs longer than 50 words are not truncated.
For CNN-DM we follow See et al. [2] and additionally truncate the source length at 400 tokens and the target at 100. We also note that in CNN-DM, we found models to work better if the target surrounds sentences with tags such that a sentence looks like `<t> w1 w2 w3 . </t>`. If you use this formatting, you can remove the tags after the inference step with the commands `sed -i 's/ <\/t>//g' FILE.txt` and `sed -i 's/<t> //g' FILE.txt`.

**Command used**:

(1) CNN-DM

```bash
onmt_preprocess -train_src data/cnndm/train.txt.src \
                -train_tgt data/cnndm/train.txt.tgt.tagged \
                -valid_src data/cnndm/val.txt.src \
                -valid_tgt data/cnndm/val.txt.tgt.tagged \
                -save_data data/cnndm/CNNDM \
                -src_seq_length 10000 \
                -tgt_seq_length 10000 \
                -src_seq_length_trunc 400 \
                -tgt_seq_length_trunc 100 \
                -dynamic_dict \
                -share_vocab \
                -shard_size 100000
```

(2) Gigaword

```bash
onmt_preprocess -train_src data/giga/train.article.txt \
                -train_tgt data/giga/train.title.txt \
                -valid_src data/giga/valid.article.txt \
                -valid_tgt data/giga/valid.title.txt \
                -save_data data/giga/GIGA \
                -src_seq_length 10000 \
                -dynamic_dict \
                -share_vocab \
                -shard_size 100000
```


### Training

The training procedure described in this section for the most part follows parameter choices and implementation similar to that of See et al. [2]. We describe notable options in the following list:

- `copy_attn`: This is the most important option, since it allows the model to copy words from the source.
- `global_attention mlp`: This makes the model use the  attention mechanism introduced by Bahdanau et al. [3] instead of that by Luong et al. [4] (`global_attention dot`).
- `share_embeddings`: This shares the word embeddings between encoder and decoder. This option drastically decreases the number of parameters a model has to learn. We did not find this option to helpful, but you can try it out by adding it to the command below.
-  `reuse_copy_attn`: This option reuses the standard attention as copy attention. Without this, the model learns an additional attention that is only used for copying.
-  `copy_loss_by_seqlength`: This modifies the loss to divide the loss of a sequence by the number of tokens in it. In practice, we found this to generate longer sequences during inference. However, this effect can also be achieved by using penalties during decoding.
-  `bridge`: This is an additional layer that uses the final hidden state of the encoder as input and computes an initial hidden state for the decoder. Without this, the decoder is initialized with the final hidden state of the encoder directly.
-  `optim adagrad`: Adagrad outperforms SGD when coupled with the following option.
-  `adagrad_accumulator_init 0.1`: PyTorch does not initialize the accumulator in adagrad with any values. To match the optimization algorithm with the Tensorflow version, this option needs to be added.


We are using using a 128-dimensional word-embedding, and 512-dimensional 1 layer LSTM. On the encoder side, we use a bidirectional LSTM (`brnn`), which means that the 512 dimensions are split into 256 dimensions per direction.

We additionally set the maximum norm of the gradient to 2, and renormalize if the gradient norm exceeds this value and do not use any dropout.

**commands used**:

(1) CNN-DM

```bash
onmt_train -save_model models/cnndm \
           -data data/cnndm/CNNDM \
           -copy_attn \
           -global_attention mlp \
           -word_vec_size 128 \
           -rnn_size 512 \
           -layers 1 \
           -encoder_type brnn \
           -train_steps 200000 \
           -max_grad_norm 2 \
           -dropout 0. \
           -batch_size 16 \
           -valid_batch_size 16 \
           -optim adagrad \
           -learning_rate 0.15 \
           -adagrad_accumulator_init 0.1 \
           -reuse_copy_attn \
           -copy_loss_by_seqlength \
           -bridge \
           -seed 777 \
           -world_size 2 \
           -gpu_ranks 0 1
```

(2) CNN-DM Transformer

The following script trains the transformer model on CNN-DM

```bash
onmt_train -data data/cnndm/CNNDM \
           -save_model models/cnndm \
           -layers 4 \
           -rnn_size 512 \
           -word_vec_size 512 \
           -max_grad_norm 0 \
           -optim adam \
           -encoder_type transformer \
           -decoder_type transformer \
           -position_encoding \
           -dropout 0\.2 \
           -param_init 0 \
           -warmup_steps 8000 \
           -learning_rate 2 \
           -decay_method noam \
           -label_smoothing 0.1 \
           -adam_beta2 0.998 \
           -batch_size 4096 \
           -batch_type tokens \
           -normalization tokens \
           -max_generator_batches 2 \
           -train_steps 200000 \
           -accum_count 4 \
           -share_embeddings \
           -copy_attn \
           -param_init_glorot \
           -world_size 2 \
           -gpu_ranks 0 1
```

(3) Gigaword

Gigaword can be trained equivalently. As a baseline, we show a model trained with the following command:

```
onmt_train -data data/giga/GIGA \
           -save_model models/giga \
           -copy_attn \
           -reuse_copy_attn \
           -train_steps 200000
```


### Inference

During inference, we use beam-search with a beam-size of 10. We also added specific penalties that we can use during decoding, described in the following.

- `stepwise_penalty`: Applies penalty at every step
- `coverage_penalty summary`: Uses a penalty that prevents repeated attention to the same source word
- `beta 5`: Parameter for the Coverage Penalty
- `length_penalty wu`: Uses the Length Penalty by Wu et al.
- `alpha 0.8`: Parameter for the Length Penalty.
- `block_ngram_repeat 3`: Prevent the model from repeating trigrams.
- `ignore_when_blocking "." "</t>" "<t>"`: Allow the model to repeat trigrams with the sentence boundary tokens.

**commands used**:

(1) CNN-DM

```
onmt_translate -gpu X \
               -batch_size 20 \
               -beam_size 10 \
               -model models/cnndm... \
               -src data/cnndm/test.txt.src \
               -output testout/cnndm.out \
               -min_length 35 \
               -verbose \
               -stepwise_penalty \
               -coverage_penalty summary \
               -beta 5 \
               -length_penalty wu \
               -alpha 0.9 \
               -verbose \
               -block_ngram_repeat 3 \
               -ignore_when_blocking "." "</t>" "<t>"
```



### Evaluation

#### CNN-DM

To evaluate the ROUGE scores on CNN-DM, we extended the pyrouge wrapper with additional evaluations such as the amount of repeated n-grams (typically found in models with copy attention), found [here](https://github.com/sebastianGehrmann/rouge-baselines). The repository includes a sub-repo called pyrouge. Make sure to clone the code with the `git clone --recurse-submodules https://github.com/sebastianGehrmann/rouge-baselines` command to check this out as well and follow the installation instructions on the pyrouge repository before calling this script.
The installation instructions can be found [here](https://github.com/falcondai/pyrouge/tree/9cdbfbda8b8d96e7c2646ffd048743ddcf417ed9#installation). Note that on MacOS, we found that the pointer to your perl installation in line 1 of `pyrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl` might be different from the one you have installed. A simple fix is to change this line to `#!/usr/local/bin/perl -w` if it fails.

It can be run with the following command:

```
python baseline.py -s testout/cnndm.out -t data/cnndm/test.txt.tgt.tagged -m sent_tag_verbatim -r
```

The `sent_tag_verbatim` option strips `<t>` and `</t>` tags around sentences - when a sentence previously was `<t> w w w w . </t>`, it becomes `w w w w .`.

#### Gigaword

For evaluation of large test sets such as Gigaword, we use the a parallel python wrapper around ROUGE, found [here](https://github.com/pltrdy/files2rouge).

**command used**:
`files2rouge giga.out test.title.txt --verbose`

### Scores and Models

#### CNN-DM

| Model Type    | Model    | R1 R  | R1 P  | R1 F  | R2 R  | R2 P  | R2 F  | RL R  | RL P  | RL F  |
| ------------- |  -------- | -----:| -----:| -----:|------:| -----:| -----:|-----: | -----:| -----:|
| Pointer-Generator + Coverage [2]     | [link](https://github.com/abisee/pointer-generator)  | 39.05 |	43.02 |	39.53 |	17.16 | 18.77 | 17.28  | 35.98 | 39.56 | 36.38 |
| Pointer-Generator [2]  |  [link](https://github.com/abisee/pointer-generator)  | 37.76 | 37.60| 36.44| 16.31| 16.12| 15.66| 34.66| 34.46| 33.42 |
| OpenNMT BRNN  (1 layer, emb 128, hid 512)  |  [link](https://s3.amazonaws.com/opennmt-models/Summary/ada6_bridge_oldcopy_tagged_acc_54.17_ppl_11.17_e20.pt)     | 40.90| 40.20| 	39.02| 	17.91| 	17.99| 	17.25| 	37.76	| 37.18| 	36.05 |
| OpenNMT BRNN  (1 layer, emb 128, hid 512, shared embeddings)  |  [link](https://s3.amazonaws.com/opennmt-models/Summary/ada6_bridge_oldcopy_tagged_share_acc_54.50_ppl_10.89_e20.pt)     | 38.59	| 40.60	| 37.97	| 16.75	| 17.93	| 16.59	| 35.67	| 37.60	| 35.13 |
| OpenNMT BRNN (2 layer, emb 256, hid 1024)   |  [link](https://s3.amazonaws.com/opennmt-models/Summary/ada6_bridge_oldcopy_tagged_larger_acc_54.84_ppl_10.58_e17.pt)     | 40.41	| 40.94 | 39.12 | 17.76 | 18.38 | 17.35 | 37.27 | 37.83 | 36.12 |
| OpenNMT Transformer  |  [link](https://s3.amazonaws.com/opennmt-models/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt)  | 40.31	| 41.09	| 39.25	| 17.97	| 18.46	| 17.54	| 37.41	| 38.18	| 36.45 |


#### Gigaword

| Model Type    | Model    | R1 R  | R1 P  | R1 F  | R2 R  | R2 P  | R2 F  | RL R  | RL P  | RL F  |
| ------------- |  -------- | -----:| -----:| -----:|------:| -----:| -----:|-----: | -----:| -----:|
| OpenNMT, no penalties | [link](https://s3.amazonaws.com/opennmt-models/gigaword_copy_acc_51.78_ppl_11.71_e20.pt)  | ? |	? |	35.51 |	? | ? | 17.35  | ? | ? | 33.17 |



### References

[1] Vinyals, O., Fortunato, M. and Jaitly, N., 2015. Pointer Network. NIPS

[2] See, A., Liu, P.J. and Manning, C.D., 2017. Get To The Point: Summarization with Pointer-Generator Networks. ACL

[3] Bahdanau, D., Cho, K. and Bengio, Y., 2014. Neural machine translation by jointly learning to align and translate. ICLR

[4] Luong, M.T., Pham, H. and Manning, C.D., 2015. Effective approaches to attention-based neural machine translation. EMNLP
