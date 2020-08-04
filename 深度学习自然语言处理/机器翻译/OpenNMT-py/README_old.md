# OpenNMT-py: Open-Source Neural Machine Translation

[![Build Status](https://travis-ci.org/OpenNMT/OpenNMT-py.svg?branch=master)](https://travis-ci.org/OpenNMT/OpenNMT-py)
[![Run on FH](https://img.shields.io/badge/Run%20on-FloydHub-blue.svg)](https://floydhub.com/run?template=https://github.com/OpenNMT/OpenNMT-py)

This is a [Pytorch](https://github.com/pytorch/pytorch)
port of [OpenNMT](https://github.com/OpenNMT/OpenNMT),
an open-source (MIT) neural machine translation system. It is designed to be research friendly to try out new ideas in translation, summary, image-to-text, morphology, and many other domains. Some companies have proven the code to be production ready.

We love contributions. Please consult the Issues page for any [Contributions Welcome](https://github.com/OpenNMT/OpenNMT-py/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22) tagged post. 

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>

Before raising an issue, make sure you read the requirements and the documentation examples.

Unless there is a bug, please use the [Forum](http://forum.opennmt.net) or [Gitter](https://gitter.im/OpenNMT/OpenNMT-py) to ask questions.


Table of Contents
=================
  * [Full Documentation](http://opennmt.net/OpenNMT-py/)
  * [Requirements](#requirements)
  * [Features](#features)
  * [Quickstart](#quickstart)
  * [Run on FloydHub](#run-on-floydhub)
  * [Acknowledgements](#acknowledgements)
  * [Citation](#citation)

## Requirements

Install `OpenNMT-py` from `pip`:
```bash
pip install OpenNMT-py
```

or from the sources:
```bash
git clone https://github.com/OpenNMT/OpenNMT-py.git
cd OpenNMT-py
python setup.py install
```

Note: If you have MemoryError in the install try to use `pip` with `--no-cache-dir`.

*(Optionnal)* some advanced features (e.g. working audio, image or pretrained models) requires extra packages, you can install it with:
```bash
pip install -r requirements.opt.txt
```

Note:

- some features require Python 3.5 and after (eg: Distributed multigpu, entmax)
- we currently only support PyTorch 1.2 (should work with 1.1)

## Features

- [Seq2Seq models (encoder-decoder) with multiple RNN cells (lstm/gru) and attention (dotprod/mlp) types](http://opennmt.net/OpenNMT-py/options/train.html#model-encoder-decoder)
- [Transformer models](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-the-transformer-model)
- [Copy and Coverage Attention](http://opennmt.net/OpenNMT-py/options/train.html#model-attention)
- [Pretrained Embeddings](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-pretrained-embeddings-e-g-glove)
- [Source word features](http://opennmt.net/OpenNMT-py/options/train.html#model-embeddings)
- [Image-to-text processing](http://opennmt.net/OpenNMT-py/im2text.html)
- [Speech-to-text processing](http://opennmt.net/OpenNMT-py/speech2text.html)
- [TensorBoard logging](http://opennmt.net/OpenNMT-py/options/train.html#logging)
- [Multi-GPU training](http://opennmt.net/OpenNMT-py/FAQ.html##do-you-support-multi-gpu)
- [Data preprocessing](http://opennmt.net/OpenNMT-py/options/preprocess.html)
- [Inference (translation) with batching and beam search](http://opennmt.net/OpenNMT-py/options/translate.html)
- Inference time loss functions.
- [Conv2Conv convolution model]
- SRU "RNNs faster than CNN" paper
- Mixed-precision training with [APEX](https://github.com/NVIDIA/apex), optimized on [Tensor Cores](https://developer.nvidia.com/tensor-cores)

## Quickstart

[Full Documentation](http://opennmt.net/OpenNMT-py/)


### Step 1: Preprocess the data

```bash
onmt_preprocess -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
```

We will be working with some example data in `data/` folder.

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are required and used to evaluate the convergence of the training. It usually contains no more than 5000 sentences.


After running the preprocessing, the following files are generated:

* `demo.train.pt`: serialized PyTorch file containing training data
* `demo.valid.pt`: serialized PyTorch file containing validation data
* `demo.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

### Step 2: Train the model

```bash
onmt_train -data data/demo -save_model demo-model
```

The main train command is quite simple. Minimally it takes a data file
and a save file.  This will run the default model, which consists of a
2-layer LSTM with 500 hidden units on both the encoder/decoder.
If you want to train on GPU, you need to set, as an example:
CUDA_VISIBLE_DEVICES=1,3
`-world_size 2 -gpu_ranks 0 1` to use (say) GPU 1 and 3 on this node only.
To know more about distributed training on single or multi nodes, read the FAQ section.

### Step 3: Translate

```bash
onmt_translate -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose
```

Now you have a model which you can use to predict on new data. We do this by running beam search. This will output predictions into `pred.txt`.

!!! note "Note"
    The predictions are going to be quite terrible, as the demo dataset is small. Try running on some larger datasets! For example you can download millions of parallel sentences for [translation](http://www.statmt.org/wmt16/translation-task.html) or [summarization](https://github.com/harvardnlp/sent-summary).

## Alternative: Run on FloydHub

[![Run on FloydHub](https://static.floydhub.com/button/button.svg)](https://floydhub.com/run?template=https://github.com/OpenNMT/OpenNMT-py)

Click this button to open a Workspace on [FloydHub](https://www.floydhub.com/?utm_medium=readme&utm_source=opennmt-py&utm_campaign=jul_2018) for training/testing your code.


## Pretrained embeddings (e.g. GloVe)

Please see the FAQ: [How to use GloVe pre-trained embeddings in OpenNMT-py](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-pretrained-embeddings-e-g-glove)

## Pretrained Models

The following pretrained models can be downloaded and used with translate.py.

http://opennmt.net/Models-py/

## Acknowledgements

OpenNMT-py is run as a collaborative open-source project.
The original code was written by [Adam Lerer](http://github.com/adamlerer) (NYC) to reproduce OpenNMT-Lua using Pytorch.

Major contributors are:
[Sasha Rush](https://github.com/srush) (Cambridge, MA)
[Vincent Nguyen](https://github.com/vince62s) (Ubiqus)
[Ben Peters](http://github.com/bpopeters) (Lisbon)
[Sebastian Gehrmann](https://github.com/sebastianGehrmann) (Harvard NLP)
[Yuntian Deng](https://github.com/da03) (Harvard NLP)
[Guillaume Klein](https://github.com/guillaumekln) (Systran)
[Paul Tardy](https://github.com/pltrdy) (Ubiqus / Lium)
[François Hernandez](https://github.com/francoishernandez) (Ubiqus)
[Jianyu Zhan](http://github.com/jianyuzhan) (Shanghai)
[Dylan Flaute](http://github.com/flauted (University of Dayton)
and more !

OpentNMT-py belongs to the OpenNMT project along with OpenNMT-Lua and OpenNMT-tf.

## Citation

[OpenNMT: Neural Machine Translation Toolkit](https://arxiv.org/pdf/1805.11462)

[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {Open{NMT}: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```
