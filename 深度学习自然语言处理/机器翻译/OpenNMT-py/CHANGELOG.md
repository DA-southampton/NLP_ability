
**Notes on versioning**


## [Unreleased]
### Fixes and improvements

## [1.0.0.rc1](https://github.com/OpenNMT/OpenNMT-py/tree/1.0.0.rc1) (2019-10-01)
* Fix Apex / FP16 training (Apex new API is buggy)
* Multithread preprocessing way faster (Thanks François Hernandez)
* Pip Installation v1.0.0.rc1 (thanks Paul Tardy)

## [0.9.2](https://github.com/OpenNMT/OpenNMT-py/tree/0.9.2) (2019-09-04)
* Switch to Pytorch 1.2
* Pre/post processing on the translation server
* option to remove the FFN layer in AAN + AAN optimization (faster)
* Coverage loss (per Abisee paper 2017) implementation
* Video Captioning task: Thanks Dylan Flaute!
* Token batch at inference
* Small fixes and add-ons


## [0.9.1](https://github.com/OpenNMT/OpenNMT-py/tree/0.9.1) (2019-06-13)
* New mechanism for MultiGPU training "1 batch producer / multi batch consumers"
  resulting in big memory saving when handling huge datasets
* New APEX AMP (mixed precision) API
* Option to overwrite shards when preprocessing
* Small fixes and add-ons

## [0.9.0](https://github.com/OpenNMT/OpenNMT-py/tree/0.9.0) (2019-05-16)
* Faster vocab building when processing shards (no reloading)
* New dataweighting feature
* New dropout scheduler.
* Small fixes and add-ons

## [0.8.2](https://github.com/OpenNMT/OpenNMT-py/tree/0.8.2) (2019-02-16)
* Update documentation and Library example
* Revamp args
* Bug fixes, save moving average in FP32
* Allow FP32 inference for FP16 models

## [0.8.1](https://github.com/OpenNMT/OpenNMT-py/tree/0.8.1) (2019-02-12)
* Update documentation
* Random sampling scores fixes
* Bug fixes

## [0.8.0](https://github.com/OpenNMT/OpenNMT-py/tree/0.8.0) (2019-02-09)
* Many fixes and code cleaning thanks @flauted, @guillaumekln
* Datasets code refactor (thanks @flauted) you need to r-preeprocess datasets

### New features
* FP16 Support: Experimental, using Apex, Checkpoints may break in future version.
* Continuous exponential moving average (thanks @francoishernandez, and Marian)
* Relative positions encoding (thanks @francoishernanndez, and Google T2T)
* Deprecate the old beam search, fast batched beam search supports all options


## [0.7.2](https://github.com/OpenNMT/OpenNMT-py/tree/0.7.2) (2019-01-31)
* Many fixes and code cleaning thanks @bpopeters, @flauted, @guillaumekln

### New features
* Multilevel fields for better handling of text featuer embeddinggs. 


## [0.7.1](https://github.com/OpenNMT/OpenNMT-py/tree/0.7.1) (2019-01-24)
* Many fixes and code refactoring thanks @bpopeters, @flauted, @guillaumekln

### New features
* Random sampling thanks @daphnei
* Enable sharding for huge files at translation

## [0.7.0](https://github.com/OpenNMT/OpenNMT-py/tree/0.7.0) (2019-01-02)
* Many fixes and code refactoring thanks @benopeters
* Migrated to Pytorch 1.0

## [0.6.0](https://github.com/OpenNMT/OpenNMT-py/tree/0.6.0) (2018-11-28)
* Many fixes and code improvements
* New: Ability to load a yml config file. See examples in config folder.

## [0.5.0](https://github.com/OpenNMT/OpenNMT-py/tree/0.5.0) (2018-10-24)
* Fixed advance n_best beam in translate_batch_fast
* Fixed remove valid set vocab from total vocab
* New: Ability to reset optimizer when using train_from
* New: create_vocabulary tool + fix when loading existing vocab.

## [0.4.1](https://github.com/OpenNMT/OpenNMT-py/tree/0.4.1) (2018-10-11)
* Fixed preprocessing files names, cleaning intermediary files.

## [0.4.0](https://github.com/OpenNMT/OpenNMT-py/tree/0.4.0) (2018-10-08)
* Fixed Speech2Text training (thanks Yuntian)

* Removed -max_shard_size, replaced by -shard_size = number of examples in a shard.
  Default value = 1M which works fine in most Text dataset cases. (will avoid Ram OOM in most cases)


## [0.3.0](https://github.com/OpenNMT/OpenNMT-py/tree/0.3.0) (2018-09-27)
* Now requires Pytorch 0.4.1

* Multi-node Multi-GPU with Torch Distributed

  New options are:
  -master_ip: ip address of the master node
  -master_port: port number of th emaster node
  -world_size = total number of processes to be run (total GPUs accross all nodes)
  -gpu_ranks = list of indices of processes accross all nodes

* gpuid is deprecated
See examples in https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/FAQ.md

* Fixes to img2text now working

* New sharding based on number of examples

* Fixes to avoid 0.4.1 deprecated functions.


## [0.2.1](https://github.com/OpenNMT/OpenNMT-py/tree/0.2.1) (2018-08-31)

### Fixes and improvements

* First compatibility steps with Pytorch 0.4.1 (non breaking)
* Fix TranslationServer (when various request try to load the same model at the same time)
* Fix StopIteration error (python 3.7)

### New features
* Ensemble at inference (thanks @Waino)

## [0.2](https://github.com/OpenNMT/OpenNMT-py/tree/v0.2) (2018-08-28)

### improvements

* Compatibility fixes with Pytorch 0.4 / Torchtext 0.3
* Multi-GPU based on Torch Distributed
* Average Attention Network (AAN) for the Transformer (thanks @francoishernandez )
* New fast beam search (see -fast in translate.py) (thanks @guillaumekln)
* Sparse attention / sparsemax (thanks to @bpopeters)
* Refactoring of many parts of the code base:
 - change from -epoch to -train_steps -valid_steps (see opts.py)
 - reorg of the logic train => train_multi / train_single => trainer
* Many fixes / improvements in the translationserver (thanks @pltrdy @francoishernandez)
* fix BPTT

## [0.1](https://github.com/OpenNMT/OpenNMT-py/tree/v0.1) (2018-06-08)

### First and Last Release using Pytorch 0.3.x


