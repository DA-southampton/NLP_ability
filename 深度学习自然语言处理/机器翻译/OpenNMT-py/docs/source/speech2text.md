# Speech to Text

A deep learning-based approach to learning the speech-to-text conversion, built on top of the <a href="http://opennmt.net/">OpenNMT</a> system.

Given raw audio, we first apply short-time Fourier transform (STFT), then apply Convolutional Neural Networks to get the source features. Based on this source representation, we use an LSTM decoder with attention to produce the text character by character.

### Dependencies

* `torchaudio`: `sudo apt-get install -y sox libsox-dev libsox-fmt-all; pip install git+https://github.com/pytorch/audio`
* `librosa`: `pip install librosa`

### Quick Start

To get started, we provide a toy speech-to-text example. We assume that the working directory is `OpenNMT-py` throughout this document.

0) Download the data.

```
wget -O data/speech.tgz http://lstm.seas.harvard.edu/latex/speech.tgz; tar zxf data/speech.tgz -C data/
```


1) Preprocess the data.

```
onmt_preprocess -data_type audio -src_dir data/speech/an4_dataset -train_src data/speech/src-train.txt -train_tgt data/speech/tgt-train.txt -valid_src data/speech/src-val.txt -valid_tgt data/speech/tgt-val.txt -shard_size 300 -save_data data/speech/demo
```

2) Train the model.

```
onmt_train -model_type audio -enc_rnn_size 512 -dec_rnn_size 512 -audio_enc_pooling 1,1,2,2 -dropout 0 -enc_layers 4 -dec_layers 1 -rnn_type LSTM -data data/speech/demo -save_model demo-model -global_attention mlp -gpu_ranks 0 -batch_size 8 -optim adam -max_grad_norm 100 -learning_rate 0.0003 -learning_rate_decay 0.8 -train_steps 100000
```

3) Translate the speechs.

```
onmt_translate -data_type audio -model demo-model_acc_x_ppl_x_e13.pt -src_dir data/speech/an4_dataset -src data/speech/src-val.txt -output pred.txt -gpu 0 -verbose
```


### Options

* `-src_dir`: The directory containing the audio files.

* `-train_tgt`: The file storing the tokenized labels, one label per line. It shall look like:
```
<label0_token0> <label0_token1> ... <label0_tokenN0>
<label1_token0> <label1_token1> ... <label1_tokenN1>
<label2_token0> <label2_token1> ... <label2_tokenN2>
...
```

* `-train_src`: The file storing the paths of the audio files (relative to `src_dir`).
```
<speech0_path>
<speech1_path>
<speech2_path>
...
```

* `sample_rate`: Sample rate. Default: 16000.
* `window_size`: Window size for spectrogram in seconds. Default: 0.02.
* `window_stride`: Window stride for spectrogram in seconds. Default: 0.01.
* `window`: Window type for spectrogram generation. Default: hamming.

### Acknowledgement

Our preprocessing and CNN encoder is adapted from [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch).
