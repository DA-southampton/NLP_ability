
# Translation

The example below uses the Moses tokenizer (http://www.statmt.org/moses/) to prepare the data and the moses BLEU script for evaluation. This example if for training for the WMT'16 Multimodal Translation task (http://www.statmt.org/wmt16/multimodal-task.html).

Step 0. Download the data.

```bash
mkdir -p data/multi30k
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/multi30k && rm training.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/multi30k && rm validation.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_test2016.tar.gz && tar -xf mmt_task1_test2016.tar.gz -C data/multi30k && rm mmt_task1_test2016.tar.gz
```

Step 1. Preprocess the data.

```bash
for l in en de; do for f in data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
for l in en de; do for f in data/multi30k/*.$l; do perl tools/tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done
onmt_preprocess -train_src data/multi30k/train.en.atok -train_tgt data/multi30k/train.de.atok -valid_src data/multi30k/val.en.atok -valid_tgt data/multi30k/val.de.atok -save_data data/multi30k.atok.low -lower
```

Step 2. Train the model.

```bash
onmt_train -data data/multi30k.atok.low -save_model multi30k_model -gpu_ranks 0
```

Step 3. Translate sentences.

```bash
onmt_translate -gpu 0 -model multi30k_model_*_e13.pt -src data/multi30k/test2016.en.atok -tgt data/multi30k/test2016.de.atok -replace_unk -verbose -output multi30k.test.pred.atok
```

And evaluate

```bash
perl tools/multi-bleu.perl data/multi30k/test2016.de.atok < multi30k.test.pred.atok
```
