Video to Text
=============

Recurrent
---------

This tutorial shows how to replicate the results from
`"Describing Videos by Exploiting Temporal Structure" <https://arxiv.org/pdf/1502.08029.pdf>`_
[`code <https://github.com/yaoli/arctic-capgen-vid>`_]
using OpenNMT-py.

Get `YouTubeClips.tar` from `here <http://www.cs.utexas.edu/users/ml/clamp/videoDescription/>`_.
Use ``tar -xvf YouTubeClips.tar`` to decompress the archive.

Now, visit `this repo <https://github.com/yaoli/arctic-capgen-vid>`_.
Follow the "preprocessed YouTube2Text download link."
We'll be throwing away the Googlenet features. We just need the captions.
Use ``unzip youtube2text_iccv15.zip`` to decompress the files.

Get to the following directory structure: ::

    yt2t
    |-YouTubeClips
    |-youtube2text_iccv15

Change directories to `yt2t`. We'll rename the videos to follow the "vid#.avi" format:

.. code-block:: python

    import pickle
    import os


    YT = "youtube2text_iccv15"
    YTC = "YouTubeClips"

    # load the YouTube hash -> vid### map.
    with open(os.path.join(YT, "dict_youtube_mapping.pkl"), "rb") as f:
        yt2vid = pickle.load(f, encoding="latin-1")

    for f in os.listdir(YTC):
        hashy, ext = os.path.splitext(f)
        vid = yt2vid[hashy]
        fpath_old = os.path.join(YTC, f)
        f_new = vid + ext
        fpath_new = os.path.join(YTC, f_new)
        os.rename(fpath_old, fpath_new)

Make sure all the videos have the same (low) framerate by changing to the YouTubeClips directory and using

.. code-block:: bash

    for fi in $( ls ); do ffmpeg -y -i $fi -r 2 $fi; done

Now we want to convert the frames into sequences of CNN feature vectors.
(We'll use the environment variable ``Y2T2`` to refer to the `yt2t` directory, so change directories back and use)

.. code-block:: bash

    export YT2T=`pwd`

Then change directories back to the `OpenNMT-py` directory.
Use `tools/img_feature_extractor.py`.
Set the ``--world_size`` argument to the number of GPUs you have available
(You can use the environment variable ``CUDA_VISIBLE_DEVICES`` to restrict the GPUs used).

.. code-block:: bash

    PYTHONPATH=$PWD:$PYTHONPATH python tools/vid_feature_extractor.py --root_dir $YT2T/YouTubeClips --out_dir $YT2T/r152

Ensure the count is equal to 1970.
You can use ``ls -1 $YT2T/r152 | wc -l``.
If not, rerun the script. It will only process on the missing feature vectors.
(Note this is unexpected behavior and consider opening an issue.)

Now we turn our attention to the annotations. Each video has multiple associated captions. We want to
train the model on each video + single caption pair. We'll collect all the captions per video, then we'll
flatten them into files listing the feature vector sequence filenames (repeating for each caption) and the
annotations. We skip the test videos since they are handled separately at translation time.

Change directories back to ``YT2T``:

.. code-block:: bash

    cd $YT2T

.. code-block:: python

    import pickle
    import os
    from random import shuffle


    YT = "youtube2text_iccv15"
    SHUFFLE = True

    with open(os.path.join(YT, "CAP.pkl"), "rb") as f:
        ann = pickle.load(f, encoding="latin-1")

    vid2anns = {}
    for vid_name, data in ann.items():
        for d in data:
            try:
                vid2anns[vid_name].append(d["tokenized"])
            except KeyError:
                vid2anns[vid_name] = [d["tokenized"]]

    with open(os.path.join(YT, "train.pkl"), "rb") as f:
        train = pickle.load(f, encoding="latin-1")

    with open(os.path.join(YT, "valid.pkl"), "rb") as f:
        val = pickle.load(f, encoding="latin-1")

    with open(os.path.join(YT, "test.pkl"), "rb") as f:
        test = pickle.load(f, encoding="latin-1")

    train_files = open("yt2t_train_files.txt", "w")
    val_files = open("yt2t_val_files.txt", "w")
    val_folded = open("yt2t_val_folded_files.txt", "w")
    test_files = open("yt2t_test_files.txt", "w")

    train_cap = open("yt2t_train_cap.txt", "w")
    val_cap = open("yt2t_val_cap.txt", "w")

    vid_names = vid2anns.keys()
    if SHUFFLE:
        vid_names = list(vid_names)
        shuffle(vid_names)


    for vid_name in vid_names:
        anns = vid2anns[vid_name]
        vid_path = vid_name + ".npy"
        for i, an in enumerate(anns):
            an = an.replace("\n", " ")  # some caps have newlines
            split_name = vid_name + "_" + str(i)
            if split_name in train:
                train_files.write(vid_path + "\n")
                train_cap.write(an + "\n")
            elif split_name in val:
                if i == 0:
                    val_folded.write(vid_path + "\n")
                val_files.write(vid_path + "\n")
                val_cap.write(an + "\n")
            else:
                # Don't need to save out the test captions,
                # just the files. And, don't need to repeat
                # it for each caption
                assert split_name in test
                if i == 0:
                    test_files.write(vid_path + "\n")

Return to the `OpenNMT-py` directory. Now we preprocess the data for training.
We preprocess with a small shard size of 1000. This keeps the amount of data in memory (RAM) to a
manageable 10 G. If you have more RAM, you can increase the shard size.

Preprocess the data with

.. code-block:: bash

    onmt_preprocess -data_type vec -train_src $YT2T/yt2t_train_files.txt -src_dir $YT2T/r152/ -train_tgt $YT2T/yt2t_train_cap.txt -valid_src $YT2T/yt2t_val_files.txt -valid_tgt $YT2T/yt2t_val_cap.txt -save_data data/yt2t --shard_size 1000

Train with

.. code-block:: bash

    onmt_train -data data/yt2t -save_model yt2t-model -world_size 2 -gpu_ranks 0 1 -model_type vec -batch_size 64 -train_steps 10000 -valid_steps 500 -save_checkpoint_steps 500 -encoder_type brnn -optim adam -learning_rate .0001 -feat_vec_size 2048

Translate with

.. code-block::

    onmt_translate -model yt2t-model_step_7200.pt -src $YT2T/yt2t_test_files.txt -output pred.txt -verbose -data_type vec -src_dir $YT2T/r152 -gpu 0 -batch_size 10

.. note::

    Generally, you want to keep the model that has the lowest validation perplexity. That turned out to be
    at step 7200, but choosing a different validation frequency or random seed could result in different results.


Then you can use `coco-caption <https://github.com/tylin/coco-caption/tree/master/pycocoevalcap>`_ to evaluate the predictions.
(Note that the fork `flauted <https://github.com/flauted/coco-caption>`_ can be used for Python 3 compatibility).
Install the git repository with pip using


.. code-block:: bash

    pip install git+<clone URL>

Then use the following Python code to evaluate:

.. code-block:: python

    import os
    from pprint import pprint
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice


    if __name__ == "__main__":
        pred = open("pred.txt")

        import pickle
        import os

        YT = os.path.join(os.environ["YT2T"], "youtube2text_iccv15")

        with open(os.path.join(YT, "CAP.pkl"), "rb") as f:
            ann = pickle.load(f, encoding="latin-1")

        vid2anns = {}
        for vid_name, data in ann.items():
            for d in data:
                try:
                    vid2anns[vid_name].append(d["tokenized"])
                except KeyError:
                    vid2anns[vid_name] = [d["tokenized"]]

        test_files = open(os.path.join(os.environ["YT2T"], "yt2t_test_files.txt"))

        scorers = {
            "Bleu": Bleu(4),
            "Meteor": Meteor(),
            "Rouge": Rouge(),
            "Cider": Cider(),
            "Spice": Spice()
        }

        gts = {}
        res = {}
        for outp, filename in zip(pred, test_files):
            filename = filename.strip("\n")
            outp = outp.strip("\n")
            vid_id = os.path.splitext(filename)[0]
            anns = vid2anns[vid_id]
            gts[vid_id] = anns
            res[vid_id] = [outp]

        scores = {}
        for name, scorer in scorers.items():
            score, all_scores = scorer.compute_score(gts, res)
            if isinstance(score, list):
                for i, sc in enumerate(score, 1):
                    scores[name + str(i)] = sc
            else:
                scores[name] = score
        pprint(scores)

Here are our results ::

    {'Bleu1': 0.7888553878084233,
     'Bleu2': 0.6729376621109295,
     'Bleu3': 0.5778428507344473,
     'Bleu4': 0.47633625833397897,
     'Cider': 0.7122415518428051,
     'Meteor': 0.31829562714082704,
     'Rouge': 0.6811305229481235,
     'Spice': 0.044147089472463576}


So how does this stack up against the paper? These results should be compared to the "Global (Temporal Attention)"
row in Table 1. The authors report BLEU4 0.4028, METEOR 0.2900, and CIDEr 0.4801. So, our results are a significant
improvement. Our architecture follows the general encoder + attentional decoder described in the paper, but the
actual attention implementation is slightly different. The paper downsamples by choosing 26 equally spaced frames from
the first 240, while we downsample the video to 2 fps. Also, we use ResNet features instead of GoogLeNet, and we
lowercase while the paper does not, so some improvement is expected.

Transformer
-----------

Now we will try to replicate the baseline transformer results from
`"TVT: Two-View Transformer Network for Video Captioning" <http://proceedings.mlr.press/v95/chen18b.html>`_
on the MSVD (YouTube2Text) dataset. See Table 3, Base model(R).

In Section 4.3, the authors report most of their preprocessing and hyperparameters.

Create a folder called *yt2t_2*. Copy *youtube2text_iccv15* directory and *YouTubeClips.tar* into
the new directory and untar *YouTubeClips*. Rerun the renaming code. Subssample at 5 FPS using

..  code-block:: bash

    for fi in $( ls ); do ffmpeg -y -i $fi -r 5 $fi; done

Set the environment variable ``$YT2T`` to this new directory and change to the repo directory.
Run the feature extraction command again to extract ResNet features on the frames.
Then use this reprocessing code. Note that it shuffles the data differently, and it performs
tokenization similar to what the authors report.

.. code-block:: python

    import pickle
    import os
    import random
    import string

    seed = 2345
    random.seed(seed)


    YT = "youtube2text_iccv15"
    SHUFFLE = True

    with open(os.path.join(YT, "CAP.pkl"), "rb") as f:
        ann = pickle.load(f, encoding="latin-1")

    def clean(caption):
        caption = caption.lower()
        caption = caption.replace("\n", " ").replace("\t", " ").replace("\r", " ")
        # remove punctuation
        caption = caption.translate(str.maketrans("", "", string.punctuation))
        # multiple whitespace
        caption = " ".join(caption.split())
        return caption


    with open(os.path.join(YT, "train.pkl"), "rb") as f:
        train = pickle.load(f, encoding="latin-1")

    with open(os.path.join(YT, "valid.pkl"), "rb") as f:
        val = pickle.load(f, encoding="latin-1")

    with open(os.path.join(YT, "test.pkl"), "rb") as f:
        test = pickle.load(f, encoding="latin-1")

    train_data = []
    val_data = []
    test_data = []
    for vid_name, data in ann.items():
        vid_path = vid_name + ".npy"
        for i, d in enumerate(data):
            split_name = vid_name + "_" + str(i)
            datum = (vid_path, i, clean(d["caption"]))
            if split_name in train:
                train_data.append(datum)
            elif split_name in val:
                val_data.append(datum)
            elif split_name in test:
                test_data.append(datum)
            else:
                assert False

    if SHUFFLE:
        random.shuffle(train_data)

    train_files = open("yt2t_train_files.txt", "w")
    train_cap = open("yt2t_train_cap.txt", "w")

    for vid_path, _, an in train_data:
        train_files.write(vid_path + "\n")
        train_cap.write(an + "\n")

    train_files.close()
    train_cap.close()

    val_files = open("yt2t_val_files.txt", "w")
    val_folded = open("yt2t_val_folded_files.txt", "w")
    val_cap = open("yt2t_val_cap.txt", "w")

    for vid_path, i, an in val_data:
        if i == 0:
            val_folded.write(vid_path + "\n")
        val_files.write(vid_path + "\n")
        val_cap.write(an + "\n")

    val_files.close()
    val_folded.close()
    val_cap.close()

    test_files = open("yt2t_test_files.txt", "w")

    for vid_path, i, an in test_data:
        # Don't need to save out the test captions,
        # just the files. And, don't need to repeat
        # it for each caption
        if i == 0:
            test_files.write(vid_path + "\n")

    test_files.close()

Then preprocess the data with max-length filtering. (Note you will be prompted to remove the
old data. Do this, i.e. ``rm data/yt2t.*.pt.``)

.. code-block:: bash

    onmt_preprocess -data_type vec -train_src $YT2T/yt2t_train_files.txt -src_dir $YT2T/r152/ -train_tgt $YT2T/yt2t_train_cap.txt -valid_src $YT2T/yt2t_val_files.txt -valid_tgt $YT2T/yt2t_val_cap.txt -save_data data/yt2t --shard_size 1000 --src_seq_length 50 --tgt_seq_length 20

Delete the old checkpoints and train a transformer model on this data.

.. code-block:: bash

    rm -r yt2t-model_step_*.pt; onmt_train -data data/yt2t -save_model yt2t-model -world_size 2 -gpu_ranks 0 1 -model_type vec -batch_size 64 -train_steps 8000 -valid_steps 400 -save_checkpoint_steps 400 -optim adam -learning_rate .0001 -feat_vec_size 2048 -layers 4 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.3 -param_init 0 -param_init_glorot -report_every 400 --share_decoder_embedding --seed 7000

Note we use the hyperparameters described in the paper.
We estimate the length of 20 epochs with ``-train_steps``. Note that this depends on
using a world size of 2. If you use a different world size, scale the ``-train_steps`` (and
``-save_checkpoint_steps``, along with other parameters) accordingly.

The batch size is not specified in the paper, so we assume one checkpoint
per our estimated epoch. And, sharing
the decoder embeddings is not mentioned, although we find this helps performance. Like the paper, we perform
"early-stopping" with the COCO scores. We use beam search on the early stopping,
although this too is not mentioned. You can reproduce our early-stops with these scripts
(namely, running `find_val_stops.sh` and then `test_early_stops.sh` -
`process_results.py` is a dependency of `find_val_stops.sh`):

.. code-block:: python
    :caption: `process_results.py`

    import argparse

    from collections import defaultdict
    import pandas as pd


    def load_results(fname="results.txt"):
        index = []
        data = []
        with open(fname, "r") as f:
            while True:
                try:
                    filename = next(f).strip()
                except:
                    break
                step = int(filename.split("_")[-1].split(".")[0])
                next(f)  # blank
                next(f)  # spice junk
                next(f)  # length stats
                next(f)  # ratios
                scores = {}
                while True:
                    score_line = next(f).strip().strip("{").strip(",")
                    metric, score = score_line.split(": ")
                    metric = metric.strip("'")
                    score_num = float(score.strip("}").strip(","))
                    scores[metric] = float(score_num)
                    if score.endswith("}"):
                        break
                next(f)  # blank
                next(f)  # blank
                next(f)  # blank
                index.append(step)
                data.append(scores)
        df = pd.DataFrame(data, index=index)
        return df


    def find_absolute_stops(df):
        return df.idxmax()


    def find_early_stops(df, stop_count):
        maxes = defaultdict(lambda: 0)
        argmaxes = {}
        count_since_max = {}
        ended_metrics = set()
        for index, row in df.iterrows():
            for metric, score in row.items():
                if metric in ended_metrics:
                    continue
                if score >= maxes[metric]:
                    maxes[metric] = score
                    argmaxes[metric] = index
                    count_since_max[metric] = 0
                else:
                    count_since_max[metric] += 1
                    if count_since_max[metric] == stop_count:
                        ended_metrics.add(metric)
                        if len(ended_metrics) == len(row):
                            break
        return pd.Series(argmaxes)


    def find_stops(df, stop_count):
        if stop_count > 0:
            return find_early_stops(df, stop_count)
        else:
            return find_absolute_stops(df)


    if __name__ == "__main__":
        parser = argparse.ArgumentParser("Find locations of best scores")
        parser.add_argument(
            "-s", "--stop_count", type=int, default=0,
            help="Stop after this many scores worse than running max (0 to disable).")
        args = parser.parse_args()
        df = load_results()
        maxes = find_stops(df, args.stop_count)
        for metric, idx in maxes.iteritems():
            print(f"{metric} maxed @ {idx}")
            print(df.loc[idx])
            print()


.. code-block:: bash
    :caption: `find_val_stops.sh`

    rm results.txt
    touch results.txt
    for file in $( ls -1v yt2t-model_step*.pt )
    do
        echo $file
        onmt_translate -model $file -src $YT2T/yt2t_val_folded_files.txt -output pred.txt -verbose -data_type vec -src_dir $YT2T/r152 -gpu 0 -batch_size 16 -max_length 20 >/dev/null 2>/dev/null
        echo -e "$file\n" >> results.txt
        python coco.py -s val >> results.txt
        echo -e "\n\n" >> results.txt
    done
    python process_results.py -s 10 > val_stops.txt

.. code-block:: bash
    :caption: `test_early_stops.sh`

    rm test_results.txt
    touch test_results.txt
    while IFS='' read -r line || [[ -n "$line" ]]; do
        if [[ $line == *"maxed"* ]]; then
            metric=$(echo $line | awk '{print $1}')
        step=$(echo $line | awk '{print $NF}')
        echo $metric early stopped @ $step | tee -a test_results.txt
        onmt_translate -model "yt2t-model_step_${step}.pt" -src $YT2T/yt2t_test_files.txt -output pred.txt -data_type vec -src_dir $YT2T/r152 -gpu 0 -batch_size 16 -max_length 20 >/dev/null 2>/dev/null
        python coco.py -s 'test' >> test_results.txt
        echo -e "\n\n" >> test_results.txt
        fi
    done < val_stops.txt
    cat test_results.txt

Thus we test the checkpoint at step 2000 and find the following scores::

    Meteor early stopped @ 2000
    SPICE evaluation took: 2.522 s
    {'testlen': 3410, 'reflen': 3417, 'guess': [3410, 2740, 2070, 1400], 'correct': [2664, 1562, 887, 386]}
    ratio: 0.9979514193734276
    {'Bleu1': 0.7796296150773093,
     'Bleu2': 0.6659837622637965,
     'Bleu3': 0.5745524496015597,
     'Bleu4': 0.4779574102543823,
     'Cider': 0.7541600090591118,
     'Meteor': 0.3259497476899707,
     'Rouge': 0.6800279518634998,
     'Spice': 0.046435637924854}


Note our scores are an improvement over the recurrent approach.

The paper reports
BLEU4 50.25, CIDEr 72.11, METEOR 33.41, ROUGE 70.16.

The CIDEr score is higher than the paper (but, considering the sensitivity of this
metric, not by much), while the other metrics are slightly lower.
This could be indicative of an implementation difference. Note that Table 5 reports
24M parameters for a 2-layer transformer with ResNet inputs, while we find a few M less. This
could be due to generator or embedding differences, or perhaps linear layers on the
residual connections. Alternatively, the difference could be the initial tokenization.
The paper reports 9861 tokens, while we find fewer.

Part of this could be due to using
the annotations from the other repository, where perhaps some annotations have been
stripped. We also do not know the batch size or checkpoint frequency from the original
work.

Different random initializations could account for some of the difference, although
our random seed gives good results.

Overall, however, the scores are nearly reproduced
and the scores are favorable.
