#!/usr/bin/env bash
# Author : Thamme Gowda
# Created : Nov 06, 2017

ONMT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

#======= EXPERIMENT SETUP ======
# Activate python environment if needed
source ~/.bashrc
# source activate py3

# update these variables
NAME="run1"
OUT="onmt-runs/$NAME"

DATA="$ONMT/onmt-runs/data"
TRAIN_SRC=$DATA/*train.src
TRAIN_TGT=$DATA/*train.tgt
VALID_SRC=$DATA/*dev.src
VALID_TGT=$DATA/*dev.tgt
TEST_SRC=$DATA/*test.src
TEST_TGT=$DATA/*test.tgt

BPE="" # default
BPE="src" # src, tgt, src+tgt

# applicable only when BPE="src" or "src+tgt"
BPE_SRC_OPS=10000

# applicable only when BPE="tgt" or "src+tgt"
BPE_TGT_OPS=10000

GPUARG="" # default
GPUARG="0"


#====== EXPERIMENT BEGIN ======

# Check if input exists
for f in $TRAIN_SRC $TRAIN_TGT $VALID_SRC $VALID_TGT $TEST_SRC $TEST_TGT; do
    if [[ ! -f "$f" ]]; then
        echo "Input File $f doesnt exist. Please fix the paths"
        exit 1
    fi
done

function lines_check {
    l1=`wc -l $1`
    l2=`wc -l $2`
    if [[ $l1 != $l2 ]]; then
        echo "ERROR: Record counts doesnt match between: $1 and $2"
        exit 2
    fi
}
lines_check $TRAIN_SRC $TRAIN_TGT
lines_check $VALID_SRC $VALID_TGT
lines_check $TEST_SRC $TEST_TGT


echo "Output dir = $OUT"
[ -d $OUT ] || mkdir -p $OUT
[ -d $OUT/data ] || mkdir -p $OUT/data
[ -d $OUT/models ] || mkdir $OUT/models
[ -d $OUT/test ] || mkdir -p  $OUT/test


echo "Step 1a: Preprocess inputs"
if [[ "$BPE" == *"src"* ]]; then
    echo "BPE on source"
    # Here we could use more  monolingual data
    $ONMT/tools/learn_bpe.py -s $BPE_SRC_OPS < $TRAIN_SRC > $OUT/data/bpe-codes.src

    $ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.src <  $TRAIN_SRC > $OUT/data/train.src
    $ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.src <  $VALID_SRC > $OUT/data/valid.src
    $ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.src <  $TEST_SRC > $OUT/data/test.src
else
    ln -sf $TRAIN_SRC $OUT/data/train.src
    ln -sf $VALID_SRC $OUT/data/valid.src
    ln -sf $TEST_SRC $OUT/data/test.src
fi


if [[ "$BPE" == *"tgt"* ]]; then
    echo "BPE on target"
    # Here we could use more  monolingual data
    $ONMT/tools/learn_bpe.py -s $BPE_SRC_OPS < $TRAIN_TGT > $OUT/data/bpe-codes.tgt

    $ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.tgt <  $TRAIN_TGT > $OUT/data/train.tgt
    $ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.tgt <  $VALID_TGT > $OUT/data/valid.tgt
    #$ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.tgt <  $TEST_TGT > $OUT/data/test.tgt
    # We dont touch the test References, No BPE on them!
    ln -sf $TEST_TGT $OUT/data/test.tgt
else
    ln -sf $TRAIN_TGT $OUT/data/train.tgt
    ln -sf $VALID_TGT $OUT/data/valid.tgt
    ln -sf $TEST_TGT $OUT/data/test.tgt
fi


#: <<EOF
echo "Step 1b: Preprocess"
python $ONMT/preprocess.py \
    -train_src $OUT/data/train.src \
    -train_tgt $OUT/data/train.tgt \
    -valid_src $OUT/data/valid.src \
    -valid_tgt $OUT/data/valid.tgt \
    -save_data $OUT/data/processed


echo "Step 2: Train"
GPU_OPTS=""
if [[ ! -z $GPUARG ]]; then
    GPU_OPTS="-gpu_ranks $GPUARG"
fi
CMD="python $ONMT/train.py -data $OUT/data/processed -save_model $OUT/models/$NAME $GPU_OPTS"
echo "Training command :: $CMD"
eval "$CMD"

#EOF

# select a model with high accuracy and low perplexity
# TODO: currently using linear scale, maybe not be the best
model=`ls $OUT/models/*.pt| awk -F '_' 'BEGIN{maxv=-1000000} {score=$(NF-3)-$(NF-1); if (score > maxv) {maxv=score; max=$0}}  END{ print max}'`
echo "Chosen Model = $model"
if [[ -z "$model" ]]; then
    echo "Model not found. Looked in $OUT/models/"
    exit 1
fi

GPU_OPTS=""
if [ ! -z $GPUARG ]; then
    GPU_OPTS="-gpu $GPUARG"
fi

echo "Step 3a: Translate Test"
python $ONMT/translate.py -model $model \
    -src $OUT/data/test.src \
    -output $OUT/test/test.out \
    -replace_unk  -verbose $GPU_OPTS > $OUT/test/test.log

echo "Step 3b: Translate Dev"
python $ONMT/translate.py -model $model \
    -src $OUT/data/valid.src \
    -output $OUT/test/valid.out \
    -replace_unk -verbose $GPU_OPTS > $OUT/test/valid.log

if [[ "$BPE" == *"tgt"* ]]; then
    echo "BPE decoding/detokenising target to match with references"
    mv $OUT/test/test.out{,.bpe}
    mv $OUT/test/valid.out{,.bpe} 
    cat $OUT/test/valid.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/valid.out
    cat $OUT/test/test.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test.out
fi

echo "Step 4a: Evaluate Test"
$ONMT/tools/multi-bleu-detok.perl $OUT/data/test.tgt < $OUT/test/test.out > $OUT/test/test.tc.bleu
$ONMT/tools/multi-bleu-detok.perl -lc $OUT/data/test.tgt < $OUT/test/test.out > $OUT/test/test.lc.bleu

echo "Step 4b: Evaluate Dev"
$ONMT/tools/multi-bleu-detok.perl $OUT/data/valid.tgt < $OUT/test/valid.out > $OUT/test/valid.tc.bleu
$ONMT/tools/multi-bleu-detok.perl -lc $OUT/data/valid.tgt < $OUT/test/valid.out > $OUT/test/valid.lc.bleu

#===== EXPERIMENT END ======
