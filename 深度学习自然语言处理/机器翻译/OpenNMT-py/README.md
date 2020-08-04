# 机器翻译竞赛-唱园杯-Pytorch代码-Baseline

几天前看到一个机器翻译竞赛，本来想要积累一下中英文翻译数据，后来发现是编码之后的数据...就没有然后了。所以就没有花太多时间去做这个东西，简单跑了一个baselines，线上分数大概是：。因为没想上分，就把代码分享给大家。

Baseline代码很简单，就是用OpenNMT-py这个库做的机器翻译，不过中文关于这块的资料很少，当初啃这个库也是一点点看的源代码，细节还挺多的。

默默吐槽一句代码组织架构有点乱，有些地方真的是让人摸不到头脑。。。

我也用这个文章做一个简单的OpenNMT-py的教程。如果是参加竞赛的话，后期可能需要修改源代码，所以建议大家不用安装的OpenNMT的库，而是直接下载源代码，方便修改。

首先，使用环境如下，大家照此下载就可以:

torchtext==0.4
OpenNMT-py==1.0
python==3.5
cuda==9.0

如果 OpenNMT-py==1.0 这个版本大家找不到，直接来我github上下载下来用就可以。

## 数据预处理

在 /data 目录下，需要包含四个文件，分别是 src-train.txt  src-val.txt  tgt-train.txt  tgt-val.txt

假如我们是中文翻译成英文，那么我们的 src-train.txt 和 src-val.txt 就是中文文件， tgt-train.txt  和 tgt-val.txt 就是英文文件。

其中文件内容格式为每行为一句文本，以空格进行分割。

对于唱园杯的数据，我们需要对其进行一些简单的修改，以满足上面的要求，我这边给出一个简单的处理代码，以字为单位，代码文件名称为「process_ori_data.py」：

```pyth9on
file=open('train_data.csv','r')
lines=file.readlines()

src_train=open('src-train.txt','w')
tgt_train=open('tgt-train.txt','w')
src_val=open('src-val.txt','w')
tgt_val=open('tgt-val.txt','w')

chinese_lists=[]
english_lists=[]
index=0
for line in lines:
    if index ==0:
        index+=1
        continue
    line=line.strip().split(',')
    chinese=line[1].strip().split('_')
    english=line[2].strip().split('_')
    chinese_lists.append(' '.join(chinese))
    english_lists.append(' '.join(english))
    index+=1
assert len(chinese_lists)==len(english_lists)
split_num=int(0.85*index)

for num in range(len(english_lists)):
    if num<=split_num:
        src_train.write(chinese_lists[num]+'\n')
        tgt_train.write(english_lists[num]+'\n')
    else:
        src_val.write(chinese_lists[num]+'\n')
        tgt_val.write(english_lists[num]+'\n')

src_train.close()
tgt_train.close()
src_val.close()
tgt_val.close()

```

在对原始数据进行处理之后，我们还需要进一步处理，代码如下：

```python
nohup python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/data  -src_seq_length 500 -tgt_seq_length 500 >preposs_datalog &
```

在这里需要提一个很重要的小细节，就是 src_seq_length 参数 和tgt_seq_length 参数的设定问题。默认这里是50。它的含义是如果句子长度小于50，不会被读入dataset！！！因为唱园杯的数据普遍比较长，所以你如果这里保持默认的话，会出现你只处理了一小部分原始数据的问题。

具体这个数值你设定为多少，看你自己具体情况。因为唱园杯在数据说明中说到已经去掉了特殊字符等，所以我就全部保留了。

## 模型进行预测

直接使用 Transformer 进行训练。Opennmt使用特定参数复现了 Transformer 的效果，这里我们直接套用就可以。

或者你可以使用全部数据双向预测，只需要训练一个模型就可以，不过我没有尝试，有兴趣的可以去试一试。

```python
nohup python train.py -data data/data -save_model data-model \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
        -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 \
        -world_size 4 -gpu_ranks 0 1 2 3  &
```

## 预测

在预测之前，我们需要看一下测试数据，发现是双向预测，所以我们需要将上面的数据颠倒过来再来一次，训练另一个模型即可。

预测代码如下：

```python
python  translate.py  -model demo-model_200000.pt -src data/src-test.txt -output pred.txt 
```

## 优化思路

因为是编码之后的数据，所有常规的优化没啥用，这里简单提两个：

1. 使用全部数据（训练数据和测试数据）训练Word2vec/Glove/Bert 等，然后作为输入，从而加入先验信息

2. 如果不想自己训练，可以使用词频对应到编码之后的数据，得到一个大致的结果，从而可以使用我们正常的word2vec/glove/bert
