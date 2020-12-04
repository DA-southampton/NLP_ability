ALBERT：更小更少但并不快

今天分享[ALBERT](https://arxiv.org/pdf/1909.11942.pdf, "ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS")，先说一个细节，同等规格的ALBERT和BERT相比，虽然ALBERT参数量少了，但是计算量并未降低，速度也并没有快多少。

举个形象的例子就是，（这个例子并不严谨，只是帮助理解）参数共享让它训练的时候把多层压缩为一层去训练，但是在预测的时候，我们需要再展开多层去进行预测。

这点需要注意。

主要掌握以下的几个知识点：

1. 词向量嵌入参数分解
2. 跨层参数分享
3. 取消NSP，使用SOP
4. 预训练的时候采用更满的数据/n-gram mask方式

# 1.Factorized embedding parameterization

词向量嵌入参数分解，简单说就是将词向量矩阵分解为了两个小矩阵，将隐藏层的大小和词汇矩阵的大小分离开。

在Bert中，词汇表embedding大小是$V*H$；

Albert 的参数分解是这样的，将这个矩阵分解为两个小矩阵：$V*E$和$E*H$

这样做有什么好处呢？

如果说，我觉得我的模型表达能力不够，我想要通过增大隐层H的大小来提升我们模型能力的表达能力，那么在提升H的时候，不仅仅隐层参数增多，词汇表的embedding矩阵维度也在增多，参数量也在增大。

矩阵分解之后，我们可以只是做到提升隐层大小，而不去改变表词汇表的大小。

# 2.cross-layer parameter sharing

跨层参数分享，这个操作可以防止参数随着网络层数的增大而增加。

![跨层参数共享](/Users/zida/Desktop/%E8%B7%A8%E5%B1%82%E5%8F%82%E6%95%B0%E5%85%B1%E4%BA%AB.png)

分为三种形式，只是共享attentions，只是共享FFN，全部共享。

共享的意思就是我这部分结构只使用同样的参数，在训练的时候只需要训练这一部分的参数就可以了。

看表格我们可以发现一个细节，就是只是共享FFN比只是共享attention的参数，模型效果要降低的多。

小声嘀咕一下，这是不是说明FFN比attention在信息表达上要重要啊。或者说attention在学习信息表达的时候。12层学习共性比较多。FFN学习到的差异性比较多。

# 3. sentence-order prediction (SOP) 

作者认为，NSP不必要。与MLM相比，NSP失效的主要原因是其缺乏任务难度。

NSP将主题预测和连贯性预测合并为一个单项任务

但是，与连贯性预测相比，主题预测更容易学习，并且与使用MLM损失学习的内容相比，重叠性更大。

对于ALBERT，作者使用了句子顺序预测（SOP）损失，它避免了主题预测，而是着重于句间建模。

其实就是预测句子顺序，正样本是顺着，负样本是颠倒过来。都是来自同一个文档。

![SOP](/Users/zida/Desktop/SOP.png)



# 其他细节

1. 数据格式：**Segments-Pair**

这个在RoBERTa中也有谈到，更长的序列长度可以提升性能。

2. Masked-ngram-LM

![Masked-ngram-LM](/Users/zida/Desktop/Masked-ngram-LM.png)

这就有点类似百度的ERINE和SpanBERT了

# 总结

总之，ALBERT并不如论文名称那样轻量级，只要版本规格小于xlarge，那么同一规格的ALBERT效果都是不如BERT的。

所以，同一规格的ALBERT和BERT预测速度是一样的，甚至真要较真的话，其实ALBERT应该更慢一些，因为ALBERT对Embedding层用了矩阵分解，这一步会带来额外的计算量，虽然这个计算量一般来说我们都感知不到。

总结一下可以学习的思路：

1. 预训练的时候，数据填充的更满，到512这种，有利于提升模型效果，这点在RoBERTa有谈到
2. mask n-gram有利于提升效果，这点类似百度的ERINE和SpanBERT了
3. 词向量矩阵分解能减少参数，但是也会降低性能
4. 跨层参数分享可以降低参数，也会降低性能，通过实验图知道，attention共享效果还好，FFN共享效果降低有点多
5. 取消NSP，使用SOP，正负样本来自同一个文档，但是顺序不同。



参考链接：

如何看待瘦身成功版BERT——ALBERT？ - 小莲子的回答 - 知乎 https://www.zhihu.com/question/347898375/answer/863537122

[用ALBERT和ELECTRA之前，请确认你真的了解它们](https://kexue.fm/archives/7846)